from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from structure_xyz_fill_runtime import (  # noqa: E402
    FAMILY,
    make_stream,
    optimizer_for,
    restore_sampling_state,
    sha256,
)
from train_structure_xyz_fill import (  # noqa: E402
    optimizer_parameter_names,
    read_reviewed,
    restore_continuation,
    training_contract,
)


DATASET = range(5000)


def _config() -> dict:
    path = Path(__file__).resolve().parents[2] / "experiments/structure_xyz_fill_random_512_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _consume_to_assets(stream, assets: int) -> None:
    target_cursor = assets // stream.world_size
    while stream.cursor < target_cursor:
        stream.take()
    assert stream.cursor == target_cursor


def _checkpoint_counters(stream, assets: int) -> tuple[int, int]:
    step = int(np.searchsorted(stream.global_ends, assets, side="right"))
    frames = int(np.dot(
        stream.update_sizes[:step].astype(np.int64),
        stream.frame_counts[:step].astype(np.int64),
    ))
    return step, frames


def _completed_stream(config: dict):
    stream = make_stream(config, DATASET, rank=0, world=2)
    _consume_to_assets(stream, int(config["training"]["target_assets"]))
    return stream


def _expanded_config(config: dict, target_assets: int) -> dict:
    expanded = deepcopy(config)
    milestones = [128, 256, 512, 1024, target_assets]
    expanded["training"].update(
        target_assets=target_assets,
        evaluation_assets=milestones,
        checkpoint_assets=milestones,
    )
    return expanded


def test_partial_checkpoint_replays_exact_next_requests_and_frame_plan() -> None:
    config = _config()
    uninterrupted = make_stream(config, DATASET, rank=0, world=2)
    _consume_to_assets(uninterrupted, 128)
    state = uninterrupted.state_dict()
    step, frames = _checkpoint_counters(uninterrupted, 128)

    restored = restore_sampling_state(
        config,
        DATASET,
        state,
        asset_exposures=128,
        supervised_frame_exposures=frames,
        step=step,
        rank=0,
        world=2,
    )
    assert restored.cursor == uninterrupted.cursor
    assert np.array_equal(restored.frame_counts, uninterrupted.frame_counts)
    assert np.array_equal(restored.order, uninterrupted.order)

    while uninterrupted.cursor < uninterrupted.local_total:
        assert restored.take() == uninterrupted.take()
    assert restored.cursor == uninterrupted.cursor


def test_completed_checkpoint_extends_to_2048_with_prefix_preserved() -> None:
    old_config = _config()
    old_stream = _completed_stream(old_config)
    state = old_stream.state_dict()
    step, frames = _checkpoint_counters(old_stream, 512)
    new_config = _expanded_config(old_config, 2048)

    restored = restore_sampling_state(
        new_config,
        DATASET,
        state,
        asset_exposures=512,
        supervised_frame_exposures=frames,
        step=step,
        rank=0,
        world=2,
    )
    uninterrupted = make_stream(new_config, DATASET, rank=0, world=2)
    uninterrupted.cursor = old_stream.cursor

    assert restored.target_assets == 2048
    assert restored.cursor == old_stream.cursor
    assert np.array_equal(restored.order[:512], old_stream.order)
    assert np.array_equal(restored.frame_counts[:len(old_stream.frame_counts)], old_stream.frame_counts)
    for _ in range(3):
        assert restored.take() == uninterrupted.take()


def test_completed_checkpoint_same_budget_is_valid_for_final_evaluation_recovery() -> None:
    config = _config()
    completed = _completed_stream(config)
    step, frames = _checkpoint_counters(completed, 512)

    restored = restore_sampling_state(
        config,
        DATASET,
        completed.state_dict(),
        asset_exposures=512,
        supervised_frame_exposures=frames,
        step=step,
        rank=0,
        world=2,
    )
    assert restored.cursor == restored.local_total
    with pytest.raises(StopIteration):
        restored.take()


def test_restore_rejects_wrong_cursor_and_counters() -> None:
    config = _config()
    stream = make_stream(config, DATASET, rank=0, world=2)
    _consume_to_assets(stream, 128)
    state = stream.state_dict()
    step, frames = _checkpoint_counters(stream, 128)
    cases = (
        (dict(state, cursor=state["cursor"] + 1), 128, frames, step),
        (state, 127, frames, step),
        (state, 128, frames + 1, step),
        (state, 128, frames, step + 1),
    )
    for bad_state, assets, bad_frames, bad_step in cases:
        with pytest.raises(ValueError):
            restore_sampling_state(
                config,
                DATASET,
                bad_state,
                asset_exposures=assets,
                supervised_frame_exposures=bad_frames,
                step=bad_step,
                rank=0,
                world=2,
            )


def test_restore_rejects_unfinished_extension_and_budget_shrink() -> None:
    config = _config()
    partial = make_stream(config, DATASET, rank=0, world=2)
    _consume_to_assets(partial, 128)
    state = partial.state_dict()
    step, frames = _checkpoint_counters(partial, 128)
    with pytest.raises(ValueError):
        restore_sampling_state(
            _expanded_config(config, 2048),
            DATASET,
            state,
            asset_exposures=128,
            supervised_frame_exposures=frames,
            step=step,
            rank=0,
            world=2,
        )

    completed = _completed_stream(config)
    step, frames = _checkpoint_counters(completed, 512)
    smaller = deepcopy(config)
    smaller["training"].update(target_assets=256, evaluation_assets=[128, 256], checkpoint_assets=[128, 256])
    with pytest.raises(ValueError):
        restore_sampling_state(
            smaller,
            DATASET,
            completed.state_dict(),
            asset_exposures=512,
            supervised_frame_exposures=frames,
            step=step,
            rank=0,
            world=2,
        )


def test_read_reviewed_requires_matching_receipt_provenance(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    expected = {"code_commit": "commit-123", "config_sha256": "config-456"}
    payload = {"passed": True, **expected}
    path.write_text(json.dumps(payload), encoding="utf-8")
    spec = {"path": str(path), "sha256": sha256(path), **expected}

    assert read_reviewed(spec, name="checkpoint test") == payload

    for field in expected:
        mismatched_spec = dict(spec, **{field: "wrong"})
        with pytest.raises(ValueError):
            read_reviewed(mismatched_spec, name="checkpoint test")

    payload["code_commit"] = "different-commit"
    path.write_text(json.dumps(payload), encoding="utf-8")
    mismatched_artifact = dict(spec, sha256=sha256(path))
    with pytest.raises(ValueError):
        read_reviewed(mismatched_artifact, name="checkpoint test")


class _TinyFourModuleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in ("surface_tokenizer", "motion_encoder", "structure_decoder", "fill_decoder"):
            setattr(self, name, nn.Linear(2, 2))

    def configuration(self) -> dict[str, str]:
        return {"model": "tiny-four-module-v1"}


def _optimizer_state_by_name(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return {
        names[id(parameter)]: {
            name: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for name, value in state.items()
        }
        for parameter, state in optimizer.state.items()
    }


def _assert_optimizer_states_equal(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for parameter_name in actual:
        assert actual[parameter_name].keys() == expected[parameter_name].keys()
        for state_name, value in actual[parameter_name].items():
            reference = expected[parameter_name][state_name]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, reference)
            else:
                assert value == reference


def test_restore_continuation_restores_cpu_model_optimizer_rng_and_next_request(tmp_path: Path) -> None:
    config = _config()
    config["runtime"].update(world_size=1, physical_gpus=[0])
    config["training"]["resume_checkpoint"] = None
    source_model = _TinyFourModuleModel()
    source_optimizer = optimizer_for(source_model, config)
    for _ in range(3):
        source_optimizer.zero_grad(set_to_none=True)
        loss = sum(parameter.square().sum() for parameter in source_model.parameters())
        loss.backward()
        source_optimizer.step()

    source_stream = make_stream(config, DATASET, rank=0, world=1)
    _consume_to_assets(source_stream, 128)
    step, frames = _checkpoint_counters(source_stream, 128)
    assert step == 3
    assert source_stream.global_ends[:3].tolist() == [48, 96, 128]

    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    torch.set_rng_state(torch_state)
    expected_torch = torch.rand(4)
    np.random.set_state(numpy_state)
    expected_numpy = np.random.random(4)
    random.setstate(python_state)
    expected_python = [random.random() for _ in range(4)]
    torch.manual_seed(12345)
    np.random.seed(12345)
    random.seed(12345)

    cuda_state = torch.tensor([3, 1, 4, 1], dtype=torch.uint8)
    payload = {
        "schema": FAMILY,
        "mode": "train",
        "model": source_model.state_dict(),
        "optimizer": source_optimizer.state_dict(),
        "optimizer_parameter_names": optimizer_parameter_names(source_model, source_optimizer),
        "rank_states": [{
            "stream": source_stream.state_dict(),
            "torch_rng": torch_state,
            "cuda_rng": cuda_state,
            "numpy_rng": numpy_state,
            "python_rng": python_state,
        }],
        "training_contract": training_contract(config),
        "architecture": source_model.configuration(),
        "asset_exposures": 128,
        "supervised_frame_exposures": frames,
        "step": 3,
        "initialization": {"test": "tiny-four-module"},
        "source": "cpu-test",
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)
    config["training"]["resume_checkpoint"] = {"path": str(checkpoint), "sha256": sha256(checkpoint)}

    restored_model = _TinyFourModuleModel()
    restored_optimizer = optimizer_for(restored_model, config)
    with patch("torch.cuda.set_rng_state") as set_cuda_rng:
        restored_stream, assets, restored_frames, restored_step, initialization = restore_continuation(
            restored_model,
            restored_optimizer,
            DATASET,
            config,
            rank=0,
            world=1,
            device=torch.device("cpu"),
        )

    assert set_cuda_rng.call_count == 1
    torch.testing.assert_close(torch.rand(4), expected_torch)
    np.testing.assert_array_equal(np.random.random(4), expected_numpy)
    assert [random.random() for _ in range(4)] == expected_python
    for name, value in source_model.state_dict().items():
        torch.testing.assert_close(restored_model.state_dict()[name], value)
    _assert_optimizer_states_equal(
        _optimizer_state_by_name(restored_model, restored_optimizer),
        _optimizer_state_by_name(source_model, source_optimizer),
    )
    assert (assets, restored_frames, restored_step) == (128, frames, 3)
    assert initialization == payload["initialization"]
    assert restored_stream.take() == source_stream.take()

    bad_name_order = deepcopy(payload)
    bad_name_order["optimizer_parameter_names"][0]["parameters"][0] = "wrong.parameter"
    bad_step = deepcopy(payload)
    first_state = next(iter(bad_step["optimizer"]["state"].values()))
    first_state["step"] = torch.tensor(2.0)
    bad_moment_shape = deepcopy(payload)
    first_state = next(iter(bad_moment_shape["optimizer"]["state"].values()))
    first_state["exp_avg"] = torch.zeros(1)
    for suffix, bad_payload in (
        ("bad-name-order", bad_name_order),
        ("bad-step", bad_step),
        ("bad-moment-shape", bad_moment_shape),
    ):
        bad_checkpoint = tmp_path / f"{suffix}.pt"
        torch.save(bad_payload, bad_checkpoint)
        bad_config = deepcopy(config)
        bad_config["training"]["resume_checkpoint"] = {
            "path": str(bad_checkpoint),
            "sha256": sha256(bad_checkpoint),
        }
        bad_model = _TinyFourModuleModel()
        bad_optimizer = optimizer_for(bad_model, bad_config)
        with pytest.raises(ValueError):
            with patch("torch.cuda.set_rng_state"):
                restore_continuation(
                    bad_model,
                    bad_optimizer,
                    DATASET,
                    bad_config,
                    rank=0,
                    world=1,
                    device=torch.device("cpu"),
                )
