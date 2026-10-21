from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from structure_xyz_fill_initialization import (
    load_official_surface_parameters,
    official_text_positions,
    validate_initialization_policy,
)


POSITION_KEY = "transformer.model.decoder.embed_positions.weight"
SURFACE_KEYS = (
    "mesh_encoder.weight",
    "mesh_encoder.bias",
    "output_proj.weight",
    "output_proj.bias",
)


def _policy(fill_kind: str = "random") -> dict:
    return {
        "common_initialization": {
            "kind": "official_surface_fresh_motion_structure",
            "trained_experiment_weights_allowed": False,
            "optimizer_restored": False,
        },
        "fill_initialization": {"kind": fill_kind, "seed": 20261022},
        "initialization": {
            "checkpoint": "/official/model.ckpt",
            "official_checkpoint": "/official/model.ckpt",
            "seed": 20261013,
        },
    }


@pytest.mark.parametrize("fill_kind", ["random", "official_text_positions"])
def test_official_surface_policy_accepts_only_declared_initialization(fill_kind: str) -> None:
    config = _policy(fill_kind)
    before = deepcopy(config)
    assert validate_initialization_policy(config) is None
    assert config == before


@pytest.mark.parametrize("section,kind", [
    ("common_initialization", "accepted_b65_shared_and_structure"),
    ("fill_initialization", "structure_and_position"),
    ("fill_initialization", "unknown"),
])
def test_policy_rejects_old_experiment_and_unknown_initialization(section: str, kind: str) -> None:
    config = _policy()
    config[section]["kind"] = kind
    with pytest.raises(ValueError):
        validate_initialization_policy(config)


@pytest.mark.parametrize("extra", ["checkpoint", "sha256", "inherited_updates", "unexpected"])
def test_policy_rejects_any_extra_common_field(extra: str) -> None:
    config = _policy()
    config["common_initialization"][extra] = "forbidden"
    with pytest.raises(ValueError):
        validate_initialization_policy(config)


@pytest.mark.parametrize("field", ["kind", "trained_experiment_weights_allowed", "optimizer_restored"])
def test_policy_requires_complete_common_contract(field: str) -> None:
    config = _policy()
    del config["common_initialization"][field]
    with pytest.raises(ValueError):
        validate_initialization_policy(config)


@pytest.mark.parametrize("field", ["trained_experiment_weights_allowed", "optimizer_restored"])
@pytest.mark.parametrize("value", [True, 0, "false", None])
def test_policy_flags_must_be_literal_false(field: str, value: object) -> None:
    config = _policy()
    config["common_initialization"][field] = value
    with pytest.raises(ValueError):
        validate_initialization_policy(config)


def test_policy_rejects_checkpoint_that_is_not_declared_official_checkpoint() -> None:
    config = _policy()
    config["initialization"]["checkpoint"] = "/old-ab/checkpoint_asset_065000.pt"
    with pytest.raises(ValueError):
        validate_initialization_policy(config)


@pytest.mark.parametrize("section", ["initialization", "fill_initialization"])
def test_policy_requires_explicit_seed(section: str) -> None:
    config = _policy()
    del config[section]["seed"]
    with pytest.raises(ValueError):
        validate_initialization_policy(config)


@pytest.mark.parametrize("section", ["initialization", "fill_initialization"])
@pytest.mark.parametrize("seed", [True, False, 20261013.0, "20261013", None])
def test_policy_rejects_noninteger_seed(section: str, seed: object) -> None:
    config = _policy()
    config[section]["seed"] = seed
    with pytest.raises(ValueError):
        validate_initialization_policy(config)


def _surface_fixture():
    surface = nn.Linear(2, 3)
    projection = nn.Linear(3, 2)
    state = {
        "mesh_encoder.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "mesh_encoder.bias": torch.tensor([10., 11., 12.]),
        "output_proj.weight": torch.arange(20., 26.).reshape(2, 3),
        "output_proj.bias": torch.tensor([30., 31.]),
        "transformer.model.decoder.layers.0.self_attn.q_proj.weight": torch.full((2, 2), 41.),
        "transformer.model.decoder.layers.0.fc1.weight": torch.full((2, 2), 42.),
        "transformer.lm_head.weight": torch.full((2, 2), 43.),
    }
    return surface, projection, state


def test_surface_load_is_exact_and_leaves_transformer_state_untouched() -> None:
    surface, projection, state = _surface_fixture()
    before = {key: value.clone() for key, value in state.items()}
    receipt = load_official_surface_parameters(surface, projection, state)

    assert isinstance(receipt, dict)
    for prefix, module in (("mesh_encoder", surface), ("output_proj", projection)):
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, before[f"{prefix}.{name}"], rtol=0, atol=0)
    assert state.keys() == before.keys()
    for key in state:
        torch.testing.assert_close(state[key], before[key], rtol=0, atol=0)


@pytest.mark.parametrize("key", SURFACE_KEYS)
def test_surface_load_rejects_missing_parameter(key: str) -> None:
    surface, projection, state = _surface_fixture()
    del state[key]
    with pytest.raises((ValueError, RuntimeError)):
        load_official_surface_parameters(surface, projection, state)


@pytest.mark.parametrize("prefix", ["mesh_encoder", "output_proj"])
def test_surface_load_rejects_extra_parameter_in_owned_prefix(prefix: str) -> None:
    surface, projection, state = _surface_fixture()
    state[f"{prefix}.unexpected"] = torch.zeros(1)
    with pytest.raises((ValueError, RuntimeError)):
        load_official_surface_parameters(surface, projection, state)


@pytest.mark.parametrize("key", SURFACE_KEYS)
def test_surface_load_rejects_wrong_parameter_shape(key: str) -> None:
    surface, projection, state = _surface_fixture()
    state[key] = torch.zeros(state[key].numel() + 1)
    with pytest.raises((ValueError, RuntimeError)):
        load_official_surface_parameters(surface, projection, state)


@pytest.mark.parametrize("key", SURFACE_KEYS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_surface_load_rejects_nonfinite_parameter(key: str, value: float) -> None:
    surface, projection, state = _surface_fixture()
    state[key].reshape(-1)[-1] = value
    with pytest.raises((ValueError, RuntimeError)):
        load_official_surface_parameters(surface, projection, state)


def test_official_positions_preserve_3078_rows_with_2052_after_offset_1026() -> None:
    table = torch.arange(3078 * 4, dtype=torch.float32).reshape(3078, 4)
    _, _, state = _surface_fixture()
    state[POSITION_KEY] = table
    before = {key: value.clone() for key, value in state.items()}

    positions, row_offset = official_text_positions(
        state, dim=4, query_tokens=1024, position_capacity=3076,
    )
    assert isinstance(positions, torch.Tensor)
    assert row_offset == 1026
    assert positions.shape == (3078, 4)
    assert positions[row_offset:].shape == (2052, 4)
    torch.testing.assert_close(positions, table, rtol=0, atol=0)
    torch.testing.assert_close(positions[row_offset], table[1026], rtol=0, atol=0)
    assert state.keys() == before.keys()
    for key in state:
        torch.testing.assert_close(state[key], before[key], rtol=0, atol=0)


def test_official_positions_require_exact_checkpoint_key() -> None:
    state = {"model." + POSITION_KEY: torch.zeros(3078, 4)}
    with pytest.raises((ValueError, KeyError)):
        official_text_positions(state, dim=4, query_tokens=1024, position_capacity=3076)


@pytest.mark.parametrize("shape", [(3077, 4), (3079, 4), (3078, 3), (3078,), (1, 3078, 4)])
def test_official_positions_reject_boundary_and_rank_shape_errors(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        official_text_positions(
            {POSITION_KEY: torch.zeros(shape)}, dim=4, query_tokens=1024, position_capacity=3076,
        )


@pytest.mark.parametrize("row", [0, 1026, 3077])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_official_positions_require_finite_values_in_every_row(row: int, value: float) -> None:
    table = torch.zeros(3078, 4)
    table[row, -1] = value
    with pytest.raises(ValueError):
        official_text_positions(
            {POSITION_KEY: table}, dim=4, query_tokens=1024, position_capacity=3076,
        )
