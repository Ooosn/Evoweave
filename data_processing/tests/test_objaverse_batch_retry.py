from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import export_objaverse_xl_pass1_batch as batch  # noqa: E402


def make_args(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        out_root=root / "out",
        blender="blender",
        blender_threads=2,
        frames=40,
        motion_fps_descriptor_vertices=1024,
        timeout_sec=360,
        timeout_retries=1,
        max_joints=256,
        min_vertices=1,
        max_vertices=300000,
        max_faces=600000,
        bbox_ratio_hard_min=0.0,
        bbox_ratio_hard_max=0.0,
        min_motion_p95_bbox=0.0,
        active_skin_threshold=0.0,
    )


class ObjaverseBatchRetryTest(unittest.TestCase):
    def test_timeout_is_retried_before_manifest_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "asset.fbx"
            source.write_bytes(b"fbx")
            args = make_args(root)
            out_json = args.out_root / "json" / "asset_seq0.json"
            out_npz = args.out_root / "npz" / "asset_seq0.npz"
            calls = 0

            def fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
                nonlocal calls
                calls += 1
                out_json.parent.mkdir(parents=True, exist_ok=True)
                if calls == 1:
                    out_json.write_text(
                        json.dumps(
                            {
                                "asset_id": "asset",
                                "status": "reject",
                                "reject_reasons": ["blender_export_timeout"],
                            }
                        ),
                        encoding="utf-8",
                    )
                    return SimpleNamespace(returncode=2, stdout="timed out")

                out_npz.parent.mkdir(parents=True, exist_ok=True)
                out_npz.write_bytes(b"npz")
                out_json.write_text(
                    json.dumps({"asset_id": "asset", "status": "clean"}),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="clean")

            with mock.patch.object(batch.subprocess, "run", side_effect=fake_run):
                row = {"asset_id": "asset", "path": str(source)}
                initial_record = batch.run_one(row, args)
                self.assertEqual(calls, 1)
                self.assertEqual(initial_record["status"], "reject")
                record = batch.retry_serially(row, args, initial_record)

            self.assertEqual(calls, 2)
            self.assertEqual(record["status"], "clean")
            self.assertEqual(record["batch_attempts"], 2)
            self.assertEqual(record["transient_attempt_reasons"], ["blender_export_timeout"])
            self.assertEqual(record["npz_path"], str(out_npz))

    def test_parent_timeout_discards_partial_output_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "asset.fbx"
            source.write_bytes(b"fbx")
            args = make_args(root)
            out_json = args.out_root / "json" / "asset_seq0.json"
            out_npz = args.out_root / "npz" / "asset_seq0.npz"
            calls = 0

            def fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
                nonlocal calls
                calls += 1
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_npz.parent.mkdir(parents=True, exist_ok=True)
                if calls == 1:
                    out_json.write_text(
                        json.dumps({"asset_id": "asset", "status": "clean"}),
                        encoding="utf-8",
                    )
                    out_npz.write_bytes(b"partial")
                    raise batch.subprocess.TimeoutExpired(cmd="blender", timeout=390)

                self.assertFalse(out_json.exists())
                self.assertFalse(out_npz.exists())
                out_npz.write_bytes(b"complete")
                out_json.write_text(
                    json.dumps({"asset_id": "asset", "status": "clean"}),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="clean")

            row = {"asset_id": "asset", "path": str(source)}
            with mock.patch.object(batch.subprocess, "run", side_effect=fake_run):
                initial_record = batch.run_one(row, args)
                self.assertEqual(initial_record["reject_reasons"], ["batch_export_timeout"])
                self.assertFalse(out_json.exists())
                self.assertFalse(out_npz.exists())
                record = batch.retry_serially(row, args, initial_record)

            self.assertEqual(calls, 2)
            self.assertEqual(record["status"], "clean")
            self.assertEqual(record["batch_attempts"], 2)
            self.assertEqual(record["transient_attempt_reasons"], ["batch_export_timeout"])
            self.assertEqual(out_npz.read_bytes(), b"complete")

    def test_resume_manifest_removes_only_transient_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            rows = [
                {"asset_id": "clean", "status": "clean"},
                {
                    "asset_id": "retry",
                    "status": "reject",
                    "reject_reasons": ["blender_export_timeout"],
                },
                {
                    "asset_id": "data_reject",
                    "status": "reject",
                    "reject_reasons": ["no_armature"],
                },
                {"asset_id": "clean", "status": "clean", "batch_attempts": 1},
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            done, retry_asset_ids = batch.prepare_manifest_for_resume(manifest)
            retained = batch.read_jsonl(manifest)

            self.assertEqual(done, {"clean", "data_reject"})
            self.assertEqual(retry_asset_ids, ["retry"])
            self.assertEqual([row["asset_id"] for row in retained], ["clean", "data_reject"])
            self.assertEqual(retained[0]["batch_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
