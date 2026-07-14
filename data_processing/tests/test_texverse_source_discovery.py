from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import texverse_quality_audit  # noqa: E402
from export_texverse_clip import blender_export_script, sanitize_glb_import_source  # noqa: E402
from texverse_archive_utils import find_import_candidates  # noqa: E402


def write_sized_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def write_test_glb(path: Path, gltf: dict) -> None:
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_chunk = json_bytes + b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_chunk = b"\0\0\0\0"
    total_len = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, total_len)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(bin_chunk), 0x004E4942)
        + bin_chunk
    )


class TexVerseSourceDiscoveryTest(unittest.TestCase):
    def test_fbx_export_disables_recursive_missing_image_search(self) -> None:
        args = SimpleNamespace(
            source=Path("asset.fbx"),
            asset_id="asset",
            out_npz=Path("asset.npz"),
            out_json=Path("asset.json"),
            frames=40,
            min_vertices=1,
            max_joints=256,
            max_vertices=300000,
            max_faces=600000,
            bbox_ratio_hard_min=0.0,
            bbox_ratio_hard_max=0.0,
            min_motion_p95_bbox=0.0,
            topk_weights=8,
            active_skin_threshold=0.0,
            motion_fps_descriptor_vertices=1024,
        )

        script = blender_export_script(args, args.source, {})

        self.assertIn(
            "bpy.ops.import_scene.fbx(filepath=path, use_image_search=False)",
            script,
        )

    def test_glb_sanitizer_does_not_mutate_draco_attribute_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "draco.glb"
            write_test_glb(
                source,
                {
                    "asset": {"version": "2.0"},
                    "meshes": [
                        {
                            "primitives": [
                                {
                                    "attributes": {
                                        "POSITION": 0,
                                        "COLOR_0": 1,
                                        "JOINTS_0": 2,
                                        "WEIGHTS_0": 3,
                                    },
                                    "extensions": {
                                        "KHR_draco_mesh_compression": {
                                            "bufferView": 0,
                                            "attributes": {
                                                "POSITION": 0,
                                                "COLOR_0": 1,
                                                "JOINTS_0": 2,
                                                "WEIGHTS_0": 3,
                                            },
                                        }
                                    },
                                }
                            ]
                        }
                    ],
                },
            )

            import_source, meta = sanitize_glb_import_source(source, root)

            self.assertEqual(import_source, source)
            self.assertEqual(
                meta["glb_sanitization_skipped_draco_attributes"], ["COLOR_0"]
            )
            self.assertNotIn("glb_import_sanitized", meta)

    def test_audit_json_survives_blender_warning_interleaving(self) -> None:
        expected = {
            "asset_id": "asset",
            "status": "ok",
            "usable": True,
            "reject_reasons": [],
        }
        stdout = (
            "Dependency cycle via "
            + texverse_quality_audit.AUDIT_OUTPUT_MARKER
            + '{"asset_id":"asset","status":"ok","usable":true,"reject_reasons":[]}'
            + " trailing dependency warning\nBlender quit\n"
        )

        self.assertEqual(texverse_quality_audit.parse_blender_audit_output(stdout), expected)

    def test_candidate_ranking_ignores_macos_sidecars_and_prefers_skin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sized_file(root / "animations" / "Animation_Walk_without_skin.fbx", 400)
            write_sized_file(root / "animations" / "Animation_Run_without_skin.fbx", 300)
            write_sized_file(root / "Animation_penguin_walk_withSkin.fbx", 100)
            write_sized_file(root / "model.fbx", 200)
            write_sized_file(root / "__MACOSX" / "._Animation_penguin_walk_withSkin.fbx", 500)

            candidates, count = find_import_candidates(root, 2)

            self.assertEqual(count, 4)
            self.assertEqual(candidates[0].name, "Animation_penguin_walk_withSkin.fbx")
            self.assertEqual(len(candidates), 2)
            self.assertFalse(any("__MACOSX" in candidate.parts for candidate in candidates))

            all_candidates, all_count = find_import_candidates(root, 0)
            self.assertEqual(all_count, 4)
            self.assertEqual(len(all_candidates), 4)

    def test_audit_extends_search_before_rejecting_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "asset.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("a.fbx", b"a" * 40)
                handle.writestr("b.fbx", b"b" * 30)
                handle.writestr("valid.fbx", b"c" * 20)
                handle.writestr("never_needed.fbx", b"d" * 10)

            args = SimpleNamespace(
                blender="blender",
                download_dir=root,
                redownload=False,
                extract_root=root / "extract",
                no_expand_nested_archives=False,
                nested_archive_depth=2,
                max_nested_archives=12,
                max_nested_archive_mb=0.0,
                nested_archive_timeout_sec=120,
                max_candidates_per_zip=2,
                keep_extracted=True,
            )

            def fake_audit(_blender: str, candidate: Path, asset_id: str, _args: object) -> dict:
                usable = candidate.name == "valid.fbx"
                return {
                    "asset_id": asset_id,
                    "candidate": str(candidate),
                    "status": "ok",
                    "usable": usable,
                    "reject_reasons": [] if usable else ["test_reject"],
                }

            zip_record = texverse_quality_audit.ZipRecord(
                asset_id="asset",
                zip_path=str(archive),
                source="download",
                status="ok",
            )
            with (
                mock.patch.object(texverse_quality_audit, "download_zip", return_value=zip_record),
                mock.patch.object(texverse_quality_audit, "expand_nested_archives", return_value=[]),
                mock.patch.object(texverse_quality_audit, "run_blender_audit", side_effect=fake_audit),
            ):
                _, records, usable = texverse_quality_audit.audit_one_asset(
                    "asset", "asset.zip", args
                )

            self.assertTrue(usable)
            self.assertEqual([Path(row["candidate"]).name for row in records], ["a.fbx", "b.fbx", "valid.fbx"])
            self.assertTrue(all(row["candidate_count"] == 4 for row in records))
            self.assertTrue(all(row["candidates_imported"] == 3 for row in records))
            self.assertTrue(all(row["candidate_search_extended"] for row in records))


if __name__ == "__main__":
    unittest.main()
