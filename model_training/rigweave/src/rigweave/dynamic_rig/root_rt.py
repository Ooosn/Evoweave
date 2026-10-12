"""Manifest-bound root RT preprocessing. This cache requires source skeleton GT.

Only proper rotations and translations are removed; scale/shear residuals stay
in the points. No mesh-only inference equivalence is implied by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np


VERSION = "retained_root_rigid_v1"
NORMALIZATION = "input_sequence_mesh_bbox"
PADDING = 1.1
RIGID_ATOL = 1.0e-6
BATCH_KEYS = (
    "root_rt_rotations", "root_rt_translations", "root_rt_augmentation",
    "root_rt_root_index", "root_rt_relative_nonrigid",
)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"root_rt {name} must be an integer >= {minimum}")
    return int(value)


def validate_descriptor(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if "root_rt" not in row:
        return None
    spec = row["root_rt"]
    if not isinstance(spec, Mapping):
        raise ValueError("root_rt must be an object, not null")
    required = {"version", "cache_path", "index", "frame_count", "reference_index", "normalization", "padding"}
    if not required.issubset(spec):
        raise ValueError(f"root_rt missing fields: {sorted(required - set(spec))}")
    if spec["version"] != VERSION or spec["normalization"] != NORMALIZATION:
        raise ValueError("root_rt unsupported version or normalization")
    padding = spec["padding"]
    if isinstance(padding, bool) or not isinstance(padding, Real) or float(padding) != PADDING:
        raise ValueError("root_rt requires padding=1.1")
    if _integer(spec["reference_index"], "reference_index") != 0:
        raise ValueError("root_rt reference_index must be 0")
    _integer(spec["index"], "index")
    _integer(spec["frame_count"], "frame_count", 1)
    if not isinstance(row.get("path"), str) or not Path(row["path"]).is_absolute():
        raise ValueError("root_rt requires the original absolute manifest path string")
    if not isinstance(spec["cache_path"], str) or not Path(spec["cache_path"]).is_absolute():
        raise ValueError("root_rt cache_path must be absolute")
    return spec


def _proper_rotations(rotations: np.ndarray, name: str) -> None:
    if rotations.shape[-2:] != (3, 3) or not np.isfinite(rotations).all():
        raise ValueError(f"{name} must contain finite 3x3 rotations")
    gram = np.swapaxes(rotations, -1, -2) @ rotations
    if np.max(np.abs(gram - np.eye(3))) > RIGID_ATOL:
        raise ValueError(f"{name} orthogonality error exceeds {RIGID_ATOL}")
    if np.max(np.abs(np.linalg.det(rotations) - 1.0)) > RIGID_ATOL:
        raise ValueError(f"{name} determinant is not +1")


def _stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@lru_cache(maxsize=2)
def _load_shard(path_string: str, stamp: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    path = Path(path_string)
    required = ("schema_version", "rotations", "translations", "frame_counts", "path_hashes",
                "frame_numbers", "root_indices", "relative_nonrigid")
    with np.load(path, allow_pickle=False) as archive:
        if not set(required).issubset(archive.files):
            raise ValueError("root_rt shard missing required arrays")
        arrays = {key: archive[key] for key in required}
    if _stamp(path) != stamp:
        raise ValueError("root_rt shard changed while loading")
    schema = arrays["schema_version"]
    if schema.shape != () or schema.dtype.kind != "U" or schema.item() != VERSION:
        raise ValueError("root_rt shard schema_version mismatch")
    rotations, translations, counts = (arrays[key] for key in ("rotations", "translations", "frame_counts"))
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3) or min(rotations.shape[:2]) < 1:
        raise ValueError("root_rt rotations must have shape [N,maxT,3,3]")
    n, frames = rotations.shape[:2]
    expected = {
        "rotations": ((n, frames, 3, 3), np.dtype("float64")),
        "translations": ((n, frames, 3), np.dtype("float64")),
        "frame_counts": ((n,), np.dtype("int32")),
        "path_hashes": ((n,), np.dtype("U64")),
        "frame_numbers": ((n, frames), np.dtype("int64")),
        "root_indices": ((n,), np.dtype("int32")),
        "relative_nonrigid": ((n,), np.dtype("bool")),
    }
    for key, (shape, dtype) in expected.items():
        if arrays[key].shape != shape or arrays[key].dtype != dtype:
            raise ValueError(f"root_rt {key} shape/dtype mismatch; expected {shape} {dtype}")
    if np.any(counts < 1) or np.any(counts > frames) or np.any(arrays["root_indices"] < 0):
        raise ValueError("root_rt invalid frame_counts or root_indices")
    if not np.isfinite(rotations).all() or not np.isfinite(translations).all():
        raise ValueError("root_rt nonfinite rotation/translation")
    for index, count in enumerate(counts):
        _proper_rotations(rotations[index, :count], "root_rt rotations")
        if np.any(arrays["frame_numbers"][index, count:] != -1):
            raise ValueError("root_rt frame_numbers padding must be -1")
        if not np.allclose(rotations[index, 0], np.eye(3), atol=RIGID_ATOL, rtol=0) or not np.allclose(
            translations[index, 0], 0.0, atol=RIGID_ATOL, rtol=0
        ):
            raise ValueError("root_rt reference frame must have identity RT")
    for digest in arrays["path_hashes"]:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("root_rt invalid path_hashes")
    for value in arrays.values():
        value.setflags(write=False)
    return arrays


@dataclass(frozen=True)
class RootRT:
    rotations: np.ndarray
    translations: np.ndarray
    root_index: int
    relative_nonrigid: bool

    def selected_metadata(self, ids: np.ndarray, augmentation: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "root_rt_rotations": self.rotations[ids].copy(),
            "root_rt_translations": self.translations[ids].copy(),
            "root_rt_augmentation": augmentation.copy(),
            "root_rt_root_index": np.asarray(self.root_index, dtype=np.int64),
            "root_rt_relative_nonrigid": np.asarray(self.relative_nonrigid, dtype=bool),
        }


def load_root_rt(
    row: Mapping[str, Any], frame_count: int, frame_numbers: np.ndarray | None,
    *, root_index: int | None = None,
) -> RootRT | None:
    spec = validate_descriptor(row)
    if spec is None:
        return None
    count = _integer(frame_count, "actual frame_count", 1)
    if spec["frame_count"] != count:
        raise ValueError("root_rt manifest/NPZ frame_count mismatch")
    if frame_numbers is None:
        raise ValueError("root_rt requires NPZ frame_numbers")
    numbers = np.asarray(frame_numbers)
    if numbers.shape != (count,) or not np.issubdtype(numbers.dtype, np.integer):
        raise ValueError("root_rt invalid NPZ frame_numbers")
    path = Path(spec["cache_path"])
    shard = _load_shard(str(path), _stamp(path))
    index = spec["index"]
    if index >= len(shard["frame_counts"]):
        raise ValueError("root_rt shard index out of bounds")
    if shard["frame_counts"][index] != count:
        raise ValueError("root_rt shard/NPZ frame_count mismatch")
    digest = hashlib.sha256(row["path"].encode("utf-8")).hexdigest()
    if shard["path_hashes"][index] != digest:
        raise ValueError("root_rt path hash mismatch (wrong asset/cache index)")
    if not np.array_equal(shard["frame_numbers"][index, :count], numbers):
        raise ValueError("root_rt frame_numbers mismatch")
    cached_root = int(shard["root_indices"][index])
    if root_index is not None and cached_root != root_index:
        raise ValueError("root_rt retained root index mismatch")
    return RootRT(shard["rotations"][index, :count], shard["translations"][index, :count],
                  cached_root, bool(shard["relative_nonrigid"][index]))


def align_points(points: np.ndarray, rt: RootRT) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 3 or points.shape[0] != len(rt.rotations) or points.shape[-1] != 3:
        raise ValueError("root_rt points must have shape [T,P,3]")
    if not np.isfinite(points).all():
        raise ValueError("root_rt points must be finite")
    aligned = np.empty(points.shape, dtype=np.float32)
    # Framewise float64 arithmetic bounds peak memory for large mesh sequences.
    for frame, (rotation, translation) in enumerate(zip(rt.rotations, rt.translations)):
        aligned[frame] = (points[frame].astype(np.float64) - translation) @ rotation
    if not np.isfinite(aligned).all():
        raise ValueError("root_rt aligned points exceed finite float32 range")
    return aligned


def normalize_inputs(
    vertices: np.ndarray, targets: np.ndarray, global_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Normalize selected INPUT meshes only; targets never participate in bbox.

    Optional proper Q rotates all selected geometry before bbox normalization:
    y = (x_aligned @ Q.T - center) / scale.
    """
    rotation = np.eye(3) if global_rotation is None else np.asarray(global_rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("global_rotation must have shape [3,3]")
    _proper_rotations(rotation, "global_rotation")
    vertices, targets = np.asarray(vertices), np.asarray(targets)
    if vertices.ndim != 3 or vertices.shape[-1] != 3 or min(vertices.shape[:2]) < 1:
        raise ValueError("root_rt input vertices must have nonempty shape [T,V,3]")
    if targets.ndim != 3 or targets.shape[-1] != 3 or min(targets.shape[:2]) < 1:
        raise ValueError("root_rt targets must have nonempty shape [T,J,3]")
    if vertices.shape[0] != targets.shape[0]:
        raise ValueError("root_rt vertices and targets must have matching frame counts")
    with np.errstate(over="ignore", invalid="ignore"):
        if global_rotation is not None:
            vertices = np.asarray(vertices, dtype=np.float64) @ rotation.T
            targets = np.asarray(targets, dtype=np.float64) @ rotation.T
        vertices = vertices.astype(np.float32, copy=False)
        targets = targets.astype(np.float32, copy=False)
    if not np.isfinite(vertices).all() or not np.isfinite(targets).all():
        raise ValueError("root_rt selected vertices/targets must be finite")
    lo, hi = vertices.min(axis=(0, 1)), vertices.max(axis=(0, 1))
    # Match the builder's vectorized float32 pair-window arithmetic, including
    # NumPy 1.x where multiplying a float32 scalar by a Python float promotes it.
    center = (lo + hi) * np.float32(0.5)
    scale = np.float32(PADDING) * np.max((hi - lo) * np.float32(0.5))
    if not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("root_rt input mesh union bbox is degenerate")
    vertices_n = (vertices - center) / scale
    targets_n = (targets - center) / scale
    if not np.isfinite(vertices_n).all() or not np.isfinite(targets_n).all():
        raise ValueError("root_rt normalization produced nonfinite coordinates")
    if np.max(np.abs(targets_n)) > np.float32(1.0):
        raise ValueError("root_rt normalized GT outside [-1,1] after input mesh bbox/augmentation")
    return vertices_n, targets_n, center, float(scale), rotation


def restore_points(
    normalized: np.ndarray, rotations: np.ndarray, translations: np.ndarray,
    center: np.ndarray, scale: float, augmentation: np.ndarray | None = None,
) -> np.ndarray:
    """Restore [T,P,3] predictions using selected R/t and saved mesh c/s/Q."""
    normalized = np.asarray(normalized, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    if normalized.ndim != 3 or normalized.shape[-1] != 3:
        raise ValueError("restore_points requires [T,P,3] coordinates")
    if rotations.shape != (len(normalized), 3, 3) or translations.shape != (len(normalized), 3):
        raise ValueError("restore_points selected RT frame count mismatch")
    rotation = np.eye(3) if augmentation is None else np.asarray(augmentation, dtype=np.float64)
    aligned = (normalized * float(scale) + np.asarray(center)) @ rotation
    return np.einsum("tpi,tji->tpj", aligned, rotations) + translations[:, None]
