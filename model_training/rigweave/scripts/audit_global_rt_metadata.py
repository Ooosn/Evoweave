#!/usr/bin/env python3
"""Audit rigid-transform metadata without loading mesh or animation arrays.

The rootless-v3 producer keeps deleted top-root transforms in
``recorded_root_*`` fields.  Those fields are useful metadata candidates, but
their presence or orthogonality does not establish that they are the unique
object-global transform.  This audit therefore reports the producer-backed
recorded-root family separately from explicitly named source/object transform
fields and from the identity compatibility fields used by rootless training.

The command streams manifest rows and writes one compact JSON object per row.
Only the small metadata arrays named below are read from each NPZ; mesh,
faces, skin weights, and frame geometry are never accessed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


AUDIT_SCHEMA_VERSION = "global_rt_metadata_audit_v1"
ROTATION_ORTHOGONALITY_ATOL = 1.0e-3
ROTATION_DETERMINANT_ATOL = 1.0e-3
ROTATION_INVERSE_ATOL = 1.0e-3
AFFINE_LAST_ROW_ATOL = 1.0e-3
ROTATION_CHUNK = 4096

# These are the fields written by data_processing/scripts/
# build_rootless_dynamic_npz.py.  ``recorded_root_*`` is the only family that
# contains the deleted-root transforms; the root_* fields below are explicitly
# the identity training compatibility transform in schema v3.
RECORDED_ROOT_FIELDS = (
    "recorded_root_indices",
    "recorded_root_names",
    "recorded_root_positions",
    "recorded_root_rotations_world_to_root",
    "recorded_root_rotations_root_to_world",
)
RECORDED_ROOT_CORE_FIELDS = (
    "recorded_root_indices",
    "recorded_root_positions",
    "recorded_root_rotations_world_to_root",
    "recorded_root_rotations_root_to_world",
)
TRAINING_ROOT_FIELDS = (
    "root_positions",
    "root_rotations_world_to_root",
    "root_rotations_root_to_world",
    "rootspace_from_world_mats",
    "world_from_rootspace_mats",
)

# No current Evoweave producer writes these names.  They are recognized so a
# future/source-side export is reported separately rather than being confused
# with deleted-root metadata.  Per-bone ``bone_transforms`` is intentionally
# absent: it is not a unique global/object transform.
OBJECT_MATRIX_FIELDS = (
    "source_object_to_world",
    "source_world_from_object",
    "object_to_world",
    "world_from_object",
    "object_to_world_matrix",
    "world_to_object_matrix",
    "source_world_to_object",
    "world_to_object",
    "source_world_transform",
    "source_object_transform",
    "world_transform",
    "object_transform",
    "source_global_transform",
    "global_transform",
    "source_global_rt",
    "global_rt",
)
OBJECT_POSITION_FIELDS = (
    "source_object_positions",
    "source_object_translations",
    "object_positions",
    "object_translations",
    "source_global_positions",
    "global_positions",
)
OBJECT_ROTATION_FIELDS = (
    "source_object_rotations_object_to_world",
    "source_object_rotations_world_to_object",
    "object_rotations_object_to_world",
    "object_rotations_world_to_object",
    "object_rotations",
    "source_global_rotations",
    "global_rotations",
)

ABSENT = "absent"
EMPTY = "empty"
MALFORMED = "malformed"
NONFINITE = "nonfinite"
INVALID_ROTATION = "invalid_rotation"
INVALID_AFFINE = "invalid_affine"
INVERSE_MISMATCH = "inverse_mismatch"
AMBIGUOUS = "ambiguous_multiple_roots"
UNIQUE = "unique_candidate"
IDENTITY_COMPATIBILITY = "identity_compatibility"
NONIDENTITY_COMPATIBILITY = "nonidentity_training_transform"
NOT_AUDITED = "not_audited"

_VALID_CANDIDATE_STATUSES = {UNIQUE}
_FAILURE_STATUSES = {
    EMPTY,
    MALFORMED,
    NONFINITE,
    INVALID_ROTATION,
    INVALID_AFFINE,
    INVERSE_MISMATCH,
    AMBIGUOUS,
}


def _new_family_result(status: str = ABSENT, fields_present: Iterable[str] = ()) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "field_presence": "absent" if status == ABSENT else "empty" if status == EMPTY else "present",
        "fields_present": list(fields_present),
        "shape_valid": None,
        "finite": None,
        "rotation_valid": None,
        "root_count": None,
        "frame_count": None,
        "rotation_max_orthogonality_error": None,
        "rotation_max_determinant_error": None,
        "rotation_inverse_max_error": None,
        "affine_last_row_max_error": None,
        "details": [],
    }
    return result


def _set_status(result: dict[str, Any], status: str) -> dict[str, Any]:
    result["status"] = status
    result["field_presence"] = "absent" if status == ABSENT else "empty" if status == EMPTY else "present"
    return result


def _short_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:240]


def _load_numeric(z: Any, name: str, dtype: Any) -> np.ndarray:
    value = np.asarray(z[name])
    if dtype is not None:
        value = np.asarray(value, dtype=dtype)
    return value


def _load_indices(z: Any, name: str) -> np.ndarray:
    raw = np.asarray(z[name])
    if raw.dtype.kind in "fc":
        if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
            raise ValueError(f"{name} contains non-integral or nonfinite values")
    return np.asarray(raw, dtype=np.int64)


def _rotation_metrics(rotations: np.ndarray) -> tuple[bool, float, float]:
    """Return validity and max errors without materializing all Gram matrices."""

    flat = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    if flat.shape[0] == 0:
        return True, 0.0, 0.0
    identity = np.eye(3, dtype=np.float64)
    max_orthogonality_error = 0.0
    max_determinant_error = 0.0
    for start in range(0, int(flat.shape[0]), ROTATION_CHUNK):
        chunk = flat[start : start + ROTATION_CHUNK]
        gram = np.matmul(np.swapaxes(chunk, 1, 2), chunk)
        max_orthogonality_error = max(
            max_orthogonality_error,
            float(np.max(np.abs(gram - identity))),
        )
        determinants = np.linalg.det(chunk)
        max_determinant_error = max(
            max_determinant_error,
            float(np.max(np.abs(determinants - 1.0))),
        )
    valid = (
        math.isfinite(max_orthogonality_error)
        and math.isfinite(max_determinant_error)
        and max_orthogonality_error <= ROTATION_ORTHOGONALITY_ATOL
        and max_determinant_error <= ROTATION_DETERMINANT_ATOL
    )
    return valid, max_orthogonality_error, max_determinant_error


def _record_rotation_metrics(result: dict[str, Any], rotations: Sequence[np.ndarray]) -> bool:
    arrays = [np.asarray(value, dtype=np.float64) for value in rotations]
    if not arrays:
        result["rotation_valid"] = None
        return True
    valid = True
    max_orthogonality_error = 0.0
    max_determinant_error = 0.0
    for value in arrays:
        current_valid, orth_error, determinant_error = _rotation_metrics(value)
        valid = valid and current_valid
        max_orthogonality_error = max(max_orthogonality_error, orth_error)
        max_determinant_error = max(max_determinant_error, determinant_error)
    result["rotation_valid"] = bool(valid)
    result["rotation_max_orthogonality_error"] = float(max_orthogonality_error)
    result["rotation_max_determinant_error"] = float(max_determinant_error)
    return bool(valid)


def _record_inverse_consistency(
    result: dict[str, Any],
    world_to_root: np.ndarray,
    root_to_world: np.ndarray,
) -> bool:
    inverse_error = float(
        np.max(np.abs(np.asarray(root_to_world, dtype=np.float64) - np.swapaxes(world_to_root, -1, -2)))
    )
    result["rotation_inverse_max_error"] = inverse_error
    return math.isfinite(inverse_error) and inverse_error <= ROTATION_INVERSE_ATOL


def _read_npy_shape_header(z: Any, array_name: str) -> tuple[int, ...]:
    """Read one NPZ member's NPY header without reading its array payload."""

    if array_name not in z.files:
        raise ValueError(f"missing_header_member:{array_name}")
    member_name = f"{array_name}.npy"
    with z.zip.open(member_name, "r") as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, _fortran_order, _dtype = np.lib.format.read_array_header_1_0(member)
        elif version == (2, 0):
            shape, _fortran_order, _dtype = np.lib.format.read_array_header_2_0(member)
        elif version == (3, 0):
            # NumPy 2.x exposes the v2 reader only; the header length and
            # dictionary layout are compatible for this ASCII schema.
            shape, _fortran_order, _dtype = np.lib.format.read_array_header_2_0(member)
        else:
            raise ValueError(f"unsupported_npy_header_version:{version}")
    return tuple(int(value) for value in shape)


def _record_finite(result: dict[str, Any], arrays: Sequence[np.ndarray]) -> bool:
    finite = all(bool(np.isfinite(np.asarray(value)).all()) for value in arrays)
    result["finite"] = bool(finite)
    return bool(finite)


def _record_shape_error(result: dict[str, Any], details: Sequence[str]) -> dict[str, Any]:
    result["shape_valid"] = False
    result["details"] = list(details)[:8]
    return _set_status(result, MALFORMED)


def _audit_recorded_root(z: Any) -> dict[str, Any]:
    present = [name for name in RECORDED_ROOT_FIELDS if name in z.files]
    result = _new_family_result(ABSENT, present)
    if not present:
        return result

    values: dict[str, np.ndarray] = {}
    try:
        if "recorded_root_indices" in z.files:
            values["recorded_root_indices"] = _load_indices(z, "recorded_root_indices")
        if "recorded_root_positions" in z.files:
            values["recorded_root_positions"] = _load_numeric(z, "recorded_root_positions", np.float64)
        for name in (
            "recorded_root_rotations_world_to_root",
            "recorded_root_rotations_root_to_world",
        ):
            if name in z.files:
                values[name] = _load_numeric(z, name, np.float64)
    except Exception as exc:  # noqa: BLE001 - row-level audit must continue.
        result["details"] = [f"metadata_read_error:{_short_error(exc)}"]
        return _record_shape_error(result, result["details"])

    missing_core = [name for name in RECORDED_ROOT_CORE_FIELDS if name not in values]
    if missing_core:
        return _record_shape_error(result, [f"recorded_root_core_fields_missing:{','.join(missing_core)}"])

    shape_errors: list[str] = []
    frame_counts: list[int] = []
    root_counts: list[int] = []
    positions = values.get("recorded_root_positions")
    if positions is not None:
        if positions.ndim != 3 or positions.shape[-1] != 3:
            shape_errors.append(f"recorded_root_positions_shape={tuple(positions.shape)}")
        else:
            frame_counts.append(int(positions.shape[0]))
            root_counts.append(int(positions.shape[1]))

    rotations: list[np.ndarray] = []
    for name in (
        "recorded_root_rotations_world_to_root",
        "recorded_root_rotations_root_to_world",
    ):
        value = values.get(name)
        if value is None:
            continue
        if value.ndim != 4 or tuple(value.shape[-2:]) != (3, 3):
            shape_errors.append(f"{name}_shape={tuple(value.shape)}")
        else:
            frame_counts.append(int(value.shape[0]))
            root_counts.append(int(value.shape[1]))
            rotations.append(value)

    indices = values.get("recorded_root_indices")
    if indices is not None:
        if indices.ndim != 1:
            shape_errors.append(f"recorded_root_indices_shape={tuple(indices.shape)}")
        else:
            root_counts.append(int(indices.shape[0]))

    if positions is None:
        shape_errors.append("recorded_root_positions_missing")
    if not rotations:
        shape_errors.append("recorded_root_rotations_missing")
    if frame_counts and len(set(frame_counts)) != 1:
        shape_errors.append(f"frame_count_mismatch={frame_counts}")
    if root_counts and len(set(root_counts)) != 1:
        shape_errors.append(f"root_count_mismatch={root_counts}")
    if shape_errors:
        return _record_shape_error(result, shape_errors)

    frame_count = int(frame_counts[0]) if frame_counts else 0
    root_count = int(root_counts[0]) if root_counts else 0
    result["shape_valid"] = True
    result["frame_count"] = frame_count
    result["root_count"] = root_count
    try:
        frame_header_shape = _read_npy_shape_header(z, "frame_vertices_rootspace")
    except Exception as exc:  # noqa: BLE001 - row-level audit must continue.
        return _record_shape_error(result, [f"frame_vertices_rootspace_header_error:{_short_error(exc)}"])
    if not frame_header_shape:
        return _record_shape_error(result, ["frame_vertices_rootspace_header_has_no_frame_axis"])
    actual_frame_count = int(frame_header_shape[0])
    if actual_frame_count != frame_count:
        return _record_shape_error(
            result,
            [f"recorded_frame_count_mismatch:{frame_count}!={actual_frame_count}"],
        )
    finite_arrays = [value for name, value in values.items() if name != "recorded_root_indices"]
    if not _record_finite(result, finite_arrays):
        return _set_status(result, NONFINITE)
    if frame_count == 0 or root_count == 0:
        result["rotation_valid"] = None
        return _set_status(result, EMPTY)
    if not _record_rotation_metrics(result, rotations):
        return _set_status(result, INVALID_ROTATION)
    if not _record_inverse_consistency(
        result,
        values["recorded_root_rotations_world_to_root"],
        values["recorded_root_rotations_root_to_world"],
    ):
        result["details"] = [
            f"rotation_inverse_mismatch:{result['rotation_inverse_max_error']:.9g}"
        ]
        return _set_status(result, INVERSE_MISMATCH)
    if indices is not None and int(np.unique(indices).size) != int(indices.size):
        return _record_shape_error(result, ["recorded_root_indices_not_unique"])
    if root_count > 1:
        return _set_status(result, AMBIGUOUS)
    return _set_status(result, UNIQUE)


def _audit_training_root_compatibility(z: Any) -> dict[str, Any]:
    present = [name for name in TRAINING_ROOT_FIELDS if name in z.files]
    result = _new_family_result(ABSENT, present)
    if not present:
        return result
    values: dict[str, np.ndarray] = {}
    try:
        for name in present:
            # The matrix fields are loaded only to report their compatibility
            # shape; they are still small metadata fields in the v3 producer.
            values[name] = _load_numeric(z, name, np.float64)
    except Exception as exc:  # noqa: BLE001 - row-level audit must continue.
        return _record_shape_error(result, [f"metadata_read_error:{_short_error(exc)}"])

    positions = values.get("root_positions")
    rotations = [
        values[name]
        for name in ("root_rotations_world_to_root", "root_rotations_root_to_world")
        if name in values
    ]
    shape_errors: list[str] = []
    frame_counts: list[int] = []
    if positions is not None:
        if positions.ndim != 2 or positions.shape[-1] != 3:
            shape_errors.append(f"root_positions_shape={tuple(positions.shape)}")
        else:
            frame_counts.append(int(positions.shape[0]))
    if not rotations:
        shape_errors.append("root_rotations_missing")
    for name, value in (
        ("root_rotations_world_to_root", values.get("root_rotations_world_to_root")),
        ("root_rotations_root_to_world", values.get("root_rotations_root_to_world")),
    ):
        if value is None:
            continue
        if value.ndim != 3 or tuple(value.shape[-2:]) != (3, 3):
            shape_errors.append(f"{name}_shape={tuple(value.shape)}")
        else:
            frame_counts.append(int(value.shape[0]))
    if positions is None:
        shape_errors.append("root_positions_missing")
    if frame_counts and len(set(frame_counts)) != 1:
        shape_errors.append(f"frame_count_mismatch={frame_counts}")
    if shape_errors:
        return _record_shape_error(result, shape_errors)

    frame_count = int(frame_counts[0]) if frame_counts else 0
    result["shape_valid"] = True
    result["frame_count"] = frame_count
    result["root_count"] = 1
    if not _record_finite(result, list(values.values())):
        return _set_status(result, NONFINITE)
    if frame_count == 0:
        result["rotation_valid"] = None
        return _set_status(result, EMPTY)
    if not _record_rotation_metrics(result, rotations):
        return _set_status(result, INVALID_ROTATION)

    positions_zero = bool(np.max(np.abs(positions)) <= ROTATION_ORTHOGONALITY_ATOL) if positions is not None else False
    identity = np.eye(3, dtype=np.float64)
    rotations_identity = all(bool(np.max(np.abs(value - identity)) <= ROTATION_ORTHOGONALITY_ATOL) for value in rotations)
    if positions_zero and rotations_identity:
        return _set_status(result, IDENTITY_COMPATIBILITY)
    return _set_status(result, NONIDENTITY_COMPATIBILITY)


def _normalise_object_positions(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1 and array.shape == (3,):
        return array.reshape(1, 1, 3)
    if array.ndim == 2 and array.shape[-1] == 3:
        return array[:, None, :]
    if array.ndim == 3 and array.shape[-1] == 3:
        return array
    raise ValueError(f"object position shape={tuple(array.shape)}")


def _normalise_object_rotations(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and tuple(array.shape) == (3, 3):
        return array.reshape(1, 1, 3, 3)
    if array.ndim == 3 and tuple(array.shape[-2:]) == (3, 3):
        return array[:, None, :, :]
    if array.ndim == 4 and tuple(array.shape[-2:]) == (3, 3):
        return array
    raise ValueError(f"object rotation shape={tuple(array.shape)}")


def _audit_explicit_object_transform(z: Any) -> dict[str, Any]:
    present = [
        name
        for name in (*OBJECT_MATRIX_FIELDS, *OBJECT_POSITION_FIELDS, *OBJECT_ROTATION_FIELDS)
        if name in z.files
    ]
    result = _new_family_result(ABSENT, present)
    if not present:
        return result

    matrix_name = next((name for name in OBJECT_MATRIX_FIELDS if name in z.files), None)
    try:
        if matrix_name is not None:
            value = _load_numeric(z, matrix_name, np.float64)
            if value.size == 0:
                result["shape_valid"] = True
                result["finite"] = True
                return _set_status(result, EMPTY)
            if value.ndim == 2 and tuple(value.shape) == (4, 4):
                transforms = value.reshape(1, 1, 4, 4)
            elif value.ndim == 3 and tuple(value.shape[-2:]) == (4, 4):
                transforms = value[:, None, :, :]
            elif value.ndim == 4 and tuple(value.shape[-2:]) == (4, 4):
                transforms = value
            else:
                return _record_shape_error(result, [f"{matrix_name}_shape={tuple(value.shape)}"])
            result["shape_valid"] = True
            result["frame_count"] = int(transforms.shape[0])
            result["root_count"] = int(transforms.shape[1])
            if not _record_finite(result, [transforms]):
                return _set_status(result, NONFINITE)
            affine_target = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            affine_error = float(np.max(np.abs(transforms[..., 3, :] - affine_target)))
            result["affine_last_row_max_error"] = affine_error
            if not math.isfinite(affine_error) or affine_error > AFFINE_LAST_ROW_ATOL:
                return _set_status(result, INVALID_AFFINE)
            if int(transforms.shape[0]) == 0 or int(transforms.shape[1]) == 0:
                result["rotation_valid"] = None
                return _set_status(result, EMPTY)
            if not _record_rotation_metrics(result, [transforms[..., :3, :3]]):
                return _set_status(result, INVALID_ROTATION)
            if int(transforms.shape[1]) > 1:
                return _set_status(result, AMBIGUOUS)
            return _set_status(result, UNIQUE)

        position_name = next((name for name in OBJECT_POSITION_FIELDS if name in z.files), None)
        rotation_name = next((name for name in OBJECT_ROTATION_FIELDS if name in z.files), None)
        if position_name is None or rotation_name is None:
            return _record_shape_error(result, ["object_position_or_rotation_missing"])
        positions = _normalise_object_positions(_load_numeric(z, position_name, np.float64))
        rotations = _normalise_object_rotations(_load_numeric(z, rotation_name, np.float64))
    except Exception as exc:  # noqa: BLE001 - row-level audit must continue.
        return _record_shape_error(result, [f"metadata_read_error:{_short_error(exc)}"])

    shape_errors: list[str] = []
    if positions.shape[:2] != rotations.shape[:2]:
        shape_errors.append(
            f"position_rotation_count_mismatch={tuple(positions.shape[:2])}!={tuple(rotations.shape[:2])}"
        )
    if shape_errors:
        return _record_shape_error(result, shape_errors)
    result["shape_valid"] = True
    result["frame_count"] = int(positions.shape[0])
    result["root_count"] = int(positions.shape[1])
    if not _record_finite(result, [positions, rotations]):
        return _set_status(result, NONFINITE)
    if int(positions.shape[0]) == 0 or int(positions.shape[1]) == 0:
        result["rotation_valid"] = None
        return _set_status(result, EMPTY)
    if not _record_rotation_metrics(result, [rotations]):
        return _set_status(result, INVALID_ROTATION)
    if int(positions.shape[1]) > 1:
        return _set_status(result, AMBIGUOUS)
    return _set_status(result, UNIQUE)


def _candidate_status(recorded: Mapping[str, Any], explicit: Mapping[str, Any]) -> str:
    families = (recorded, explicit)
    unique_families = [family for family in families if family.get("status") == UNIQUE]
    ambiguous_families = [family for family in families if family.get("status") == AMBIGUOUS]
    invalid_families = [
        family
        for family in families
        if family.get("status") in {MALFORMED, NONFINITE, INVALID_ROTATION, INVALID_AFFINE, INVERSE_MISMATCH}
    ]
    if len(unique_families) > 1:
        return "multiple_unique_candidate_sources"
    if unique_families and not ambiguous_families:
        if recorded.get("status") == UNIQUE:
            return "recorded_root_unique_candidate"
        return "source_object_unique_candidate"
    if ambiguous_families:
        return AMBIGUOUS
    if invalid_families:
        return "metadata_invalid"
    if any(family.get("status") == EMPTY for family in families):
        return EMPTY
    return ABSENT


def _not_audited_family() -> dict[str, Any]:
    return _new_family_result(NOT_AUDITED)


def _normalise_split(value: Any, manifest_path: Path) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"val", "valid", "validation"}:
        return "valid"
    if raw == "train":
        return "train"
    if manifest_path.name.startswith("valid") or manifest_path.name.startswith("val"):
        return "valid"
    if manifest_path.name.startswith("train"):
        return "train"
    return raw or "unknown"


def _row_source(row: Mapping[str, Any]) -> str:
    for key in ("dataset_source", "source", "data_source"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return "unknown"


def _row_reference(row_result: Mapping[str, Any], category: str) -> dict[str, Any]:
    return {
        "category": category,
        "manifest": row_result.get("manifest"),
        "manifest_line": row_result.get("manifest_line"),
        "split": row_result.get("split"),
        "source": row_result.get("source"),
        "asset_id": row_result.get("asset_id"),
        "sequence_id": row_result.get("sequence_id"),
        "path": row_result.get("path"),
    }


def _record_representatives(
    summary: dict[str, Any],
    row_result: Mapping[str, Any],
    categories: Iterable[str],
    limit: int,
) -> None:
    if int(limit) <= 0:
        return
    representatives = summary["representative_failures"]
    for category in categories:
        values = representatives.setdefault(str(category), [])
        if len(values) < int(limit):
            values.append(_row_reference(row_result, str(category)))


def _increment_status(summary: dict[str, Any], family_name: str, result: Mapping[str, Any], split: str, source: str) -> None:
    family = summary["families"][family_name]
    status = str(result.get("status", NOT_AUDITED))
    family["status_counts"][status] = int(family["status_counts"].get(status, 0)) + 1
    family["status_by_split"].setdefault(split, {})[status] = int(
        family["status_by_split"].setdefault(split, {}).get(status, 0)
    ) + 1
    family["status_by_source"].setdefault(source, {})[status] = int(
        family["status_by_source"].setdefault(source, {}).get(status, 0)
    ) + 1
    family["status_by_split_source"].setdefault(split, {}).setdefault(source, {})[status] = int(
        family["status_by_split_source"].setdefault(split, {}).setdefault(source, {}).get(status, 0)
    ) + 1
    for field_name in result.get("fields_present", []):
        family["field_counts"][str(field_name)] = int(family["field_counts"].get(str(field_name), 0)) + 1


def _new_family_summary() -> dict[str, Any]:
    return {
        "status_counts": {},
        "status_by_split": {},
        "status_by_source": {},
        "status_by_split_source": {},
        "field_counts": {},
    }


def _new_summary(manifests: Sequence[Path], representative_limit: int) -> dict[str, Any]:
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "input_manifests": [str(path) for path in manifests],
        "representative_limit_per_category": int(representative_limit),
        "rows": 0,
        "denominators": {
            "split": {},
            "source": {},
            "split_source": {},
        },
        "families": {
            "recorded_root": _new_family_summary(),
            "source_object_transform": _new_family_summary(),
            "training_root_compatibility": _new_family_summary(),
        },
        "candidate_global_rt": {
            "status_counts": {},
            "status_by_split": {},
            "status_by_source": {},
            "status_by_split_source": {},
        },
        "failure_categories": {},
        "representative_failures": {},
        "manifest_errors": 0,
    }


def _increment_denominators(summary: dict[str, Any], split: str, source: str) -> None:
    denominators = summary["denominators"]
    denominators["split"][split] = int(denominators["split"].get(split, 0)) + 1
    denominators["source"][source] = int(denominators["source"].get(source, 0)) + 1
    split_sources = denominators["split_source"].setdefault(split, {})
    split_sources[source] = int(split_sources.get(source, 0)) + 1


def _increment_candidate(summary: dict[str, Any], status: str, split: str, source: str) -> None:
    candidate = summary["candidate_global_rt"]
    candidate["status_counts"][status] = int(candidate["status_counts"].get(status, 0)) + 1
    candidate["status_by_split"].setdefault(split, {})[status] = int(
        candidate["status_by_split"].setdefault(split, {}).get(status, 0)
    ) + 1
    candidate["status_by_source"].setdefault(source, {})[status] = int(
        candidate["status_by_source"].setdefault(source, {}).get(status, 0)
    ) + 1
    candidate["status_by_split_source"].setdefault(split, {}).setdefault(source, {})[status] = int(
        candidate["status_by_split_source"].setdefault(split, {}).setdefault(source, {}).get(status, 0)
    ) + 1


def _increment_failure(summary: dict[str, Any], category: str) -> None:
    summary["failure_categories"][category] = int(summary["failure_categories"].get(category, 0)) + 1


def _failure_categories(
    row_status: str,
    candidate_status: str,
    recorded: Mapping[str, Any],
    explicit: Mapping[str, Any],
) -> list[str]:
    categories: list[str] = []
    if row_status != "audited":
        categories.append(row_status)
        return categories
    for name, result in (
        ("recorded_root", recorded),
        ("source_object_transform", explicit),
    ):
        status = str(result.get("status", NOT_AUDITED))
        if status in _FAILURE_STATUSES:
            categories.append(f"{name}:{status}")
    if candidate_status not in {"recorded_root_unique_candidate", "source_object_unique_candidate"}:
        categories.append(f"candidate_global_rt:{candidate_status}")
    return categories


def _audit_one_row(row: Mapping[str, Any], manifest_path: Path, manifest_line: int, row_index: int) -> dict[str, Any]:
    split = _normalise_split(row.get("split"), manifest_path)
    source = _row_source(row)
    raw_path = str(row.get("path") or row.get("rootless_path") or "").strip()
    path = Path(raw_path) if raw_path else None
    if path is not None and not path.is_absolute():
        path = manifest_path.parent / path
    base: dict[str, Any] = {
        "manifest": str(manifest_path),
        "manifest_line": int(manifest_line),
        "row_index": int(row_index),
        "split": split,
        "source": source,
        "asset_id": str(row.get("asset_id") or (path.stem if path is not None else "")),
        "sequence_id": str(row.get("sequence_id", "0")),
        "path": str(path) if path is not None else "",
    }
    if path is None:
        return {
            **base,
            "row_status": "manifest_missing_path",
            "metadata": {
                "recorded_root": _not_audited_family(),
                "source_object_transform": _not_audited_family(),
                "training_root_compatibility": _not_audited_family(),
            },
            "candidate_global_rt": {"status": NOT_AUDITED, "family_statuses": {}},
            "failure_categories": ["manifest_missing_path"],
        }
    if not path.exists():
        return {
            **base,
            "row_status": "npz_missing",
            "metadata": {
                "recorded_root": _not_audited_family(),
                "source_object_transform": _not_audited_family(),
                "training_root_compatibility": _not_audited_family(),
            },
            "candidate_global_rt": {"status": NOT_AUDITED, "family_statuses": {}},
            "failure_categories": ["npz_missing"],
        }

    try:
        # Do not use the model/data loaders here: they eagerly load geometry.
        # ``files`` is inspected first, and only the small named metadata arrays
        # are subsequently accessed by the family auditors.
        with np.load(path, allow_pickle=False) as z:
            recorded = _audit_recorded_root(z)
            explicit = _audit_explicit_object_transform(z)
            training = _audit_training_root_compatibility(z)
    except Exception as exc:  # noqa: BLE001 - one bad NPZ must not stop census.
        return {
            **base,
            "row_status": "npz_load_error",
            "error": _short_error(exc),
            "metadata": {
                "recorded_root": _not_audited_family(),
                "source_object_transform": _not_audited_family(),
                "training_root_compatibility": _not_audited_family(),
            },
            "candidate_global_rt": {"status": NOT_AUDITED, "family_statuses": {}},
            "failure_categories": ["npz_load_error"],
        }

    candidate_status = _candidate_status(recorded, explicit)
    categories = _failure_categories("audited", candidate_status, recorded, explicit)
    return {
        **base,
        "row_status": "audited",
        "metadata": {
            "recorded_root": recorded,
            "source_object_transform": explicit,
            "training_root_compatibility": training,
        },
        "candidate_global_rt": {
            "status": candidate_status,
            "family_statuses": {
                "recorded_root": recorded["status"],
                "source_object_transform": explicit["status"],
            },
        },
        "failure_categories": categories,
    }


def _audit_manifest_rows(
    manifests: Sequence[Path],
    rows_handle: Any,
    summary: dict[str, Any],
    *,
    limit: int,
    representative_limit: int,
    progress_every: int,
) -> None:
    processed = 0
    for manifest_path in manifests:
        with manifest_path.open("r", encoding="utf-8") as handle:
            row_index = 0
            for manifest_line, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if int(limit) > 0 and processed >= int(limit):
                    return
                row_index += 1
                processed += 1
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("manifest row is not a JSON object")
                    result = _audit_one_row(row, manifest_path, manifest_line, row_index)
                except Exception as exc:  # noqa: BLE001 - preserve a row record.
                    split = _normalise_split("", manifest_path)
                    result = {
                        "manifest": str(manifest_path),
                        "manifest_line": int(manifest_line),
                        "row_index": int(row_index),
                        "split": split,
                        "source": "unknown",
                        "asset_id": "",
                        "sequence_id": "",
                        "path": "",
                        "row_status": "manifest_row_error",
                        "error": _short_error(exc),
                        "metadata": {
                            "recorded_root": _not_audited_family(),
                            "source_object_transform": _not_audited_family(),
                            "training_root_compatibility": _not_audited_family(),
                        },
                        "candidate_global_rt": {"status": NOT_AUDITED, "family_statuses": {}},
                        "failure_categories": ["manifest_row_error"],
                    }
                rows_handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                rows_handle.flush()

                split = str(result["split"])
                source = str(result["source"])
                _increment_denominators(summary, split, source)
                summary["rows"] = int(summary["rows"]) + 1
                if result["row_status"] == "manifest_row_error":
                    summary["manifest_errors"] = int(summary["manifest_errors"]) + 1
                for family_name in ("recorded_root", "source_object_transform", "training_root_compatibility"):
                    _increment_status(summary, family_name, result["metadata"][family_name], split, source)
                candidate_status = str(result["candidate_global_rt"]["status"])
                _increment_candidate(summary, candidate_status, split, source)
                for category in result.get("failure_categories", []):
                    _increment_failure(summary, str(category))
                _record_representatives(summary, result, result.get("failure_categories", []), representative_limit)
                if int(progress_every) > 0 and processed % int(progress_every) == 0:
                    print(
                        json.dumps(
                            {
                                "event": "global_rt_metadata_audit_progress",
                                "rows": processed,
                                "manifest": str(manifest_path),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )


def run_census(
    manifests: Sequence[str | Path],
    output_rows: str | Path,
    output_summary: str | Path,
    *,
    limit: int = 0,
    representative_limit: int = 10,
    progress_every: int = 250,
) -> dict[str, Any]:
    """Run a streaming metadata census and return the written summary."""

    manifest_paths = [Path(path) for path in manifests]
    if not manifest_paths:
        raise ValueError("at least one manifest is required")
    if int(limit) < 0 or int(representative_limit) < 0 or int(progress_every) < 0:
        raise ValueError("limit, representative_limit, and progress_every must be nonnegative")
    rows_path = Path(output_rows)
    summary_path = Path(output_summary)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = _new_summary(manifest_paths, int(representative_limit))
    with rows_path.open("w", encoding="utf-8") as rows_handle:
        _audit_manifest_rows(
            manifest_paths,
            rows_handle,
            summary,
            limit=int(limit),
            representative_limit=int(representative_limit),
            progress_every=int(progress_every),
        )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        dest="manifests",
        type=Path,
        action="append",
        required=True,
        help="Accepted manifest JSONL; repeat for train and valid.",
    )
    parser.add_argument(
        "--output-rows",
        "--output-jsonl",
        "--rows-jsonl",
        dest="output_rows",
        type=Path,
        required=True,
        help="Compact per-row audit JSONL.",
    )
    parser.add_argument(
        "--output-summary",
        "--summary-json",
        dest="output_summary",
        type=Path,
        required=True,
        help="Summary JSON with split/source denominators and bounded representatives.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum rows across all manifests; 0 means all rows.",
    )
    parser.add_argument(
        "--representative-limit",
        type=int,
        default=10,
        help="Maximum representative rows retained per failure category.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Emit progress JSON every N rows; 0 disables progress output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_census(
        args.manifests,
        args.output_rows,
        args.output_summary,
        limit=int(args.limit),
        representative_limit=int(args.representative_limit),
        progress_every=int(args.progress_every),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
