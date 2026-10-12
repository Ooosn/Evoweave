"""Mesh-only rigid gauges for read-only common-coordinate diagnostics."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.transform import Rotation


def fit_points(moving, reference, *, robust=False):
    moving, reference = np.asarray(moving, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if moving.shape != reference.shape or moving.ndim != 2 or moving.shape[1] != 3 or len(moving) < 3:
        raise ValueError("rigid alignment needs at least three corresponding 3D points")
    if not np.isfinite(moving).all() or not np.isfinite(reference).all():
        raise ValueError("nonfinite alignment points")
    weights = np.ones(len(moving), dtype=np.float64)
    warning_messages = []
    for _ in range(5 if robust else 1):
        source_center = np.average(moving, axis=0, weights=weights)
        target_center = np.average(reference, axis=0, weights=weights)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            rotation, _ = Rotation.align_vectors(reference - target_center, moving - source_center, weights=weights)
        warning_messages += [str(item.message) for item in recorded]
        matrix = rotation.as_matrix()
        translation = target_center - matrix @ source_center
        residual = np.linalg.norm(moving @ matrix.T + translation - reference, axis=1)
        if robust:
            scale = max(float(np.median(residual)), 1e-10)
            weights = np.minimum(1., 1.5 * scale / np.maximum(residual, 1e-12))
    singular = np.linalg.svd((moving - source_center) * np.sqrt(weights[:, None]), compute_uv=False)
    return matrix, translation, {
        "warnings": sorted(set(warning_messages)),
        "second_to_first_singular_ratio": float(singular[1] / max(singular[0], 1e-12)),
        "minimum_weight": float(weights.min()),
    }


def transform_sequence(points, rotations, translations, *, normals=False):
    points = np.asarray(points)
    result = np.einsum("tij,tnj->tni", rotations, points)
    if not normals:
        result += np.asarray(translations)[:, None]
    result[0] = points[0]
    return result.astype(points.dtype, copy=False)


def mesh_gauge(vertices, method, *, fit_vertices=5000):
    vertices = np.asarray(vertices)
    if vertices.ndim != 3 or vertices.shape[-1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("expected finite [T,V,3] mesh sequence")
    if method not in {"translation", "rigid", "robust_rigid"}:
        raise ValueError("unknown mesh gauge")
    if vertices.shape[1] < 6:
        raise ValueError("split-fit stability requires six or more vertices")
    indices = np.linspace(0, vertices.shape[1] - 1, min(fit_vertices, vertices.shape[1]), dtype=np.int64)
    reference = vertices[0, indices].astype(np.float64)
    bbox_side = float(np.ptp(vertices[0].astype(np.float64), axis=0).max())
    if bbox_side <= 0:
        raise ValueError("degenerate reference mesh bbox")
    rotations = np.repeat(np.eye(3)[None], len(vertices), axis=0)
    translations = np.zeros((len(vertices), 3), dtype=np.float64)
    frames = []
    for frame in range(len(vertices)):
        moving = vertices[frame, indices].astype(np.float64)
        info = {"warnings": [], "second_to_first_singular_ratio": None, "minimum_weight": 1.}
        angle, displacement, split_alignment = 0., 0., 0.
        if frame:
            if method == "translation":
                translations[frame] = reference.mean(0) - moving.mean(0)
                left_t = reference[::2].mean(0) - moving[::2].mean(0)
                right_t = reference[1::2].mean(0) - moving[1::2].mean(0)
                displacement = float(np.linalg.norm(left_t - right_t) / bbox_side)
                split_alignment = displacement
            else:
                rotations[frame], translations[frame], info = fit_points(moving, reference, robust=method == "robust_rigid")
                left_r, left_t, _ = fit_points(moving[::2], reference[::2], robust=method == "robust_rigid")
                right_r, right_t, _ = fit_points(moving[1::2], reference[1::2], robust=method == "robust_rigid")
                angle = float(np.degrees(Rotation.from_matrix(left_r @ right_r.T).magnitude()))
                displacement = float(np.linalg.norm(left_t - right_t) / bbox_side)
                difference = moving @ left_r.T + left_t - (moving @ right_r.T + right_t)
                split_alignment = float(np.sqrt(np.mean(np.sum(difference ** 2, axis=-1))) / bbox_side)
        posed = vertices[frame].astype(np.float64) @ rotations[frame].T + translations[frame]
        residual = np.linalg.norm(posed - vertices[0], axis=-1)
        restored = (posed - translations[frame]) @ rotations[frame]
        frames.append({
            "frame_position": frame, "fit_rms_bbox": float(np.sqrt(np.mean(residual ** 2)) / bbox_side),
            "split_rotation_degrees": angle, "split_translation_bbox": displacement,
            "split_alignment_rms_bbox": split_alignment,
            "inverse_max_error_bbox": float(np.max(np.abs(restored - vertices[frame])) / bbox_side),
            "determinant": float(np.linalg.det(rotations[frame])), **info,
        })
    return rotations, translations, {"method": method, "fit_vertex_count": len(indices), "frames": frames,
                                     "gauge_not_authoritative_global_rt": True}


def root_gauge(bone_transforms, root_index, selected, center, scale):
    deltas = np.asarray(bone_transforms, dtype=np.float64)[selected, int(root_index)]
    if deltas.shape != (len(selected), 4, 4) or not np.isfinite(deltas).all():
        raise ValueError("invalid raw-root bone delta shape or values")
    if not np.allclose(deltas[:, 3], [0., 0., 0., 1.], atol=1e-5):
        raise ValueError("raw-root deltas are not affine")
    delta_rotations = deltas[:, :3, :3]
    if not np.allclose(delta_rotations @ delta_rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3):
        raise ValueError("raw-root deltas contain nonrigid scale or shear")
    if not np.allclose(np.linalg.det(delta_rotations), 1., atol=1e-3):
        raise ValueError("raw-root deltas do not contain proper rotations")
    relative = deltas[0] @ np.linalg.inv(deltas)
    rotations, translation_world = relative[:, :3, :3], relative[:, :3, 3]
    center = np.asarray(center, dtype=np.float64)
    translations = (np.einsum("tij,j->ti", rotations, center) + translation_world - center) / float(scale)
    rotations[0], translations[0] = np.eye(3), 0.
    return rotations, translations


def coordinate_summary(points):
    points = np.asarray(points)
    if not np.isfinite(points).all():
        raise ValueError("nonfinite common-coordinate points")
    maxima = np.max(np.abs(points), axis=-1)
    return {"maxabs": float(maxima.max()), "point_fraction_outside_1": float(np.mean(maxima > 1.)),
            "point_fraction_outside_2": float(np.mean(maxima > 2.)),
            "point_fraction_outside_4": float(np.mean(maxima > 4.)),
            "per_frame_maxabs": np.max(maxima, axis=-1).tolist()}
