from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

def _resolve_manifest_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"manifest path does not exist: {path}")
    return path


@dataclass(frozen=True)
class DynamicRigSample:
    path: str
    cls: str
    frame_vertices: torch.Tensor
    faces: torch.LongTensor
    vertex_normals: torch.Tensor
    face_normals: torch.Tensor
    input_ids: torch.LongTensor
    attention_mask: torch.Tensor
    target_joints: torch.Tensor
    target_parents: torch.LongTensor
    branch_prior_roots: torch.Tensor
    branch_prior_children: torch.Tensor
    branch_prior_mask: torch.Tensor
    selected_frames: torch.LongTensor
    query_center: torch.Tensor
    query_scale: torch.Tensor
    joint_count: int
    source_joint_count: int
    active_skin_joint_count: int
    target_start_index: int
    alignment_root_index: int
    vertex_count: int
    face_count: int


def _load_manifest(path: Path, limit: int = 0) -> list[Path]:
    rows: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            import json

            row = json.loads(line)
            value = row.get("path") or row.get("npz_path") or row.get("file")
            if value is None:
                raise ValueError(f"manifest JSON row has no path-like field: {line[:160]}")
            rows.append(_resolve_manifest_path(value))
        else:
            rows.append(_resolve_manifest_path(line))
    if limit > 0:
        rows = rows[:limit]
    return rows


def _parents_to_list(parents: np.ndarray) -> list[int | None]:
    out: list[int | None] = []
    for p in parents:
        ip = int(p)
        out.append(None if ip < 0 else ip)
    return out


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norm, 1.0e-12)
    return normals.astype(np.float32)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray, face_normals: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norm, 1.0e-12)
    return normals.astype(np.float32)


def _stable_seed(path: Path, index: int, base_seed: int) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(path.resolve()).encode("utf-8"))
    # Deterministic validation/debug sampling must be invariant to manifest
    # ordering. Training uses `random_query=True`, so the index is intentionally
    # ignored here.
    h.update(int(base_seed).to_bytes(8, "little", signed=False))
    return int.from_bytes(h.digest(), "little") % (2**32)


def _rng(path: Path, index: int, *, random_query: bool, seed: int) -> np.random.Generator:
    if random_query:
        return np.random.default_rng()
    return np.random.default_rng(_stable_seed(path, index, seed))


def _farthest_frames(features: np.ndarray, candidates: np.ndarray, count: int) -> list[int]:
    if count <= 0:
        return []
    if candidates.shape[0] <= count:
        return [int(x) for x in candidates.tolist()]
    cand_feat = features[candidates].astype(np.float32, copy=False)
    chosen_local: list[int] = [int(np.linalg.norm(cand_feat, axis=1).argmax())]
    min_dist = np.sum((cand_feat - cand_feat[chosen_local[0]][None]) ** 2, axis=1)
    min_dist[chosen_local[0]] = -np.inf
    for _ in range(1, count):
        next_local = int(np.argmax(min_dist))
        chosen_local.append(next_local)
        dist = np.sum((cand_feat - cand_feat[next_local][None]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[chosen_local] = -np.inf
    return [int(candidates[i]) for i in chosen_local]


def _select_query_sequence(
    frame_vertices: np.ndarray,
    posed_joints: np.ndarray,
    posed_tails: np.ndarray | None,
    *,
    frame_count: int,
    path: Path,
    index: int,
    random_query: bool,
    seed: int,
    motion_fps_ratio: float,
    motion_vertex_samples: int,
    input_space_policy: str = "mesh_query_bbox",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Select `[query, evidence...]` frames from one clean asset sequence.

    The query frame is the target pose.  Evidence frames are sampled from the
    same asset sequence with a motion-diverse FPS component and a random
    component. If the sequence cannot provide the requested number of frames,
    it should have been rejected during strict cleaning.
    """

    total_frames = int(frame_vertices.shape[0])
    if frame_count <= 1:
        raise ValueError("frame_count must be at least 2 for dynamic-rig training")
    if total_frames < frame_count:
        raise ValueError(f"asset sequence has {total_frames} frames, but requested frame_count={frame_count}")

    rng = _rng(path, index, random_query=random_query, seed=seed)
    query_idx = int(rng.integers(0, total_frames))

    if input_space_policy != "mesh_query_bbox":
        raise ValueError(f"strict rootless training only supports input_space_policy='mesh_query_bbox', got {input_space_policy!r}")
    aligned_frames = np.asarray(frame_vertices, dtype=np.float32)
    aligned_joints = np.asarray(posed_joints, dtype=np.float32)
    aligned_tails = None if posed_tails is None else np.asarray(posed_tails, dtype=np.float32)

    candidates = np.asarray([i for i in range(total_frames) if i != query_idx], dtype=np.int64)
    evidence_count = frame_count - 1
    if candidates.shape[0] < evidence_count:
        raise ValueError("not enough non-query frames for evidence sampling")

    vertex_count = aligned_frames.shape[1]
    sample_count = min(int(motion_vertex_samples), vertex_count)
    sample_ids = rng.choice(vertex_count, size=sample_count, replace=False)
    delta = aligned_frames[:, sample_ids] - aligned_frames[query_idx : query_idx + 1, sample_ids]
    features = delta.reshape(total_frames, -1)

    random_count = int(round((1.0 - motion_fps_ratio) * evidence_count))
    random_count = int(np.clip(random_count, 0, evidence_count))
    fps_count = evidence_count - random_count
    fps_ids = _farthest_frames(features, candidates, fps_count)
    remaining = np.asarray([i for i in candidates.tolist() if int(i) not in set(fps_ids)], dtype=np.int64)
    if random_count > 0:
        random_ids = rng.choice(remaining, size=random_count, replace=False).astype(np.int64).tolist()
    else:
        random_ids = []
    selected = np.asarray([query_idx, *fps_ids, *[int(x) for x in random_ids]], dtype=np.int64)
    if selected.shape[0] != frame_count:
        raise RuntimeError(f"internal sampling error: selected {selected.shape[0]} frames for frame_count={frame_count}")
    target_tails = None if aligned_tails is None else aligned_tails[query_idx]
    return aligned_frames[selected], aligned_joints[query_idx], target_tails, selected


def _derive_tokenizer_tails(joints: np.ndarray, parents: list[int | None]) -> np.ndarray:
    """Build deterministic UniRig-compatibility tails from the joint tree.

    These tails are not data targets. They exist only because the legacy flat
    UniRig tokenizer accepts a per-row tail field.
    """

    pts = np.asarray(joints, dtype=np.float32)
    children: list[list[int]] = [[] for _ in range(int(pts.shape[0]))]
    for child, parent in enumerate(parents):
        if parent is not None:
            children[int(parent)].append(int(child))
    tails = pts.copy()
    for idx, kids in enumerate(children):
        if kids:
            tails[idx] = pts[int(kids[0])]
            continue
        parent = parents[idx]
        if parent is None:
            tails[idx] = pts[idx]
            continue
        direction = pts[idx] - pts[int(parent)]
        if float(np.linalg.norm(direction)) <= 1.0e-8:
            tails[idx] = pts[idx]
        else:
            tails[idx] = pts[idx] + direction
    return tails.astype(np.float32, copy=False)


def _make_token_input(
    joints: np.ndarray,
    parents: list[int | None],
    cls: str,
    tokenizer: Any,
    tails: np.ndarray | None,
) -> np.ndarray:
    from src.tokenizer.spec import TokenizeInput

    parent_joints = []
    branch = []
    child_count = np.zeros((joints.shape[0],), dtype=np.int64)
    for i, p in enumerate(parents):
        if p is not None:
            child_count[p] += 1
        parent_joints.append(joints[i] if p is None else joints[p])
        branch.append(False if i == 0 else p != i - 1)

    bones = np.concatenate([np.asarray(parent_joints, dtype=np.float32), joints.astype(np.float32)], axis=-1)
    if tails is None:
        tails = _derive_tokenizer_tails(joints, parents)
    else:
        tails = np.asarray(tails, dtype=np.float32)
    if tails.shape != joints.shape:
        raise ValueError(f"tails shape {tails.shape} does not match joints shape {joints.shape}")
    token_input = TokenizeInput(
        bones=bones,
        tails=tails,
        branch=np.asarray(branch, dtype=bool),
        is_leaf=child_count == 0,
        no_skin=None,
        cls=cls,
        parts_bias={},
    )
    return tokenizer.tokenize(token_input)


def _make_branch_prior_targets(
    joints: np.ndarray,
    parents: list[int | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return branch-action parent/child coordinate targets.

    UniRig-style serialization emits a branch action when the next child is not
    attached to the immediately previous joint.  The coarse branch prior uses
    these branch actions as supervised proposals: parent/root coordinate plus
    the first child coordinate of the new branch.
    """

    roots: list[np.ndarray] = []
    children: list[np.ndarray] = []
    for child, parent in enumerate(parents):
        if child == 0 or parent is None:
            continue
        if int(parent) != child - 1:
            roots.append(joints[int(parent)].astype(np.float32, copy=False))
            children.append(joints[int(child)].astype(np.float32, copy=False))
    if not roots:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )
    roots_arr = np.stack(roots, axis=0).astype(np.float32)
    children_arr = np.stack(children, axis=0).astype(np.float32)
    return roots_arr, children_arr, np.ones((roots_arr.shape[0],), dtype=bool)


class DynamicRigManifestDataset(Dataset):
    """Strict dynamic-rig asset-sequence manifest dataset."""

    def __init__(
        self,
        manifest: str | Path,
        tokenizer: Any,
        *,
        frame_count: int = 24,
        limit: int = 0,
        cls_name: str = "articulationxl",
        random_query: bool = True,
        seed: int = 20260529,
        motion_fps_ratio: float = 0.7,
        motion_vertex_samples: int = 512,
        target_active_skin_only: bool = False,
        active_skin_threshold: float = 1.0e-4,
        target_start_policy: str = "joint0",
        target_root_policy: str = "legacy",
        input_space_policy: str = "mesh_query_bbox",
    ) -> None:
        self.paths = _load_manifest(Path(manifest), limit=limit)
        self.tokenizer = tokenizer
        self.frame_count = int(frame_count)
        self.cls_name = cls_name
        self.random_query = bool(random_query)
        self.seed = int(seed)
        self.motion_fps_ratio = float(motion_fps_ratio)
        self.motion_vertex_samples = int(motion_vertex_samples)
        if target_active_skin_only:
            raise ValueError("strict rootless dataset does not support target_active_skin_only pruning")
        if float(active_skin_threshold) != 1.0e-4:
            raise ValueError("strict rootless dataset does not use active_skin_threshold")
        if target_start_policy != "joint0":
            raise ValueError(f"strict rootless dataset requires target_start_policy='joint0', got {target_start_policy!r}")
        self.target_start_policy = target_start_policy
        if target_root_policy != "legacy":
            raise ValueError(f"strict rootless dataset requires target_root_policy='legacy', got {target_root_policy!r}")
        self.target_root_policy = target_root_policy
        if input_space_policy != "mesh_query_bbox":
            raise ValueError(f"strict rootless dataset requires input_space_policy='mesh_query_bbox', got {input_space_policy!r}")
        self.input_space_policy = input_space_policy

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> DynamicRigSample:
        path = self.paths[int(index)]
        raw = np.load(path, allow_pickle=True)
        required = [
            "canonical_schema_version",
            "frame_vertices_rootspace",
            "faces",
            "rest_joints",
            "raw_root_index",
            "target_joints_rootspace",
            "target_parents",
            "target_has_skin",
            "target_is_synthetic_root",
        ]
        missing = [name for name in required if name not in raw.files]
        if missing:
            raise KeyError(f"{path} is not a strict rootless training NPZ; missing {missing}")
        schema_arr = np.asarray(raw["canonical_schema_version"])
        schema_version = str(schema_arr.reshape(-1)[0])
        if schema_version not in {"rootless_dynamic_npz_v1", "rootless_dynamic_npz_v2", "rootless_dynamic_npz_v3"}:
            raise ValueError(
                f"{path} canonical_schema_version={schema_version!r}, expected rootless_dynamic_npz_v1/v2/v3"
            )

        raw_frame_vertices = np.asarray(raw["frame_vertices_rootspace"], dtype=np.float32)
        faces = np.asarray(raw["faces"], dtype=np.int64)
        bone_heads = np.asarray(raw["rest_joints"], dtype=np.float32)
        source_joint_count = int(bone_heads.shape[0])
        target_start_index = 0
        alignment_root_index = int(np.asarray(raw["raw_root_index"], dtype=np.int64))
        target_parent_array = np.asarray(raw["target_parents"], dtype=np.int64).reshape(-1)
        posed_joints = np.asarray(raw["target_joints_rootspace"], dtype=np.float32)
        if posed_joints.ndim != 3 or posed_joints.shape[0] != raw_frame_vertices.shape[0] or posed_joints.shape[-1] != 3:
            raise ValueError(f"{path} target_joints_rootspace shape {posed_joints.shape} does not match frames")
        if target_parent_array.shape != (posed_joints.shape[1],):
            raise ValueError(
                f"{path} target_parents shape {target_parent_array.shape} does not match target joints {posed_joints.shape[1]}"
            )
        roots = [int(i) for i, p in enumerate(target_parent_array.tolist()) if int(p) < 0]
        if len(roots) != 1:
            raise ValueError(f"{path} rootless target_parents roots={roots}, expected exactly one")
        target_has_skin = np.asarray(raw["target_has_skin"], dtype=bool).reshape(-1)
        if target_has_skin.shape != (posed_joints.shape[1],):
            raise ValueError(f"{path} target_has_skin shape {target_has_skin.shape} != target joints {posed_joints.shape[1]}")
        target_is_synthetic_root = np.asarray(raw["target_is_synthetic_root"], dtype=bool).reshape(-1)
        if target_is_synthetic_root.shape != (posed_joints.shape[1],):
            raise ValueError(
                f"{path} target_is_synthetic_root shape {target_is_synthetic_root.shape} != target joints {posed_joints.shape[1]}"
            )
        if bool(target_is_synthetic_root.any()):
            raise ValueError(f"{path} target still contains synthetic root entries")
        active_skin_joint_count = int(target_has_skin.sum())
        posed_tails = None
        frame_vertices, target_joints, target_tails, _selected = _select_query_sequence(
            raw_frame_vertices,
            posed_joints,
            posed_tails,
            frame_count=self.frame_count,
            path=path,
            index=int(index),
            random_query=self.random_query,
            seed=self.seed,
            motion_fps_ratio=self.motion_fps_ratio,
            motion_vertex_samples=self.motion_vertex_samples,
            input_space_policy="mesh_query_bbox",
        )

        # Normalize by the query mesh bbox. Root removal has already been done
        # by the cleaned rootless NPZ builder; the model reader does not derive
        # or rewrite roots.
        query_vertices = frame_vertices[0]
        lo = query_vertices.min(axis=0)
        hi = query_vertices.max(axis=0)
        query_center = ((lo + hi) * 0.5).astype(np.float32)
        query_scale = float(np.max((hi - lo) * 0.5))
        if not np.isfinite(query_scale) or query_scale < 1.0e-8:
            raise ValueError(f"{path} query mesh has degenerate bbox after query-root alignment")
        frames_n = ((frame_vertices - query_center) / query_scale).astype(np.float32)
        target_parents = _parents_to_list(target_parent_array)
        joints_n = ((target_joints - query_center) / query_scale).astype(np.float32)
        tails_n = _derive_tokenizer_tails(joints_n, target_parents)

        face_normals_seq = np.stack([_face_normals(frame, faces) for frame in frames_n], axis=0)
        vertex_normals_seq = np.stack(
            [_vertex_normals(frame, faces, face_normals_seq[i]) for i, frame in enumerate(frames_n)],
            axis=0,
        )

        tokens = _make_token_input(joints_n, target_parents, self.cls_name, self.tokenizer, tails=tails_n)
        branch_roots, branch_children, branch_mask = _make_branch_prior_targets(joints_n, target_parents)

        return DynamicRigSample(
            path=str(path),
            cls=self.cls_name,
            frame_vertices=torch.from_numpy(frames_n),
            faces=torch.from_numpy(faces),
            vertex_normals=torch.from_numpy(vertex_normals_seq),
            face_normals=torch.from_numpy(face_normals_seq),
            input_ids=torch.from_numpy(tokens.astype(np.int64)),
            attention_mask=torch.ones((tokens.shape[0],), dtype=torch.float32),
            target_joints=torch.from_numpy(joints_n.astype(np.float32)),
            target_parents=torch.from_numpy(np.asarray([-1 if p is None else int(p) for p in target_parents], dtype=np.int64)),
            branch_prior_roots=torch.from_numpy(branch_roots),
            branch_prior_children=torch.from_numpy(branch_children),
            branch_prior_mask=torch.from_numpy(branch_mask),
            selected_frames=torch.from_numpy(_selected.astype(np.int64)),
            query_center=torch.from_numpy(query_center),
            query_scale=torch.tensor(float(query_scale), dtype=torch.float32),
            joint_count=int(joints_n.shape[0]),
            source_joint_count=source_joint_count,
            active_skin_joint_count=active_skin_joint_count,
            target_start_index=int(target_start_index),
            alignment_root_index=int(alignment_root_index),
            vertex_count=int(raw_frame_vertices.shape[1]),
            face_count=int(faces.shape[0]),
        )


def _pad_tensor_1d(values: list[torch.Tensor], pad_value: int | float = 0) -> torch.Tensor:
    max_len = max(int(value.shape[0]) for value in values)
    out = values[0].new_full((len(values), max_len), pad_value)
    for i, value in enumerate(values):
        out[i, : value.shape[0]] = value
    return out


def _pad_frame_vertices(values: list[torch.Tensor]) -> torch.Tensor:
    frame_count = int(values[0].shape[0])
    max_vertices = max(int(value.shape[1]) for value in values)
    out = values[0].new_zeros((len(values), frame_count, max_vertices, 3))
    for i, value in enumerate(values):
        if int(value.shape[0]) != frame_count:
            raise ValueError("all batch items must have the same sampled frame_count")
        out[i, :, : value.shape[1]] = value
    return out


def _pad_frame_faces(values: list[torch.Tensor]) -> torch.Tensor:
    frame_count = int(values[0].shape[0])
    max_faces = max(int(value.shape[1]) for value in values)
    out = values[0].new_zeros((len(values), frame_count, max_faces, 3))
    for i, value in enumerate(values):
        if int(value.shape[0]) != frame_count:
            raise ValueError("all batch items must have the same sampled frame_count")
        out[i, :, : value.shape[1]] = value
    return out


def _pad_faces(values: list[torch.Tensor]) -> torch.Tensor:
    max_faces = max(int(value.shape[0]) for value in values)
    out = values[0].new_zeros((len(values), max_faces, 3))
    for i, value in enumerate(values):
        out[i, : value.shape[0]] = value
    return out


def _pad_branch_targets(values: list[torch.Tensor]) -> torch.Tensor:
    max_items = max(int(value.shape[0]) for value in values)
    trailing = tuple(values[0].shape[1:])
    out = values[0].new_zeros((len(values), max_items, *trailing))
    for i, value in enumerate(values):
        out[i, : value.shape[0]] = value
    return out


def dynamic_rig_collate(batch: list[DynamicRigSample], pad_token: int) -> dict[str, Any]:
    input_ids = _pad_tensor_1d([item.input_ids for item in batch], pad_value=pad_token)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    for i, item in enumerate(batch):
        attention_mask[i, : item.input_ids.shape[0]] = 1.0

    return {
        "path": [item.path for item in batch],
        "cls": [item.cls for item in batch],
        "frame_vertices": _pad_frame_vertices([item.frame_vertices for item in batch]),
        "faces": _pad_faces([item.faces for item in batch]),
        "vertex_normals": _pad_frame_vertices([item.vertex_normals for item in batch]),
        "face_normals": _pad_frame_faces([item.face_normals for item in batch]),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_joints": _pad_branch_targets([item.target_joints for item in batch]),
        "target_parents": _pad_tensor_1d([item.target_parents for item in batch], pad_value=-2),
        "branch_prior_roots": _pad_branch_targets([item.branch_prior_roots for item in batch]),
        "branch_prior_children": _pad_branch_targets([item.branch_prior_children for item in batch]),
        "branch_prior_mask": _pad_branch_targets([item.branch_prior_mask.to(dtype=torch.float32) for item in batch]).to(dtype=torch.bool),
        "selected_frames": torch.stack([item.selected_frames for item in batch], dim=0),
        "query_center": torch.stack([item.query_center for item in batch], dim=0),
        "query_scale": torch.stack([item.query_scale for item in batch], dim=0),
        "joint_count": torch.tensor([item.joint_count for item in batch], dtype=torch.long),
        "source_joint_count": torch.tensor([item.source_joint_count for item in batch], dtype=torch.long),
        "active_skin_joint_count": torch.tensor([item.active_skin_joint_count for item in batch], dtype=torch.long),
        "target_start_index": torch.tensor([item.target_start_index for item in batch], dtype=torch.long),
        "alignment_root_index": torch.tensor([item.alignment_root_index for item in batch], dtype=torch.long),
        "vertex_count": torch.tensor([item.vertex_count for item in batch], dtype=torch.long),
        "face_count": torch.tensor([item.face_count for item in batch], dtype=torch.long),
        "pad_token": pad_token,
    }
