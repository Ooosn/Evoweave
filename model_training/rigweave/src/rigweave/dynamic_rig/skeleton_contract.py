"""Skeleton target contract helpers.

This module defines the data-side skeleton contract used by the dynamic rig
dataset.  It intentionally works only from explicit parent indices and skin
weights; it never assumes joint 0 is the true root.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


SYNTHETIC_ROOT_NAME = "__object_root__"


@dataclass(frozen=True)
class SyntheticRootContract:
    """Mapping from a raw armature tree to the normalized training target tree."""

    raw_root: int
    raw_order: list[int]
    parents: np.ndarray
    names: list[str]
    kept_raw_indices: list[int]
    dropped_raw_indices: list[int]
    active_raw_indices: list[int]
    entry_raw_indices: list[int]
    synthetic_child_count: int


def parent_array(values: np.ndarray | Sequence[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"parents must be 1D, got shape {arr.shape}")
    return arr


def children_from_parents(parents: np.ndarray | Sequence[int]) -> list[list[int]]:
    arr = parent_array(parents)
    children: list[list[int]] = [[] for _ in range(int(arr.shape[0]))]
    for child, parent in enumerate(arr.tolist()):
        p = int(parent)
        if p >= 0:
            if p >= len(children):
                raise ValueError(f"invalid parent index {p} for child {child}")
            if p == child:
                raise ValueError(f"self parent at joint {child}")
            children[p].append(int(child))
    return children


def single_root_index(parents: np.ndarray | Sequence[int], *, multi_root_policy: str = "reject") -> int:
    arr = parent_array(parents)
    roots = np.flatnonzero(arr < 0)
    if roots.shape[0] == 1:
        return int(roots[0])
    if multi_root_policy == "reject":
        raise ValueError(f"expected exactly one raw root, got {roots.tolist()}")
    raise ValueError(f"unsupported multi_root_policy={multi_root_policy!r}")


def active_skin_mask(skin_weights: np.ndarray, threshold: float) -> np.ndarray:
    skin = np.asarray(skin_weights, dtype=np.float32)
    if skin.ndim != 2:
        raise ValueError(f"skin_weights must be 2D, got shape {skin.shape}")
    return skin.max(axis=0) > float(threshold)


def _normal_name(name: str) -> str:
    text = str(name).strip().lower()
    text = text.split(":")[-1]
    for char in ".- ":
        text = text.replace(char, "_")
    return text


def looks_like_dummy_root(name: str) -> bool:
    text = _normal_name(name)
    if text in {"root", "rt", "armature", "scene", "origin", "object", "object_root"}:
        return True
    return text.startswith("root_") or text.endswith("_root") or text.startswith("rt_")


def looks_like_tail_or_end(name: str) -> bool:
    text = _normal_name(name)
    return (
        text in {"tail", "end"}
        or text.endswith("_tail")
        or text.endswith("_end")
        or text.endswith("tail")
        or text.endswith("end")
    )


def active_descendant_mask(parents: np.ndarray, active: np.ndarray) -> np.ndarray:
    arr = parent_array(parents)
    if active.shape[0] != arr.shape[0]:
        raise ValueError("active mask and parents length differ")
    has_active = active.astype(bool, copy=True)
    for node in np.flatnonzero(active).tolist():
        p = int(arr[int(node)])
        while p >= 0:
            if has_active[p]:
                # Ancestors above this point have already been marked from this
                # or another active descendant.
                pass
            has_active[p] = True
            p = int(arr[p])
    return has_active


def _active_child_count(children: list[list[int]], has_active: np.ndarray, node: int) -> int:
    return sum(1 for child in children[int(node)] if bool(has_active[int(child)]))


def _first_contract_entry(
    start: int,
    *,
    children: list[list[int]],
    active: np.ndarray,
    has_active: np.ndarray,
    names: Sequence[str],
) -> int:
    node = int(start)
    while True:
        if bool(active[node]):
            return node
        if looks_like_tail_or_end(names[node]) and bool(children[node]):
            return node
        active_children = [int(child) for child in children[node] if bool(has_active[int(child)])]
        if len(active_children) != 1:
            return node
        node = int(active_children[0])


def build_synthetic_root_contract(
    parents: np.ndarray | Sequence[int],
    skin_weights: np.ndarray,
    names: Sequence[str] | None = None,
    *,
    active_skin_threshold: float = 1.0e-4,
    multi_root_policy: str = "reject",
) -> SyntheticRootContract:
    """Build the target skeleton under a synthetic origin root.

    Contract:
    - raw root is found from parent indices; joint 0 is never assumed root.
    - multiple raw roots are rejected by default.
    - target index 0 is always ``__object_root__`` with parent -1.
    - every active/skinned raw joint is retained.
    - unskinned ancestors required to connect retained joints are retained,
      except the disposable top root chain replaced by the synthetic root.
    - real tail/end joints are retained when they are active or parent a
      retained descendant.
    """

    arr = parent_array(parents)
    n = int(arr.shape[0])
    if names is None:
        name_list = [str(i) for i in range(n)]
    else:
        name_list = [str(x) for x in names]
        if len(name_list) != n:
            raise ValueError(f"names length {len(name_list)} != parent count {n}")
    active = active_skin_mask(skin_weights, active_skin_threshold)
    if active.shape[0] != n:
        raise ValueError(f"skin joint count {active.shape[0]} != parent count {n}")
    active_ids = [int(i) for i in np.flatnonzero(active).tolist()]
    if not active_ids:
        raise ValueError("no active/skinned joints found")

    root = single_root_index(arr, multi_root_policy=multi_root_policy)
    children = children_from_parents(arr)
    has_active = active_descendant_mask(arr, active)

    # Synthetic root replaces a disposable raw root.  If the raw root is itself
    # skinned or semantically real, keep it as a child of the synthetic root.
    entries: list[int] = []
    if bool(active[root]) or not looks_like_dummy_root(name_list[root]):
        entries = [
            _first_contract_entry(
                root,
                children=children,
                active=active,
                has_active=has_active,
                names=name_list,
            )
        ]
    else:
        for child in children[root]:
            if bool(has_active[int(child)]):
                entries.append(
                    _first_contract_entry(
                        int(child),
                        children=children,
                        active=active,
                        has_active=has_active,
                        names=name_list,
                    )
                )
        if not entries:
            # This should be unreachable because active_ids is non-empty, but
            # keep the error explicit if a malformed parent tree appears.
            raise ValueError("dummy root has no active child subtree")

    entry_set = set(entries)
    keep: set[int] = set()
    for active_id in active_ids:
        path: list[int] = []
        node = int(active_id)
        while node >= 0:
            path.append(node)
            if node in entry_set:
                break
            node = int(arr[node])
        if not path or path[-1] not in entry_set:
            raise ValueError(f"active joint {active_id} is not under a target entry")
        keep.update(path)

    raw_order: list[int] = []

    def visit(node: int) -> None:
        if node not in keep:
            return
        raw_order.append(int(node))
        for child in children[int(node)]:
            visit(int(child))

    for entry in entries:
        visit(int(entry))

    old_to_new = {raw: new + 1 for new, raw in enumerate(raw_order)}
    target_parents = np.full((len(raw_order) + 1,), -1, dtype=np.int64)
    target_names = [SYNTHETIC_ROOT_NAME]
    for raw in raw_order:
        p = int(arr[int(raw)])
        while p >= 0 and p not in old_to_new:
            p = int(arr[p])
        target_parents[old_to_new[int(raw)]] = 0 if p < 0 else int(old_to_new[p])
        target_names.append(name_list[int(raw)])

    kept = sorted(int(x) for x in keep)
    dropped = [int(i) for i in range(n) if i not in keep]
    return SyntheticRootContract(
        raw_root=int(root),
        raw_order=[int(x) for x in raw_order],
        parents=target_parents,
        names=target_names,
        kept_raw_indices=kept,
        dropped_raw_indices=dropped,
        active_raw_indices=active_ids,
        entry_raw_indices=[int(x) for x in entries],
        synthetic_child_count=int(sum(1 for p in target_parents.tolist() if int(p) == 0)),
    )


def apply_synthetic_root_to_points(
    joints: np.ndarray,
    tails: np.ndarray,
    contract: SyntheticRootContract,
    *,
    synthetic_root_xyz: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Apply a synthetic-root contract to already-normalized joint/tail arrays."""

    joints_arr = np.asarray(joints, dtype=np.float32)
    tails_arr = np.asarray(tails, dtype=np.float32)
    if joints_arr.ndim != 2 or joints_arr.shape[-1] != 3:
        raise ValueError(f"joints must have shape (N,3), got {joints_arr.shape}")
    if tails_arr.shape != joints_arr.shape:
        raise ValueError(f"tails shape {tails_arr.shape} does not match joints {joints_arr.shape}")
    root = np.asarray(synthetic_root_xyz, dtype=np.float32).reshape(1, 3)
    order = np.asarray(contract.raw_order, dtype=np.int64)
    target_joints = np.concatenate([root, joints_arr[order]], axis=0).astype(np.float32, copy=False)
    target_tails = np.concatenate([root.copy(), tails_arr[order]], axis=0).astype(np.float32, copy=False)
    return target_joints, target_tails, contract.parents.copy(), list(contract.names)


def validate_synthetic_root_target(
    joints: np.ndarray,
    parents: np.ndarray,
    *,
    atol: float = 1.0e-7,
) -> list[str]:
    """Return human-readable invariant violations for a synthetic-root target."""

    issues: list[str] = []
    arr = parent_array(parents)
    pts = np.asarray(joints, dtype=np.float32)
    if arr.shape[0] != pts.shape[0]:
        issues.append(f"parent count {arr.shape[0]} != joint count {pts.shape[0]}")
        return issues
    if arr.shape[0] == 0:
        issues.append("empty target")
        return issues
    if int(arr[0]) != -1:
        issues.append(f"synthetic root parent is {int(arr[0])}, expected -1")
    if not np.allclose(pts[0], np.zeros((3,), dtype=np.float32), atol=atol):
        issues.append(f"synthetic root xyz is {pts[0].tolist()}, expected [0,0,0]")
    roots = [int(i) for i, p in enumerate(arr.tolist()) if int(p) < 0]
    if roots != [0]:
        issues.append(f"target roots are {roots}, expected [0]")
    for child, parent in enumerate(arr.tolist()):
        p = int(parent)
        if child == 0:
            continue
        if p < 0 or p >= child:
            issues.append(f"invalid parent edge child={child} parent={p}")
    return issues
