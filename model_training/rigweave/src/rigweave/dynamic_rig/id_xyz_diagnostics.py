"""Pure teacher-forced and free-rollout diagnostics for ID+XYZ streams.

The codec remains the grammar authority: complete ground-truth streams are
validated with ``codec.decode`` and optional legal-choice metrics use
``codec.allowed_tokens(prefix)``.  Logits are explicitly next-token aligned:
row ``i`` predicts ``full_ids[i + 1]``.  No model, checkpoint, or state is
loaded or modified here.
"""

from __future__ import annotations

import numbers
from typing import Any

import torch


ROLE_NAMES = (
    "class",
    "new_joint_id",
    "parent_id",
    "branch",
    "eos",
    "x",
    "y",
    "z",
)
_AXIS_ROLES = ("x", "y", "z")


def _token_list(values: Any, name: str) -> list[int]:
    """Convert a one-dimensional integer container without coercing floats."""

    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        values = values.detach().cpu().tolist()
    elif hasattr(values, "ndim") and hasattr(values, "tolist"):
        if int(values.ndim) != 1:
            raise ValueError(f"{name} must be one-dimensional")
        values = values.tolist()
    elif isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a one-dimensional integer sequence")
    else:
        try:
            values = list(values)
        except TypeError as exc:
            raise ValueError(f"{name} must be a one-dimensional integer sequence") from exc

    result: list[int] = []
    for index, value in enumerate(values):
        if not isinstance(value, numbers.Integral) or isinstance(value, (bool,)):
            raise ValueError(f"{name}[{index}] must be an integer")
        value = int(value)
        if value < 0:
            raise ValueError(f"{name}[{index}] must be nonnegative")
        result.append(value)
    return result


def _validated_gt(codec: Any, full_ids: Any, *, ignore_pad: bool) -> tuple[list[int], int]:
    """Validate a complete stream and return it with its non-padding length."""

    values = _token_list(full_ids, "full_ids")
    if not values:
        raise ValueError("full_ids must contain a complete stream")
    try:
        codec.decode(values)
    except ValueError as exc:
        raise ValueError(f"invalid GT ID+XYZ stream: {exc}") from exc

    pad = int(codec.pad)
    pad_positions = [index for index, token in enumerate(values) if token == pad]
    if pad_positions and not ignore_pad:
        raise ValueError("full_ids contains PAD; pass ignore_pad=True to ignore declared padding")
    valid_length = pad_positions[0] if pad_positions else len(values)
    if pad_positions and any(token != pad for token in values[valid_length:]):
        raise ValueError("PAD is only allowed as a trailing declared suffix")
    return values, valid_length


def classify_gt_roles(codec: Any, full_ids: Any, *, ignore_pad: bool = False) -> list[str | None]:
    """Return one role per full-stream position, with BOS/PAD represented by ``None``.

    The returned list is aligned to ``full_ids``.  Position zero is the shared
    BOS and is not a target; position one is the ``class`` target.  Joint-ID
    and coordinate role labels are assigned only after native codec validation.
    """

    values, valid_length = _validated_gt(codec, full_ids, ignore_pad=ignore_pad)
    roles: list[str | None] = [None] * len(values)
    roles[1] = "class"
    coordinate_ids = set(int(token) for token in codec.coordinate_ids)
    joint_offset = int(codec.joint_offset)
    vocab_size = int(codec.vocab_size)
    axis = 0
    for index in range(2, valid_length):
        token = values[index]
        if token == int(codec.eos):
            roles[index] = "eos"
        elif token == int(codec.branch):
            roles[index] = "branch"
        elif token in coordinate_ids:
            roles[index] = _AXIS_ROLES[axis]
            axis = (axis + 1) % 3
        elif joint_offset <= token < vocab_size:
            roles[index] = "parent_id" if values[index - 1] == int(codec.branch) else "new_joint_id"
            axis = 0
        else:
            raise ValueError(f"unclassified valid GT token {token} at position {index}")
    return roles


def _metric(count: int, nll_sum: float, top1_correct: int) -> dict[str, float | int | None]:
    return {
        "count": int(count),
        "nll_sum": float(nll_sum),
        "top1_correct": int(top1_correct),
        "top1_accuracy": (float(top1_correct) / count) if count else None,
    }


def _empty_legal_metric() -> dict[str, float | int | None]:
    return {
        "count": None,
        "nll_sum": None,
        "top1_correct": None,
        "top1_accuracy": None,
        "nontrivial_choice_count": None,
    }


def teacher_forced_diagnostics(
    codec: Any,
    full_ids: Any,
    logits: torch.Tensor,
    *,
    ignore_pad: bool = False,
    legal_choice: bool = False,
) -> dict[str, Any]:
    """Compute full-vocabulary and optional legal-choice next-token metrics.

    ``logits`` must have shape ``[len(full_ids) - 1, vocab]``.  Its row ``i``
    predicts ``full_ids[i + 1]``; a same-length logits tensor is deliberately
    rejected as ambiguous.  The output's ``full_vocab`` metrics never apply a
    grammar mask.  When ``legal_choice=True``, ``legal_choice`` metrics apply
    the native ``codec.allowed_tokens(full_ids[:i])`` set independently at
    every target position.  Declared trailing PAD rows are shape-checked but
    excluded from all metrics.
    """

    values, valid_length = _validated_gt(codec, full_ids, ignore_pad=ignore_pad)
    roles = classify_gt_roles(codec, values, ignore_pad=ignore_pad)
    if not isinstance(logits, torch.Tensor):
        raise ValueError("logits must be a floating PyTorch tensor")
    if logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("logits must have floating shape [len(full_ids)-1, vocab]")
    if logits.shape[0] != len(values) - 1:
        raise ValueError("logits row count must equal len(full_ids)-1 for next-token alignment")
    if logits.shape[1] < int(codec.vocab_size):
        raise ValueError("logits vocabulary must cover the codec vocabulary")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must be finite at every vocabulary entry")

    target_count = valid_length - 1
    target = torch.as_tensor(values[1:valid_length], dtype=torch.long, device=logits.device)
    if target_count <= 0 or bool((target < 0).any()) or bool((target >= logits.shape[1]).any()):
        raise ValueError("valid GT targets must fit the logits vocabulary")

    scores = logits.detach().float()[:target_count]
    log_prob = torch.log_softmax(scores, dim=-1)
    target_log_prob = log_prob.gather(1, target[:, None]).squeeze(1)
    nll = -target_log_prob
    top1 = scores.argmax(dim=-1).eq(target)
    target_roles = roles[1:valid_length]
    if any(role not in ROLE_NAMES for role in target_roles):
        raise ValueError("every non-BOS/non-PAD target must have a known role")

    per_role: dict[str, dict[str, Any]] = {}
    for role in ROLE_NAMES:
        row_indices = [row for row, target_role in enumerate(target_roles) if target_role == role]
        row_tensor = torch.as_tensor(row_indices, dtype=torch.long, device=logits.device)
        role_nll = nll[row_tensor] if row_indices else nll.new_empty((0,))
        role_top1 = top1[row_tensor] if row_indices else top1.new_empty((0,))
        full = _metric(len(row_indices), float(role_nll.sum().item()), int(role_top1.sum().item()))
        per_role[role] = {
            "full_vocab_count": full["count"],
            "full_vocab_nll_sum": full["nll_sum"],
            "full_vocab_top1_correct": full["top1_correct"],
            "full_vocab_top1_accuracy": full["top1_accuracy"],
            "legal_choice": _empty_legal_metric(),
        }

    full = _metric(target_count, float(nll.sum().item()), int(top1.sum().item()))
    legal_summary: dict[str, Any] | None = None
    if legal_choice:
        legal_nll_values: list[torch.Tensor] = []
        legal_top1_values: list[bool] = []
        nontrivial_by_role = {role: 0 for role in ROLE_NAMES}
        legal_nll_by_role = {role: [] for role in ROLE_NAMES}
        legal_top1_by_role = {role: [] for role in ROLE_NAMES}
        for row, target_index in enumerate(range(1, valid_length)):
            allowed = [int(token) for token in codec.allowed_tokens(values[:target_index])]
            if len(set(allowed)) != len(allowed) or not allowed:
                raise ValueError(f"native grammar returned invalid choices at position {target_index}")
            if values[target_index] not in allowed:
                raise ValueError(f"GT target is not legal at position {target_index}")
            if any(token < 0 or token >= logits.shape[1] for token in allowed):
                raise ValueError("native legal choices exceed the logits vocabulary")
            allowed_tensor = torch.as_tensor(allowed, dtype=torch.long, device=logits.device)
            allowed_scores = scores[row].index_select(0, allowed_tensor)
            constrained_log_prob = allowed_scores - torch.logsumexp(allowed_scores, dim=0)
            target_offset = allowed.index(values[target_index])
            legal_nll_value = -constrained_log_prob[target_offset]
            legal_top1_value = bool(int(allowed_tensor[allowed_scores.argmax()].item()) == values[target_index])
            role = target_roles[row]
            legal_nll_values.append(legal_nll_value)
            legal_top1_values.append(legal_top1_value)
            legal_nll_by_role[role].append(legal_nll_value)
            legal_top1_by_role[role].append(legal_top1_value)
            if len(allowed) > 1:
                nontrivial_by_role[role] += 1

        legal_nll = torch.stack(legal_nll_values) if legal_nll_values else nll.new_empty((0,))
        legal_top1_count = sum(legal_top1_values)
        legal_summary = {
            **_metric(target_count, float(legal_nll.sum().item()), legal_top1_count),
            "nontrivial_choice_count": sum(nontrivial_by_role.values()),
        }
        for role in ROLE_NAMES:
            role_nll_values = legal_nll_by_role[role]
            role_nll = torch.stack(role_nll_values) if role_nll_values else nll.new_empty((0,))
            role_top1 = legal_top1_by_role[role]
            role_legal = {
                **_metric(len(role_nll_values), float(role_nll.sum().item()), sum(role_top1)),
                "nontrivial_choice_count": nontrivial_by_role[role],
            }
            per_role[role]["legal_choice"] = role_legal

    return {
        "sequence_length": len(values),
        "valid_length": valid_length,
        "ignored_pad_count": len(values) - valid_length,
        "target_count": target_count,
        "role_names": ROLE_NAMES,
        "target_roles": roles,
        "full_vocab": full,
        "per_role": per_role,
        "legal_choice": legal_summary,
    }


def first_divergence_summary(
    codec: Any,
    generated_tokens: Any,
    gt_tokens: Any,
    *,
    ignore_pad: bool = False,
) -> dict[str, Any]:
    """Summarize the first positional free-rollout mismatch after BOS/CLS.

    This intentionally compares saved lists by absolute token position.  It
    does not align insertions, infer semantic joint correspondence, or claim
    that a later matching token recovered from an earlier error.  GT is
    native-codec validated; generated tokens may be malformed because this is
    a failure summary rather than a parser.
    """

    gt_values, valid_length = _validated_gt(codec, gt_tokens, ignore_pad=ignore_pad)
    gt_values = gt_values[:valid_length]
    generated = _token_list(generated_tokens, "generated_tokens")
    if int(codec.pad) in generated:
        raise ValueError("generated_tokens must not contain PAD")
    roles = classify_gt_roles(codec, gt_values)
    common = {
        "generated_length": len(generated),
        "gt_length": len(gt_values),
        "compared_from": 2,
        "positional_only": True,
    }
    if len(generated) < 2 or len(gt_values) < 2:
        return {**common, "status": "uncomparable_prefix", "index": None, "role": None}
    for index in range(min(2, len(generated), len(gt_values))):
        if generated[index] != gt_values[index]:
            return {
                **common,
                "status": "prefix_mismatch",
                "index": index,
                "role": None,
                "generated_token": generated[index],
                "gt_token": gt_values[index],
            }
    for index in range(2, min(len(generated), len(gt_values))):
        if generated[index] != gt_values[index]:
            return {
                **common,
                "status": "first_mismatch",
                "index": index,
                "role": roles[index],
                "generated_token": generated[index],
                "gt_token": gt_values[index],
            }
    if len(generated) < len(gt_values):
        index = len(generated)
        return {
            **common,
            "status": "generated_ended_early",
            "index": index,
            "role": roles[index],
            "generated_token": None,
            "gt_token": gt_values[index],
        }
    if len(generated) > len(gt_values):
        index = len(gt_values)
        return {
            **common,
            "status": "generated_has_extra_tokens",
            "index": index,
            "role": None,
            "generated_token": generated[index],
            "gt_token": None,
        }
    return {**common, "status": "match", "index": None, "role": None}


diagnose_teacher_forced = teacher_forced_diagnostics
summarize_first_divergence = first_divergence_summary


__all__ = [
    "ROLE_NAMES",
    "classify_gt_roles",
    "teacher_forced_diagnostics",
    "diagnose_teacher_forced",
    "first_divergence_summary",
    "summarize_first_divergence",
]
