"""Teacher-forced cross-frame structure consistency for a single ID+XYZ AR stream."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def structural_consistency_ce(logits, labels, structure_mask, frame_mask):
    """Leave-one-out probability-target CE, equally weighted by frame and asset.

    All L positions are NEXT-TOKEN aligned. The caller marks new IDs, parent
    references, BRANCH and EOS, excluding BOS/CLS/XYZ. Labels -100 are ignored.
    Valid frames must share structural positions and labels. Returned position
    counts are per asset (not multiplied by T); KL is a detached CE-minus-target-
    entropy diagnostic and may have tiny floating-point roundoff around zero.
    """
    tensors = (logits, labels, structure_mask, frame_mask)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise ValueError('logits, labels, structure_mask and frame_mask must be tensors')
    if logits.ndim != 4 or not logits.is_floating_point() or any(size < 1 for size in logits.shape):
        raise ValueError('logits must be floating [B,T,L,V] with nonempty dimensions')
    batch, frames, length, vocab = logits.shape
    if labels.shape != (batch, frames, length) or labels.dtype != torch.long:
        raise ValueError('labels must be int64 [B,T,L], aligned to next-token logits')
    if structure_mask.shape != labels.shape or structure_mask.dtype != torch.bool:
        raise ValueError('structure_mask must be bool [B,T,L]')
    if frame_mask.shape != (batch, frames) or frame_mask.dtype != torch.bool:
        raise ValueError('frame_mask must be bool [B,T]')
    if any(value.device != logits.device for value in tensors[1:]):
        raise ValueError('all loss inputs must share one device')
    frame_counts = frame_mask.sum(dim=1)
    if bool((frame_counts < 2).any()):
        raise ValueError('each asset requires at least two valid frames')
    rows = torch.arange(batch, device=logits.device)
    first = frame_mask.long().argmax(dim=1)
    reference_mask = structure_mask[rows, first]
    reference_labels = labels[rows, first]
    if bool(((structure_mask != reference_mask[:, None]) & frame_mask[:, :, None]).any()):
        raise ValueError('structural masks must match across valid frames of each asset')
    structural_mismatch = ((labels != reference_labels[:, None])
                           & reference_mask[:, None] & frame_mask[:, :, None])
    if bool(structural_mismatch.any()):
        raise ValueError('structural labels, including ignored positions, must match across valid frames')
    positions = reference_mask & (reference_labels != -100)
    position_counts = positions.sum(dim=1)
    if bool((position_counts == 0).any()):
        raise ValueError('each asset requires at least one valid structural position, including EOS')
    selected_labels = reference_labels[positions]
    if bool(((selected_labels < 0) | (selected_labels >= vocab)).any()):
        raise ValueError('valid structural labels must belong to the full vocabulary')

    # Pack structural positions before softmax: excluded XYZ/PAD logits never
    # participate, and probability sums produce all LOO targets in O(T), not T^2.
    asset_ids, position_ids = positions.nonzero(as_tuple=True)
    selected = logits[asset_ids, :, position_ids, :]
    valid_frames = frame_mask[asset_ids]
    if not bool(torch.isfinite(selected[valid_frames]).all()):
        raise ValueError('valid structural logits must be finite')
    selected = torch.where(valid_frames[..., None], selected.float(), 0.)
    log_probabilities = F.log_softmax(selected, dim=-1, dtype=torch.float32)
    probabilities = log_probabilities.detach().exp() * valid_frames[..., None]
    targets = ((probabilities.sum(dim=1, keepdim=True) - probabilities)
               / (frame_counts[asset_ids, None, None] - 1))
    ce = -(targets * log_probabilities).sum(dim=-1)
    entropy = -torch.xlogy(targets, targets).sum(dim=-1)

    def average(values):
        per_asset_frame = values.new_zeros(batch, frames).index_add(
            0, asset_ids, values * valid_frames)
        return (per_asset_frame / position_counts[:, None]).sum(dim=1) / frame_counts

    loss_per_asset = average(ce)
    return {'loss': loss_per_asset.mean(), 'loss_per_asset': loss_per_asset,
            'kl_diagnostic': (loss_per_asset.detach() - average(entropy)).mean(),
            'structure_positions_per_asset': position_counts}
