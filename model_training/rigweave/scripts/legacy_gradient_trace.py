"""First-observed nonfinite autograd-node diagnostics, without tensor payloads."""
from __future__ import annotations

import torch


def _gradient_summary(gradient):
    if gradient is None:
        return None
    if not isinstance(gradient, torch.Tensor):
        return {'inspection_error': f'unsupported gradient type: {type(gradient).__name__}'}
    summary = {'shape': list(gradient.shape), 'dtype': str(gradient.dtype),
               'layout': str(gradient.layout), 'numel': gradient.numel()}
    value = gradient.detach()
    if value.layout == torch.sparse_coo:
        value = value.coalesce().values()
    elif value.layout in (torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc):
        value = value.values()
    elif value.layout != torch.strided:
        summary['inspection_error'] = f'unsupported gradient layout: {value.layout}'
        return summary
    summary['inspected_numel'] = value.numel()
    nonfinite = int((~torch.isfinite(value)).sum().item())
    summary.update(finite=nonfinite == 0, nonfinite_count=nonfinite,
                   nan_count=int(torch.isnan(value).sum().item()) if nonfinite else 0,
                   inf_count=int(torch.isinf(value).sum().item()) if nonfinite else 0)
    return summary


def _forward_traceback(node):
    trace = node.metadata.get('traceback_')
    if isinstance(trace, str):
        trace = trace.splitlines(keepends=True)
    if not isinstance(trace, (list, tuple)) or not all(isinstance(line, str) for line in trace):
        return {}
    selected = [line[:500] for line in trace[-12:]]
    return {'forward_traceback': selected,
            'forward_traceback_truncated': len(trace) > 12 or any(len(line) > 500 for line in trace[-12:])}


def install_nonfinite_gradient_hooks(loss, report_callback):
    """Return removable node hooks; report once, then raise on nonfinite grad_inputs.

    The first node observed by the autograd engine is reported, not necessarily
    the numerical root cause. Sparse counts cover stored/coalesced values;
    implicit zeros are finite. Remove every returned handle after diagnostics.
    """
    if not isinstance(loss, torch.Tensor) or loss.grad_fn is None:
        raise ValueError('loss must be a tensor with an autograd grad_fn')
    if not callable(report_callback):
        raise TypeError('report_callback must be callable')

    # Retain wrappers throughout traversal so Python ids cannot be recycled.
    nodes, seen, pending = [], set(), [loss.grad_fn]
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        nodes.append(node)
        pending.extend(parent for parent, _ in node.next_functions if parent is not None)

    reported = False

    def make_hook(node):
        name, traceback = node.name(), _forward_traceback(node)

        def hook(grad_inputs, grad_outputs):
            nonlocal reported
            if reported:
                return None
            inputs = [_gradient_summary(gradient) for gradient in grad_inputs]
            bad = [i for i, item in enumerate(inputs) if item is not None and item.get('finite') is False]
            errors = [i for i, item in enumerate(inputs) if item is not None and 'inspection_error' in item]
            if not bad and not errors:
                return None
            outputs = [_gradient_summary(gradient) for gradient in grad_outputs]
            output_values = [item for item in outputs if item is not None]
            outputs_finite = (None if any('inspection_error' in item for item in output_values)
                              else all(item['finite'] for item in output_values))
            if reported:
                return None
            reported = True
            event = 'gradient_inspection_failed' if errors else 'nonfinite_backward_gradient'
            report = {'event': event, 'node_name': name, 'grad_inputs': inputs,
                      'grad_outputs': outputs, 'incoming_grad_outputs_finite': outputs_finite,
                      'nonfinite_grad_input_indices': bad, 'uninspectable_grad_input_indices': errors,
                      **traceback}
            message = f'{event} at {name}; grad_inputs={bad or errors}'
            try:
                report_callback(report)
            except Exception as exc:
                raise RuntimeError(message + '; report callback failed') from exc
            raise RuntimeError(message)

        return hook

    handles = []
    try:
        for node in nodes:
            handles.append(node.register_hook(make_hook(node)))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles
