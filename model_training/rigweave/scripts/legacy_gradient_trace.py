"""First-observed nonfinite autograd-node diagnostics, without tensor payloads."""
from __future__ import annotations

import json

import torch


def _gradient_summary(gradient, *, include_range=False):
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
    if include_range and not nonfinite:
        if value.is_complex():
            summary.update(min=None, max=None, range_error='complex gradients have no ordered scalar range')
        elif not gradient.numel():
            summary.update(min=None, max=None)
        else:
            low, high = (value.amin().item(), value.amax().item()) if value.numel() else (0, 0)
            if gradient.layout != torch.strided and gradient.numel() > value.numel():
                low, high = min(low, 0), max(high, 0)
            summary.update(min=low, max=high)
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


def install_nonfinite_gradient_hooks(loss, report_callback, *, tensor_callback=None):
    """Return removable node hooks; report once, then raise on nonfinite grad_inputs.

    The first node observed by the autograd engine is reported, not necessarily
    the numerical root cause. Sparse counts cover stored/coalesced values;
    implicit zeros are finite. Remove every returned handle after diagnostics.
    An optional tensor_callback receives raw gradient tuples at the first bad
    node only, and must return a JSON-safe dictionary. Its errors are reported.
    """
    if not isinstance(loss, torch.Tensor) or loss.grad_fn is None:
        raise ValueError('loss must be a tensor with an autograd grad_fn')
    if not callable(report_callback):
        raise TypeError('report_callback must be callable')
    if tensor_callback is not None and not callable(tensor_callback):
        raise TypeError('tensor_callback must be callable or None')

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
            outputs = [_gradient_summary(gradient, include_range=True) for gradient in grad_outputs]
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
            if tensor_callback is not None and bad:
                report['tensor_details'] = None
                try:
                    details = tensor_callback(name, grad_inputs, grad_outputs)
                    if not isinstance(details, dict):
                        raise TypeError('tensor_callback must return a JSON-safe dictionary')
                    json.dumps(details, allow_nan=False)
                    report['tensor_details'] = details
                except Exception as exc:
                    report['tensor_details_error'] = {'type': type(exc).__name__, 'message': str(exc)[:500]}
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
