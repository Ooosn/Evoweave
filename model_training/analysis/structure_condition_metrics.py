"""NumPy metrics for fixed-legal-prefix conditional counterfactuals.

The partner condition is not an inference candidate. These comparisons retain
an identical GT or generated prefix and test conditional sensitivity, not generation quality.
All logarithms, cross-entropies and Jensen-Shannon divergences use natural logs.
"""

from __future__ import annotations

import numpy as np


CATEGORIES = ('new_joint_id', 'branch', 'parent_reference', 'eos')


def build_prefix_layout(tokens, codec) -> dict:
    """Validate a complete BOS...EOS stream using the model's actual grammar.

    Row i scores tokens[i + 1] given tokens[:i + 1]. Choice means more than
    one grammar-legal next token, including for parent references and EOS.
    """
    from rigweave.dynamic_rig.structure_xyz_fill import allowed_structure_tokens

    values = np.asarray(tokens)
    if values.ndim != 1 or values.dtype.kind not in 'iu':
        raise ValueError('tokens must be a one-dimensional integer sequence')
    if len(values) < 3 or values[0] != codec.bos or values[-1] != codec.eos:
        raise ValueError('tokens must contain a complete BOS...EOS skeleton')
    if len(values) > codec.max_sequence_tokens:
        raise ValueError('complete token sequence exceeds codec capacity')
    if np.any(values < 0) or np.any(values >= codec.vocab_size):
        raise ValueError('token is outside codec vocabulary capacity')
    values = values.astype(np.int64).tolist()
    allowed = np.zeros((len(values) - 1, codec.vocab_size), dtype=np.bool_)
    categories = []
    for index, label in enumerate(values[1:]):
        legal = allowed_structure_tokens(values[:index + 1], codec)
        if label not in legal:
            raise ValueError(f'illegal GT next token at position {index + 1}: {label}')
        allowed[index, legal] = True
        if values[index] == codec.branch:
            category = 'parent_reference'
        elif label == codec.branch:
            category = 'branch'
        elif label == codec.eos:
            category = 'eos'
        else:
            category = 'new_joint_id'
        categories.append(category)
    if allowed_structure_tokens(values, codec):
        raise ValueError('complete structure did not terminate with EOS')
    return {'labels': np.asarray(values[1:], dtype=np.int64), 'categories': categories,
            'allowed': allowed, 'choice': allowed.sum(axis=1) > 1}


def _validate_layout(layout):
    labels = np.asarray(layout['labels'])
    allowed = np.asarray(layout['allowed'])
    choice = np.asarray(layout['choice'])
    categories = layout['categories']
    if (labels.ndim != 1 or labels.dtype.kind not in 'iu' or allowed.ndim != 2
            or allowed.dtype != np.bool_ or allowed.shape[0] != len(labels)
            or allowed.shape[1] == 0 or choice.shape != labels.shape
            or choice.dtype != np.bool_ or len(categories) != len(labels)
            or any(category not in CATEGORIES for category in categories)):
        raise ValueError('layout labels, categories, allowed mask and choice must align')
    if np.any(labels < 0) or np.any(labels >= allowed.shape[1]):
        raise ValueError('layout label is outside vocabulary')
    if not np.all(allowed.any(axis=1)):
        raise ValueError('every scored prefix must have at least one allowed token')
    if not np.all(allowed[np.arange(len(labels)), labels]):
        raise ValueError('layout label is not legal for its prefix')
    if not np.array_equal(choice, allowed.sum(axis=1) > 1):
        raise ValueError('choice must identify exactly the non-forced positions')
    return labels, allowed, choice, np.asarray(categories)


def _validate_logits(logits, shape):
    values = np.asarray(logits)
    if values.shape != shape or values.dtype.kind not in 'iuf':
        raise ValueError('logits must be a real numeric array matching layout [N,V]')
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError('logits must be finite')
    return values


def _log_probabilities(logits, allowed=None):
    values = logits if allowed is None else np.where(allowed, logits, -np.inf)
    # Subtract before log-sum-exp so large common offsets do not lose precision.
    with np.errstate(over='ignore', under='ignore'):
        shifted = values - values.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _aggregate(columns, selected):
    count = int(selected.sum())
    return {'tokens': count, **{name: float(values[selected].mean()) if count else None
                               for name, values in columns.items()}}


def summarize_logits(logits, layout) -> dict:
    """Report token-weighted raw and grammar-renormalized CE/accuracy.

    Empty aggregates have zero tokens and None metrics. An empty legal set in
    a scored row is invalid, rather than a distribution with NaN probabilities.
    """
    labels, allowed, choice, categories = _validate_layout(layout)
    values = _validate_logits(logits, allowed.shape)
    raw, legal = _log_probabilities(values), _log_probabilities(values, allowed)
    rows = np.arange(len(labels))
    raw_nll, legal_nll = -raw[rows, labels], -legal[rows, labels]
    if not np.isfinite(raw_nll).all() or not np.isfinite(legal_nll).all():
        raise ValueError('target NLL exceeds finite float64 range')
    raw_argmax, legal_argmax = raw.argmax(axis=1), legal.argmax(axis=1)
    columns = {'raw_ce': raw_nll, 'raw_accuracy': raw_argmax == labels,
               'legal_ce': legal_nll, 'legal_accuracy': legal_argmax == labels}
    with np.errstate(under='ignore'):
        raw_probability, legal_probability = np.exp(-raw_nll), np.exp(-legal_nll)
    positions = [
        {'label': int(label), 'category': str(categories[index]), 'choice': bool(choice[index]),
         'raw_nll': float(raw_nll[index]), 'legal_nll': float(legal_nll[index]),
         'raw_probability': float(raw_probability[index]), 'legal_probability': float(legal_probability[index]),
         'raw_argmax': int(raw_argmax[index]), 'legal_argmax': int(legal_argmax[index])}
        for index, label in enumerate(labels)
    ]
    return {'all': _aggregate(columns, np.ones(len(labels), dtype=bool)),
            'choice': _aggregate(columns, choice),
            'by_category': {name: _aggregate(columns, categories == name) for name in CATEGORIES},
            'positions': positions}


def _js_divergence(left, right):
    mixture = np.logaddexp(left, right) - np.log(2.)
    terms = []
    for log_probability in (left, right):
        difference = np.zeros_like(log_probability)
        # Illegal and numerically zero-probability entries contribute zero, not 0*NaN.
        np.subtract(log_probability, mixture, out=difference, where=np.isfinite(log_probability))
        with np.errstate(under='ignore'):
            terms.append((np.exp(log_probability) * difference).sum(axis=1))
    divergence = np.maximum(0., .5 * terms[0] + .5 * terms[1])
    divergence[np.all(left == right, axis=1)] = 0.
    return divergence


def contrast_logits(correct, counter, layout) -> dict:
    """Compare conditions at identical legal prefixes, not inference candidates."""
    labels, allowed, choice, _ = _validate_layout(layout)
    correct = _validate_logits(correct, allowed.shape)
    counter = _validate_logits(counter, allowed.shape)
    raw = [_log_probabilities(values) for values in (correct, counter)]
    legal = [_log_probabilities(values, allowed) for values in (correct, counter)]
    columns = {'raw_js': _js_divergence(*raw), 'legal_js': _js_divergence(*legal),
               'raw_argmax_change_fraction': raw[0].argmax(axis=1) != raw[1].argmax(axis=1),
               'legal_argmax_change_fraction': legal[0].argmax(axis=1) != legal[1].argmax(axis=1)}
    positions = [
        {'raw_js': float(columns['raw_js'][index]), 'legal_js': float(columns['legal_js'][index]),
         'raw_argmax_changed': bool(columns['raw_argmax_change_fraction'][index]),
         'legal_argmax_changed': bool(columns['legal_argmax_change_fraction'][index])}
        for index in range(len(labels))
    ]
    return {'all': _aggregate(columns, np.ones(len(labels), dtype=bool)),
            'choice': _aggregate(columns, choice), 'positions': positions}


def choose_partner_indices(parents: list[list[int]]) -> list[int]:
    """First cyclic different index with a different exact parent tuple.

    This deterministic conditional-counterfactual assignment need not be a
    permutation and does not nominate inference candidates or correct targets.
    """
    rows = []
    for row in parents:
        values = np.asarray(row)
        if values.ndim != 1 or values.dtype.kind not in 'iu' or not len(values):
            raise ValueError('parents must contain nonempty integer rows')
        rows.append(tuple(values.tolist()))
    if len(set(rows)) < 2:
        raise ValueError('each condition requires a partner with a different parent tuple')
    result = []
    for index, row in enumerate(rows):
        result.append(next((index + offset) % len(rows) for offset in range(1, len(rows))
                           if rows[(index + offset) % len(rows)] != row))
    return result
