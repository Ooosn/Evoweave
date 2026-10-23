"""CPU-only tests for the first nonfinite backward-node diagnostic."""
import json
from pathlib import Path
import sys
import unittest
import warnings

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from legacy_gradient_trace import install_nonfinite_gradient_hooks


class InfGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float('inf'))


class SparseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, nonfinite):
        ctx.nonfinite = bool(nonfinite)
        return value.sum()

    @staticmethod
    def backward(ctx, gradient):
        values = gradient.new_tensor([float('inf') if ctx.nonfinite else 2.])
        return torch.sparse_coo_tensor([[1]], values, (3,)), None


class GradientTraceTests(unittest.TestCase):
    def install(self, loss, reports):
        handles = install_nonfinite_gradient_hooks(loss, reports.append)
        self.addCleanup(lambda: [handle.remove() for handle in handles])
        return handles

    def test_inf_is_caught_before_downstream_gelu_backward(self):
        value = torch.tensor([1., 2.], requires_grad=True)
        gelu = F.gelu(value)
        reached_gelu = []
        handle = gelu.grad_fn.register_hook(lambda *args: reached_gelu.append(True))
        self.addCleanup(handle.remove)
        loss = InfGradient.apply(gelu).sum()
        reports = []
        self.install(loss, reports)
        with self.assertRaisesRegex(RuntimeError, 'InfGradientBackward'):
            loss.backward()
        self.assertEqual(reached_gelu, [])
        self.assertIsNone(value.grad)
        self.assertEqual(len(reports), 1)
        report = reports[0]
        self.assertEqual(report['node_name'], 'InfGradientBackward')
        self.assertTrue(report['incoming_grad_outputs_finite'])
        self.assertEqual(report['nonfinite_grad_input_indices'], [0])
        self.assertEqual(report['grad_inputs'][0]['shape'], [2])
        self.assertEqual(report['grad_inputs'][0]['dtype'], 'torch.float32')
        self.assertEqual(report['grad_inputs'][0]['inf_count'], 2)
        self.assertEqual(report['grad_inputs'][0]['nan_count'], 0)
        json.dumps(report, allow_nan=False)

    def test_finite_graph_and_shared_node_deduplication(self):
        value = torch.tensor([.2, .4], requires_grad=True)
        shared = value.sin()
        loss = (shared * shared + shared).sum()
        reports = []
        handles = self.install(loss, reports)
        self.assertEqual(len(handles), 5)  # Sum, Add, Mul, Sin, AccumulateGrad.
        loss.backward()
        torch.testing.assert_close(value.grad, (2 * value.detach().sin() + 1) * value.detach().cos())
        self.assertEqual(reports, [])

    def test_handles_remove_hooks(self):
        value = torch.tensor(1., requires_grad=True)
        loss = InfGradient.apply(value)
        reports = []
        for handle in self.install(loss, reports):
            handle.remove()
        loss.backward()
        self.assertTrue(torch.isinf(value.grad).item())
        self.assertEqual(reports, [])

    def test_sparse_inf_and_none_input_are_reported(self):
        value = torch.ones(3, requires_grad=True)
        loss = SparseGradient.apply(value, torch.tensor(True))
        reports = []
        self.install(loss, reports)
        with self.assertRaisesRegex(RuntimeError, 'SparseGradientBackward'):
            loss.backward()
        summary, absent = reports[0]['grad_inputs']
        self.assertIsNone(absent)
        self.assertEqual(summary['layout'], 'torch.sparse_coo')
        self.assertEqual(summary['shape'], [3])
        self.assertEqual(summary['inspected_numel'], 1)
        self.assertEqual(summary['inf_count'], 1)
        json.dumps(reports[0], allow_nan=False)

    def test_finite_sparse_gradient_passes(self):
        value = torch.ones(3, requires_grad=True)
        loss = SparseGradient.apply(value, torch.tensor(False))
        reports = []
        self.install(loss, reports)
        loss.backward()
        torch.testing.assert_close(value.grad.to_dense(), torch.tensor([0., 2., 0.]))
        self.assertEqual(reports, [])

    def test_anomaly_forward_traceback_is_json_safe(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with torch.autograd.detect_anomaly(check_nan=False):
                loss = InfGradient.apply(torch.tensor(1., requires_grad=True))
        reports = []
        self.install(loss, reports)
        with self.assertRaisesRegex(RuntimeError, 'InfGradientBackward'):
            loss.backward()
        self.assertTrue(reports[0]['forward_traceback'])
        self.assertTrue(all(isinstance(line, str) for line in reports[0]['forward_traceback']))
        json.dumps(reports[0], allow_nan=False)

    def test_nonfinite_incoming_gradient_is_distinguished(self):
        value = torch.ones(2, requires_grad=True)
        loss = value.square().sum()
        reports = []
        self.install(loss, reports)
        with self.assertRaisesRegex(RuntimeError, 'SumBackward0'):
            loss.backward(torch.tensor(float('inf')))
        self.assertFalse(reports[0]['incoming_grad_outputs_finite'])
        self.assertEqual(reports[0]['grad_outputs'][0]['inf_count'], 1)
        self.assertEqual(reports[0]['grad_inputs'][0]['inf_count'], 2)
        json.dumps(reports[0], allow_nan=False)

    def test_nan_counts_are_json_safe(self):
        value = torch.ones(2, requires_grad=True)
        loss = value.square().sum()
        reports = []
        self.install(loss, reports)
        with self.assertRaisesRegex(RuntimeError, 'SumBackward0'):
            loss.backward(torch.tensor(float('nan')))
        self.assertEqual(reports[0]['grad_inputs'][0]['nan_count'], 2)
        self.assertEqual(reports[0]['grad_inputs'][0]['inf_count'], 0)
        json.dumps(reports[0], allow_nan=False)

    def test_requires_a_graph_and_callback(self):
        with self.assertRaisesRegex(ValueError, 'grad_fn'):
            install_nonfinite_gradient_hooks(torch.tensor(1., requires_grad=True), lambda report: None)
        with self.assertRaisesRegex(TypeError, 'callable'):
            install_nonfinite_gradient_hooks(torch.tensor(1., requires_grad=True).square(), None)


if __name__ == '__main__':
    unittest.main()
