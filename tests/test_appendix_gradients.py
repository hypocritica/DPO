"""Check archived gradient geometry against explicit toy objectives."""

import importlib.util
import math
import unittest

import numpy as np

from self_correct_dpo.evaluation.appendix_gradients import collect_gradient_statistics
from self_correct_dpo.evaluation.appendix_gradient_reporting import summarize_gradient_statistics


def objective_metrics(raw):
    result = summarize_gradient_statistics(
        [{"pair_id": "toy", "condition": "toy", "statistics": raw}], bootstrap_repeats=8,
    )
    return {
        (row["objective"], row["k"]): {name: metric["estimate"] for name, metric in row["metrics"].items()}
        for row in result["pairs"][0]["objectives"]
    }


class AppendixGradientReportTests(unittest.TestCase):
    def test_archived_dot_products_preserve_direction_and_sign(self):
        # Each block uses margins (-log(3), log(3)), hence weights (-3/4, -1/4).
        # Rows are the interleaved source block means (h, LossAvg, MVPO K=2), then audit h.
        vectors = np.array([[1, 0], [0.25, 0], [-0.5, 0],
                            [1, 0], [0.75, 0.5], [-0.5, 0], [1, 0], [1, 0]])
        raw = {
            "ks": [1, 2], "source_views_per_block": 2, "audit_views_per_block": 1,
            "parameter_count": 2, "block_mean_gram": vectors.dot(vectors.T).tolist(),
            "audit_block_margin_means": [0, 0],
            "source_blocks": [
                {"loss_derivative_mean": -0.5, "group_squared_norm_sums": {
                    "lossavg": {"1": single, "2": combined}, "mvpo": {"1": single, "2": 0.25},
                }} for single, combined in ((3.25, 0.0625), (13.75, 0.8125))
            ],
        }
        metrics = objective_metrics(raw)
        self.assertEqual(metrics["lossavg", 1], metrics["mvpo", 1])
        self.assertAlmostEqual(metrics["lossavg", 2]["descent_cosine"], -2 / math.sqrt(5))
        self.assertAlmostEqual(metrics["lossavg", 2]["signed_projection"], -0.5)
        self.assertAlmostEqual(metrics["mvpo", 2]["descent_cosine"], 1)
        self.assertAlmostEqual(metrics["mvpo", 2]["signed_projection"], 0.5)
        self.assertAlmostEqual(metrics["mvpo", 2]["distance"], 0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch is required")
class AppendixGradientCollectorTests(unittest.TestCase):
    def test_collected_geometry_matches_autograd_of_each_group_objective(self):
        import torch
        from torch.nn.functional import softplus

        parameter = torch.zeros(2, requires_grad=True)
        slopes = torch.tensor([[-2., 0.], [4., 0.], [-4., -2.], [6., 2.], [1., 1.], [1., -1.]])
        offsets = torch.tensor([-3., 3., -4., 4., 1., -0.5])
        visits = []

        def observation(seed):
            visits.append(seed)
            margin = slopes[seed].dot(parameter) + offsets[seed]
            gradient, = torch.autograd.grad(margin, parameter)
            return margin.item(), gradient

        raw = collect_gradient_statistics(
            observation, source_seeds=((0, 1), (2, 3)), audit_seeds=((4,), (5,)),
            ks=(1, 2), source_views_per_block=2, audit_views_per_block=1,
        )
        self.assertCountEqual(visits, range(6))
        metrics = objective_metrics(raw)
        self.assertEqual(metrics["lossavg", 1], metrics["mvpo", 1])

        reference, = torch.autograd.grad(softplus(-(slopes[4:].mv(parameter) + offsets[4:]).mean()), parameter)
        audit_direction = slopes[4:].mean(dim=0)
        for (method, k), measured in metrics.items():
            with self.subTest(method=method, k=k):
                gradients = []
                for start in range(0, 4, k):
                    group = slice(start, start + k)
                    margins = slopes[group].mv(parameter) + offsets[group]
                    loss = softplus(-margins).mean() if method == "lossavg" else softplus(-margins.mean())
                    gradient, = torch.autograd.grad(loss, parameter)
                    gradients.append(gradient)
                gradients = torch.stack(gradients)
                mean = gradients.mean(dim=0)
                self.assertAlmostEqual(measured["distance"], (mean - reference).norm().item(), places=6)
                self.assertAlmostEqual(measured["signed_projection"], -mean.dot(audit_direction).item(), places=6)
                cosine = torch.nn.functional.cosine_similarity(-mean, audit_direction, dim=0).item()
                self.assertAlmostEqual(measured["descent_cosine"], cosine, places=6)
                self.assertAlmostEqual(measured["stochastic_variance"], gradients.var(dim=0).sum().item(), places=5)
        self.assertLess(metrics["lossavg", 2]["signed_projection"], 0)
        self.assertGreater(metrics["mvpo", 2]["signed_projection"], 0)


if __name__ == "__main__":
    unittest.main()
