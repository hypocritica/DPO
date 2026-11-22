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


class AppendixGradientDistributionTests(unittest.TestCase):
    @staticmethod
    def raw_vectors():
        from self_correct_dpo.evaluation.appendix_gradients import _loss_derivative
        slopes = np.array([[-2., 0., 1.], [4., 0., 2.], [-4., -2., 0.], [6., 2., -1.],
                           [1., 2., 0.], [2., -1., 2.], [-3., 1., 1.], [4., 0., -1.],
                           [1., 1., 0.], [1., -1., 1.], [2., 0., -1.], [0., 1., 1.]])
        margins = np.array([-3., 3., -4., 4., 1., -0.5, -2., 2., 0.5, -0.5, 1., 0.])
        ks = [1, 2, 4]
        vectors, blocks = [], []
        for start in (0, 4):
            h, m = slopes[start:start + 4], margins[start:start + 4]
            a = _loss_derivative(m)
            means = [h.mean(0), (a[:, None] * h).mean(0)]
            norms = {"lossavg": {}, "mvpo": {}}
            for k in ks:
                la = (a[:, None] * h).reshape(-1, k, 3).mean(1)
                mv = _loss_derivative(m.reshape(-1, k).mean(1))[:, None] * h.reshape(-1, k, 3).mean(1)
                norms["lossavg"][str(k)] = np.square(la).sum()
                norms["mvpo"][str(k)] = np.square(mv).sum()
                if k > 1:
                    means.append(mv.mean(0))
            vectors.extend(means)
            blocks.append(dict(margin_mean=m.mean(), loss_derivative_mean=a.mean(), group_squared_norm_sums=norms))
        vectors.extend([slopes[8:10].mean(0), slopes[10:12].mean(0)])
        vectors = np.asarray(vectors)
        return dict(ks=ks, source_views_per_block=4, audit_views_per_block=2, parameter_count=3,
                    source_blocks=blocks, audit_block_margin_means=[margins[8:10].mean(), margins[10:12].mean()],
                    block_mean_gram=(vectors @ vectors.T).tolist()), slopes, margins

    def test_raw_mean_pairing_decoupling_and_audit_rows_match_explicit_vectors(self):
        from self_correct_dpo.evaluation.appendix_gradients import _loss_derivative
        from self_correct_dpo.evaluation.appendix_gradient_reporting import audit_reliability_rows, paired_change_rows
        raw, h, m = self.raw_vectors()
        summary = summarize_gradient_statistics([dict(pair_id="toy", condition="toy", statistics=raw)], bootstrap_repeats=12)
        pair = summary["pairs"][0]
        objectives = {(row["objective"], row["k"]): row["metrics"] for row in pair["objectives"]}
        for k in raw["ks"]:
            self.assertEqual(objectives["lossavg", k]["mean_gradient_norm"], objectives["lossavg", 1]["mean_gradient_norm"])
        self.assertEqual(objectives["lossavg", 1], objectives["mvpo", 1])
        weights = _loss_derivative(m[:8])
        coupled = (weights[:, None] * h[:8]).mean(0)
        decoupled = np.mean([weights[i] * h[j] for i in range(8) for j in range(8) if i != j], axis=0)
        self.assertAlmostEqual(pair["coupling"]["covariance_norm"]["estimate"], np.linalg.norm(coupled - decoupled))
        halves = [h[8:10].mean(0), h[10:12].mean(0)]
        expected_cosine = halves[0] @ halves[1] / (np.linalg.norm(halves[0]) * np.linalg.norm(halves[1]))
        self.assertAlmostEqual(pair["audit_split"]["direction_cosine"], expected_cosine)
        rows = list(paired_change_rows(summary))
        self.assertTrue(all(row["change"] == 0 for row in rows if row["k"] == 1))
        audits = list(audit_reliability_rows(summary))
        self.assertTrue(any(row["metric"] == "half_direction_cosine" for row in audits))
        contrast = next(row for row in audits if row["k"] == 4 and row["metric"] == "distance")
        for i, key in enumerate(("half1", "half2")):
            split = {(r["objective"], r["k"]): r["metrics"] for r in pair["audit_split"]["objectives"]}
            self.assertAlmostEqual(contrast[key], split["mvpo", 4]["distance"][i] - split["lossavg", 4]["distance"][i])

    def test_sparse_gram_contraction_equals_dense_including_zero_vectors(self):
        from self_correct_dpo.evaluation.appendix_gradient_reporting import _dot
        rng = np.random.default_rng(23)
        basis = rng.normal(size=(56, 7))
        gram = basis @ basis.T
        for active in (0, 8, 16, 56):
            left = np.zeros((32, 56))
            right = np.zeros_like(left)
            left[:, :active] = rng.normal(size=(32, active))
            right[:, -active:] = rng.normal(size=(32, active)) if active else 0
            np.testing.assert_allclose(_dot(left, gram, right), np.sum(left.dot(gram) * right, axis=1), atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
