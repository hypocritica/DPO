"""Check the tagged-view K=4 sampler against an enumerable two-stage model."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from appendix_stage_report import finite_k4, gate


class TaggedViewGateTests(unittest.TestCase):
    def test_conditional_global_draws_and_shared_view(self):
        banks = [[-2, -2], [4, 4]]
        result = finite_k4(banks, 0, 100000, np.random.default_rng(23))
        expected = sum(math.comb(3, n) * gate(-2 + 6 * n / 4) / 8 for n in range(4))
        self.assertAlmostEqual(result["k4_stage_mean"], gate(-2))
        self.assertAlmostEqual(result["k4_global_mean"], expected, delta=0.004)
        self.assertAlmostEqual(result["k4_delta_mean"], gate(-2) - expected, delta=0.004)

    def test_equal_empirical_banks_cancel_for_every_draw(self):
        result = finite_k4([[-3, 2], [-3, 2]], 1, 128, np.random.default_rng(42))
        for key in ("mean", "q05", "q50", "q95", "mc_se"):
            self.assertEqual(result[f"k4_delta_{key}"], 0)
