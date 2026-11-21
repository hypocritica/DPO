"""Evaluate Appendix C.2.3's Gaussian flow at theta=2 by Gauss–Hermite quadrature."""
import argparse
import csv
from pathlib import Path

import numpy as np


def expected_derivative(k, order):
    # Mean of K iid standard Gaussian noises. hermgauss integrates exp(-x^2).
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    noise = nodes * np.sqrt(2.0 / k)
    m, h = 3 + 2 * noise, 2 + 2 * noise
    return float(weights @ (-h * np.exp(-np.logaddexp(0, m))) / np.sqrt(np.pi))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    # The reverse ODE from 1 to 0 maps X1 to exp(-theta)*X1. For winner 0
    # and loser 1, R=(exp(2*theta)-1)/2 and dR/dtheta=exp(2*theta).
    slope = float(np.exp(4))
    points = []
    g1 = expected_derivative(1, 160)
    for k in (1, 2, 4, 6, 8, np.inf):
        g = expected_derivative(k, 160)
        err = abs(g - expected_derivative(k, 96))
        points.append(dict(k=str(int(k)) if np.isfinite(k) else "infinity", mvpo_gradient=g,
                           lossavg_gradient=g1, mvpo_descent=-g, lossavg_descent=-g1,
                           mvpo_likelihood_ratio_slope=-slope*g, lossavg_likelihood_ratio_slope=-slope*g1,
                           abs_quadrature_difference=err))
    with (out / "exact-likelihood.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=points[0].keys())
        writer.writeheader()
        writer.writerows(points)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True)
    for ax, suffix, title in zip(axes, ("descent", "likelihood_ratio_slope"),
                                 ("Expected descent in θ", "Local change in R per unit step size")):
        x = np.arange(len(points))
        ax.axhline(0, color="0.5", linewidth=.7)
        ax.plot(x, [r["lossavg_" + suffix] for r in points], "--", label="LossAvg", color="#b25b28")
        ax.plot(x, [r["mvpo_" + suffix] for r in points], "o-", label="MVPO", color="#21615c")
        ax.set_xticks(x)
        ax.set_xticklabels(["1", "2", "4", "6", "8", "∞"])
        ax.set_xlabel("Number of views K")
        ax.set_title(title, fontsize=10)
        ax.legend()
    fig.suptitle("Scalar Gaussian flow at θ = 2 · deterministic quadrature")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"exact-likelihood.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
