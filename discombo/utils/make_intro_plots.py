"""Generate a motivational figure focused on IHPO + MC-HPO only.

This script creates one conceptual panel that shows:
1) IHPO conditional sampling via p(x | y*), with y* the best observed value so far.
2) MC-HPO targeted acceptance via annealed p-value binning.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.legend_handler import HandlerBase, HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import FuncFormatter


def objective(x: np.ndarray) -> np.ndarray:
    """Synthetic 1D objective (lower is better)."""
    x = np.asarray(x)
    return 0.85 + 0.45 * np.sin(1.1 * x + 0.35) + 0.25 * np.cos(2.4 * x - 0.5)


def gaussian_pdf_1d(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    z = (x - mean) / std
    return np.exp(-0.5 * z**2) / (std * np.sqrt(2.0 * np.pi))


def std_normal_cdf(z: np.ndarray) -> np.ndarray:
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / np.sqrt(2.0)))


@dataclass
class HardCodedPC:
    """Small hand-crafted density used as a stand-in for a membership circuit.

    The joint is modeled as a mixture of product distributions:
    p(x, y) = sum_k w_k * p_k(x) * p_k(y).
    """

    mu_x: np.ndarray
    std_x: np.ndarray
    mu_y: np.ndarray
    std_y: np.ndarray
    weights: np.ndarray

    def joint_density(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pxy = np.zeros_like(x, dtype=float)
        for mx, sx, my, sy, w in zip(self.mu_x, self.std_x, self.mu_y, self.std_y, self.weights):
            pxy += w * gaussian_pdf_1d(x, mx, sx) * gaussian_pdf_1d(y, my, sy)
        return pxy

    def support_x(self, x: np.ndarray) -> np.ndarray:
        sx = np.zeros_like(x, dtype=float)
        for m, s, w in zip(self.mu_x, self.std_x, self.weights):
            sx += w * gaussian_pdf_1d(x, m, s)
        return sx

    def conditional_x_given_y_value(self, y_value: float, x: np.ndarray) -> np.ndarray:
        # p(x | y*) is proportional to p(x, y*).
        joint_at_y = self.joint_density(x, np.full_like(x, y_value))
        z = np.trapz(joint_at_y, x)
        if z <= 1e-12:
            return np.full_like(x, 1.0 / x.shape[0])
        return joint_at_y / z

    def prob_y_greater_than(self, y_value: float, x: np.ndarray) -> np.ndarray:
        """Compute p(y > y_value | x) for a Gaussian-mixture-style hard-coded PC."""
        support = np.maximum(self.support_x(x), 1e-12)

        py_gt = np.zeros_like(x, dtype=float)
        for mx, sx, my, sy, w in zip(self.mu_x, self.std_x, self.mu_y, self.std_y, self.weights):
            pxk = w * gaussian_pdf_1d(x, mx, sx)
            z = (y_value - my) / sy
            tail = 1.0 - std_normal_cdf(z)
            py_gt += pxk * tail

        return py_gt / support


def _sample_from_pdf_grid(
    x_grid: np.ndarray, pdf: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    probs = np.clip(pdf, 0.0, None)
    probs = probs / probs.sum()
    idx = rng.choice(np.arange(x_grid.shape[0]), size=n, replace=True, p=probs)
    return x_grid[idx]


def _annealing_beta(iteration: int, cycle_length: int, beta_start: float, beta_end: float) -> float:
    return (beta_start + beta_end) / 2.0 + np.cos(np.pi * iteration / cycle_length) * (
        beta_start - beta_end
    ) / 2.0


def _annealed_bin_edges(k: int, beta: float) -> np.ndarray:
    b = np.arange(k + 1, dtype=float)
    edges = (b / k) ** beta
    edges[0] = 0.0
    edges[-1] = 1.0
    return edges


def _cdf_from_pdf_grid(x_grid: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    """Build a numerical CDF on the same x-grid from a normalized PDF."""
    dx = np.diff(x_grid)
    cdf = np.zeros_like(pdf, dtype=float)
    cdf[1:] = np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * dx)
    total = cdf[-1]
    if total <= 1e-12:
        return np.linspace(0.0, 1.0, x_grid.shape[0])
    return np.clip(cdf / total, 0.0, 1.0)


def _select_one_per_bin(
    x_samples: np.ndarray,
    p_values: np.ndarray,
    edges: np.ndarray,
    rng: np.random.Generator,
    tail_fraction: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    selected = []
    occupied = np.zeros(edges.shape[0] - 1, dtype=bool)

    for b in range(edges.shape[0] - 1):
        lo = edges[b]
        hi = edges[b + 1]
        center = 0.5 * (lo + hi)
        is_tail_bin = (center <= tail_fraction) or (center >= 1.0 - tail_fraction)
        if not is_tail_bin:
            continue

        if b == edges.shape[0] - 2:
            in_bin = (p_values >= lo) & (p_values <= hi)
        else:
            in_bin = (p_values >= lo) & (p_values < hi)

        idx = np.flatnonzero(in_bin)
        if idx.size == 0:
            continue

        occupied[b] = True
        pick = rng.choice(idx)
        selected.append(x_samples[pick])

    if len(selected) == 0:
        return np.array([]), occupied
    return np.asarray(selected), occupied


def _select_n_from_bucket_priority(
    x_samples: np.ndarray,
    p_values: np.ndarray,
    edges: np.ndarray,
    bucket_priority: np.ndarray,
    n_select: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select up to n_select samples by visiting buckets in priority order."""
    bucket_ids = np.digitize(np.clip(p_values, 0.0, 1.0), edges[1:-1], right=False)
    chosen_idx: list[int] = []
    used = np.zeros(x_samples.shape[0], dtype=bool)

    for b in bucket_priority:
        in_bucket = np.flatnonzero((bucket_ids == b) & (~used))
        if in_bucket.size == 0:
            continue
        pick = int(rng.choice(in_bucket))
        chosen_idx.append(pick)
        used[pick] = True
        if len(chosen_idx) >= n_select:
            break

    if len(chosen_idx) < n_select:
        remaining = np.flatnonzero(~used)
        if remaining.size > 0:
            extra = rng.choice(remaining, size=min(n_select - len(chosen_idx), remaining.size), replace=False)
            chosen_idx.extend(extra.tolist())

    if len(chosen_idx) == 0:
        return np.array([])
    return x_samples[np.asarray(chosen_idx, dtype=int)]


@dataclass
class GaussianLegendHandle:
    color: str
    alpha: float = 0.18


class HandlerGaussian(HandlerBase):
    """Draw a tiny filled Gaussian icon inside legend entries."""

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        t = np.linspace(-2.5, 2.5, 32)
        x_curve = xdescent + width * (t - t.min()) / (t.max() - t.min())
        y_base = ydescent + 0.15 * height
        y_curve = y_base + 0.68 * height * np.exp(-0.5 * t**2)

        poly_x = np.concatenate(([x_curve[0]], x_curve, [x_curve[-1]]))
        poly_y = np.concatenate(([y_base], y_curve, [y_base]))
        fill = Polygon(
            np.column_stack([poly_x, poly_y]),
            closed=True,
            facecolor=orig_handle.color,
            edgecolor="none",
            alpha=orig_handle.alpha,
            transform=trans,
        )
        line = Line2D(x_curve, y_curve, color=orig_handle.color, lw=1.6, transform=trans)
        return [fill, line]


def make_figure(save_path: str, dpi: int = 220, seed: int = 0) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )

    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, 8.5, 1000)
    y_gt = objective(x)
    x_plot_offset = 3.0
    x_plot = x - x_plot_offset

    # Observed data used to define y* (best-so-far).
    x_obs = np.array([5.8, 6.5, 6.9, 7.4])
    y_obs = objective(x_obs)
    y_star = float(np.min(y_obs))

    # Hard-coded PC: mixture components centered near observed data points.
    mu_x = x_obs.copy()
    mu_y = y_obs.copy()
    pc = HardCodedPC(
        mu_x=mu_x,
        std_x=np.array([0.28, 0.30, 0.28, 0.32]),
        mu_y=mu_y,
        std_y=np.array([0.09, 0.08, 0.05, 0.05]),
        weights=np.array([0.25, 0.25, 0.25, 0.25]),
    )

    p_cond = pc.conditional_x_given_y_value(y_value=y_star, x=x)
    cdf_cond = _cdf_from_pdf_grid(x, p_cond)

    # MC-HPO targeted procedure: two cycle settings (exploit vs explore).
    n_mc = 25
    k_bins = 25
    iteration = 5
    beta_start, beta_end = 5.0, 1.0

    # High exploration setting (long cycle): pick lowest buckets.
    x_mc_explore = _sample_from_pdf_grid(x, p_cond, n=n_mc, rng=rng)
    p_explore = np.clip(np.interp(x_mc_explore, x, cdf_cond), 0.0, 1.0)
    beta_explore = _annealing_beta(iteration, cycle_length=50, beta_start=beta_start, beta_end=beta_end)
    edges_explore = _annealed_bin_edges(k=k_bins, beta=beta_explore)
    lowest_priority = np.arange(k_bins)
    x_explore = _select_n_from_bucket_priority(
        x_samples=x_mc_explore,
        p_values=p_explore,
        edges=edges_explore,
        bucket_priority=lowest_priority,
        n_select=3,
        rng=rng,
    )

    # High exploitation setting (short cycle): pick middle buckets.
    x_mc_exploit = _sample_from_pdf_grid(x, p_cond, n=n_mc, rng=rng)
    p_exploit = np.clip(np.interp(x_mc_exploit, x, cdf_cond), 0.0, 1.0)
    beta_exploit = _annealing_beta(iteration, cycle_length=5, beta_start=beta_start, beta_end=beta_end)
    edges_exploit = _annealed_bin_edges(k=k_bins, beta=beta_exploit)
    mid = 0.5 * (k_bins - 1)
    middle_priority = np.argsort(np.abs(np.arange(k_bins) - mid))
    x_exploit = _select_n_from_bucket_priority(
        x_samples=x_mc_exploit,
        p_values=p_exploit,
        edges=edges_exploit,
        bucket_priority=middle_priority,
        n_select=3,
        rng=rng,
    )

    x_ihpo = _sample_from_pdf_grid(x, p_cond, n=4, rng=rng)
    y_ihpo = objective(x_ihpo)

    y_explore = objective(x_explore)
    y_exploit = objective(x_exploit)

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8))
    contour_color = "#6b5d99"
    observed_color = "#29353a"
    mc_explore_color = "#4ea72e"
    mc_exploit_color = "#ff9b60"
    ihpo_color = "#6b5d99"
    best_line_color = "#8e002f"

    y_cond = np.full_like(x, y_star)
    p_joint = pc.joint_density(x, y_cond)
    contour_norm = Normalize(vmin=np.min(p_joint), vmax=np.max(p_joint))

    ax.plot(x_plot, y_gt, color="black", lw=2.2, label="Objective f(x)", alpha=0.6)
    ax.axhline(y_star, color=best_line_color, ls="--", lw=1.2, label="y* = f(x*) (best so far)")

    xx, yy = np.meshgrid(np.linspace(0.0, 8.5, 300), np.linspace(0.3, 1.6, 240))
    zz = pc.joint_density(xx, yy)
    xx_plot = xx - x_plot_offset
    ax.contour(
        xx_plot,
        yy,
        zz,
        levels=6,
        colors=contour_color,
        norm=contour_norm,
        linewidths=1.2,
        alpha=0.6,
    )

    p_cond_scaled = 0.0 + 0.35 * (p_cond / p_cond.max())
    ax.plot(x_plot, p_cond_scaled, color=ihpo_color, lw=1.8, label="_nolegend_", alpha=0.6)

    # Show conditional support for PC sampling within ±3 standard deviations.
    cond_mean = np.trapz(x * p_cond, x)
    cond_var = np.trapz(((x - cond_mean) ** 2) * p_cond, x)
    cond_std = float(np.sqrt(max(cond_var, 1e-12)))
    cond_mask = np.abs(x - cond_mean) <= 3.0 * cond_std
    ax.fill_between(
        x_plot,
        0.0,
        p_cond_scaled,
        where=cond_mask,
        interpolate=True,
        color=contour_color,
        alpha=0.14,
        zorder=4,
    )

    # Highlight conditional-density regions covered by exploration/exploitation samples.
    coverage_half_width_explore = 0.15
    coverage_half_width_exploit = 0.08
    explore_cov_mask = np.zeros_like(x, dtype=bool)
    exploit_cov_mask = np.zeros_like(x, dtype=bool)
    for xm in x_explore:
        explore_cov_mask |= np.abs(x - xm) <= coverage_half_width_explore
    for xm in x_exploit:
        exploit_cov_mask |= np.abs(x - xm) <= coverage_half_width_exploit

    ax.fill_between(
        x_plot,
        0.0,
        p_cond_scaled,
        where=explore_cov_mask,
        interpolate=True,
        color=mc_explore_color,
        alpha=0.30,
        zorder=5,
    )
    ax.fill_between(
        x_plot,
        0.0,
        p_cond_scaled,
        where=exploit_cov_mask,
        interpolate=True,
        color=mc_exploit_color,
        alpha=0.30,
        zorder=5,
    )

    ax.scatter(
        x_obs - x_plot_offset,
        y_obs,
        s=30,
        color=observed_color,
        alpha=0.9,
        label="Observed data",
        zorder=6,
    )
    ax.scatter(
        x_ihpo - x_plot_offset,
        y_ihpo,
        s=38,
        color=ihpo_color,
        edgecolor="white",
        lw=0.5,
        zorder=7,
        label="Cond. samples",
    )
    if x_explore.size > 0:
        ax.scatter(
            x_explore - x_plot_offset,
            y_explore,
            s=180,
            color=mc_explore_color,
            edgecolor="white",
            lw=0.5,
            zorder=7,
            marker=(5, 1),
            label="_nolegend_",
        )
    else:
        ax.scatter(
            [],
            [],
            s=180,
            color=mc_explore_color,
            edgecolor="white",
            lw=0.5,
            marker=(5, 1),
            label="_nolegend_",
        )

    if x_exploit.size > 0:
        ax.scatter(
            x_exploit - x_plot_offset,
            y_exploit,
            s=180,
            color=mc_exploit_color,
            edgecolor="white",
            lw=0.5,
            zorder=7,
            marker=(5, 1),
            label="_nolegend_",
        )
    else:
        ax.scatter(
            [],
            [],
            s=180,
            color=mc_exploit_color,
            edgecolor="white",
            lw=0.5,
            marker=(5, 1),
            label="_nolegend_",
        )

    ax.set_xlim(0.0, 8.5 - x_plot_offset)
    ax.set_ylim(0.0, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_aspect(1.5)

    # Remove all ticks and numeric labels while keeping axis labels.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0)

    ax.grid(alpha=0.15, linewidth=0.7)

    mc_star_explore = Line2D(
        [0],
        [0],
        marker=(5, 1),
        linestyle="None",
        markerfacecolor=mc_explore_color,
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=12,
    )
    mc_star_exploit = Line2D(
        [0],
        [0],
        marker=(5, 1),
        linestyle="None",
        markerfacecolor=mc_exploit_color,
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=12,
    )
    gaussian_handle = GaussianLegendHandle(color=ihpo_color, alpha=0.24)
    legend_handles = [
        Line2D([0], [0], color="black", lw=2.2, alpha=0.6),
        Line2D([0], [0], color=best_line_color, lw=1.2, linestyle="--"),
        gaussian_handle,
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=observed_color,
            markeredgecolor=observed_color,
            markersize=5,
            alpha=0.9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=ihpo_color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=6,
        ),
        (mc_star_explore, mc_star_exploit),
    ]
    legend_labels = [
        "Objective f(x)",
        "y* = f(x*)",
        "Cond. p(x | y*)",
        "Obs. data",
        "Cond. samples",
        "MC samples (explore / exploit)",
    ]
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.3, 1.0, 0.20),
        mode="expand",
        ncol=3,
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.5,
        columnspacing=1.0,
        handler_map={GaussianLegendHandle: HandlerGaussian(), tuple: HandlerTuple(ndivide=None, pad=0.25)},
        fontsize=13,
    )

    fig.tight_layout(rect=(0.0, 0.18, 1.0, 1.0))
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_2d_ribbon_figure(save_path: str, dpi: int = 220, seed: int = 0) -> None:
    """Create a 2D motivational contour plot with horizontal ribbons."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )

    rng = np.random.default_rng(seed)

    x1 = np.linspace(0.0, 1.0, 260)
    x2 = np.linspace(0.0, 1.0, 260)
    xx, yy = np.meshgrid(x1, x2)

    # Multimodal landscape with two optimum basins (motivational sketch).
    basin_a = ((xx - 0.33) / 0.20) ** 2 + ((yy - 0.41) / 0.17) ** 2
    basin_b = ((xx - 0.71) / 0.22) ** 2 + ((yy - 0.63) / 0.20) ** 2
    gamma = 3.0
    base = -np.log(np.exp(-gamma * basin_a) + np.exp(-gamma * basin_b)) / gamma
    rough = 0.08 * np.sin(17.0 * xx + 6.0 * yy) * np.cos(15.0 * yy - 5.0 * xx)
    rough += 0.05 * np.cos(11.0 * xx - 3.0 * yy)
    micro = np.zeros_like(xx)
    for cx, cy, amp, sx, sy in [
        (0.34, 0.43, -0.12, 0.028, 0.030),
        (0.68, 0.60, -0.11, 0.030, 0.028),
        (0.56, 0.52, 0.06, 0.030, 0.026),
        (0.30, 0.78, 0.08, 0.030, 0.026),
        (0.74, 0.21, 0.07, 0.035, 0.030),
    ]:
        micro += amp * np.exp(-(((xx - cx) ** 2) / (2.0 * sx**2) + ((yy - cy) ** 2) / (2.0 * sy**2)))

    # Small smooth perturbation to avoid perfectly symmetric contours.
    jitter = rng.normal(0.0, 0.02, size=xx.shape)
    z = base + rough + micro #+ jitter

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8))

    # Fill the optimum regions first.
    z_min = float(np.min(z))
    z_opt = float(np.percentile(z, 13.0))
    ax.contourf(xx, yy, z, levels=[z_min, z_opt], colors=["#6b5d99"], alpha=0.3, zorder=0)

    # Draw contour lines only outside optimum regions (no lines inside filled optima).
    levels = np.linspace(z_opt, np.percentile(z, 97.0), 11)
    ax.contour(xx, yy, z, levels=levels, cmap="Greys", linewidths=1.6, alpha=0.3)

    # Priors over the y-dimension (used to define ribbon support).
    y_axis = np.linspace(0.0, 1.0, 400)
    near_mu, near_sigma = 0.435, 0.035
    off_mu, off_sigma = 0.865, 0.035
    support_k = 2.5
    near_lo = max(0.0, near_mu - support_k * near_sigma)
    near_hi = min(1.0, near_mu + support_k * near_sigma)
    off_lo = max(0.0, off_mu - support_k * off_sigma)
    off_hi = min(1.0, off_mu + support_k * off_sigma)

    # Horizontal ribbons: dashed outlines only (no fill), spanning full x-range.
    x_left, x_right = 0.0, 1.0
    ribbon_good = "black"
    ribbon_bad = "#8e002f"
    for yv in (near_lo, near_hi):
        ax.hlines(yv, x_left, x_right, colors=ribbon_good, linestyles="--", linewidth=3.0, zorder=3, alpha=0.4)
    for yv in (off_lo, off_hi):
        ax.hlines(yv, x_left, x_right, colors=ribbon_bad, linestyles="--", linewidth=3.0, zorder=3, alpha=0.4)

    # Draw Gaussian priors attached to the y-axis.

    near_pdf = np.exp(-0.5 * ((y_axis - near_mu) / near_sigma) ** 2)
    off_pdf = np.exp(-0.5 * ((y_axis - off_mu) / off_sigma) ** 2)
    prior_width = 0.10
    near_x = prior_width * near_pdf / np.max(near_pdf)
    off_x = prior_width * off_pdf / np.max(off_pdf)

    ax.plot(near_x, y_axis, color=ribbon_good, lw=2.2, alpha=0.95, zorder=4)
    ax.plot(off_x, y_axis, color=ribbon_bad, lw=2.2, alpha=0.95, zorder=4)
    ax.fill_betweenx(y_axis, 0.0, near_x, color=ribbon_good, alpha=0.10, zorder=2)
    ax.fill_betweenx(y_axis, 0.0, off_x, color=ribbon_bad, alpha=0.10, zorder=2)

    obs_data = np.array([[0.24, 0.48], [0.30, 0.30], [0.50, 0.32], [0.60, 0.78], [0.72, 0.91]])
    ax.scatter(
        obs_data[:, 0],
        obs_data[:, 1],
        s=30,
        color="#29353a",
        alpha=0.9,
        label="Observed data",
        zorder=5,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r'$\mathregular{x_1}$')
    ax.set_ylabel(r'$\mathregular{x_2}$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor="#6b5d99", alpha=0.3, edgecolor="none", label="Good regions"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#29353a", markersize=8, label="Observed data"),
        Line2D([0], [0], color=ribbon_good, linestyle="--", linewidth=3.0, label="Beneficial hint region", alpha=0.4),
        Line2D([0], [0], color=ribbon_bad, linestyle="--", linewidth=3.0, label="Misleading hint region", alpha=0.4),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.30, 1.0, 0.20),
        mode="expand",
        ncol=3,
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.5,
        columnspacing=1.0,
        fontsize=13,
    )

    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create IHPO + MC-HPO motivational figure.")
    parser.add_argument(
        "--plot-kind",
        type=str,
        choices=["mchpo", "ribbon2d"],
        default="mchpo",
        help="Choose which figure to generate.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="figure1_mchpo_motivation.pdf",
        help="Output path for the figure.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Save DPI.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.plot_kind == "ribbon2d":
        make_2d_ribbon_figure(save_path=args.save_path, dpi=args.dpi, seed=args.seed)
    else:
        make_figure(save_path=args.save_path, dpi=args.dpi, seed=args.seed)
    print(f"Saved motivational figure to: {args.save_path}")
