"""Benchmark figures.

Four plots, each answering one question a reader of a DE analysis should ask:

* ``plot_fdr_control`` -- does "FDR < 5%" mean what it says?
* ``plot_power`` -- how much signal does each method recover, and at what n?
* ``plot_curves`` -- how good is the ranking, independent of any threshold?
* ``plot_effect_accuracy`` -- are the reported fold changes the real ones?
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["plot_fdr_control", "plot_power", "plot_curves",
           "plot_effect_accuracy", "METHOD_COLORS", "METHOD_LABELS"]

DPI = 300

METHOD_COLORS = {"deseq2": "#1f6fb4", "ttest": "#e08214",
                 "wilcoxon": "#2e8b57"}
METHOD_LABELS = {"deseq2": "DESeq2", "ttest": "Welch t-test",
                 "wilcoxon": "Wilcoxon"}


def _style(ax):
    ax.grid(alpha=0.18, lw=0.5)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_fdr_control(summary: pd.DataFrame, alpha: float = 0.05,
                     output_path: Optional[str] = None, dpi: int = DPI):
    """Observed false discovery rate against the nominal level.

    Points above the dashed line are methods calling more false positives
    than they promised. Scenarios where a method made no calls at all are
    marked, since an observed FDR of zero there means silence, not accuracy.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=dpi)
    lfcs = sorted(summary["log2fc"].unique())

    for ax, lfc in zip(axes, lfcs[:2]):
        sub = summary[summary["log2fc"] == lfc]
        for method, grp in sub.groupby("method"):
            agg = grp.groupby("n_per_group")["observed_fdr"].agg(["mean", "std"])
            silent = grp.groupby("n_per_group")["n_called"].mean() == 0
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"].fillna(0),
                        marker="o", ms=5, lw=1.5, capsize=3,
                        color=METHOD_COLORS.get(method, "#666666"),
                        label=METHOD_LABELS.get(method, method))
            if silent.any():
                ax.scatter(agg.index[silent.to_numpy()],
                           agg["mean"][silent.to_numpy()], s=120,
                           facecolors="none", edgecolors="#c0392b", lw=1.4,
                           zorder=5)

        ax.axhline(alpha, color="#c0392b", ls="--", lw=1.2,
                   label=f"nominal {alpha:g}")
        ax.set_title(f"Effect size: log2FC = {lfc:g}", fontsize=10)
        ax.set_xlabel("Samples per group", fontsize=9)
        ax.set_ylabel("Observed FDR", fontsize=9)
        ax.set_xticks(sorted(sub["n_per_group"].unique()))
        ax.set_ylim(bottom=-0.005)
        _style(ax)

    axes[0].legend(fontsize=7.6, frameon=False)
    axes[1].text(0.98, 0.04, "○ = method made no calls",
                 transform=axes[1].transAxes, ha="right", fontsize=7,
                 color="#c0392b")
    fig.suptitle("Does \"FDR < 5%\" hold? Observed vs nominal false discovery "
                 "rate", fontsize=11.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        return output_path
    return fig


def plot_power(summary: pd.DataFrame, output_path: Optional[str] = None,
               dpi: int = DPI):
    """Statistical power by sample size, split by effect size."""
    lfcs = sorted(summary["log2fc"].unique())
    fig, axes = plt.subplots(1, len(lfcs), figsize=(5.6 * len(lfcs), 4.4),
                             dpi=dpi, squeeze=False)

    for ax, lfc in zip(axes[0], lfcs):
        sub = summary[summary["log2fc"] == lfc]
        for method, grp in sub.groupby("method"):
            agg = grp.groupby("n_per_group")["power"].agg(["mean", "std"])
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"].fillna(0),
                        marker="o", ms=5, lw=1.5, capsize=3,
                        color=METHOD_COLORS.get(method, "#666666"),
                        label=METHOD_LABELS.get(method, method))
        ax.set_title(f"log2FC = {lfc:g}", fontsize=10)
        ax.set_xlabel("Samples per group", fontsize=9)
        ax.set_ylabel("Power (true DE genes recovered)", fontsize=9)
        ax.set_xticks(sorted(sub["n_per_group"].unique()))
        ax.set_ylim(-0.02, 1.02)
        _style(ax)
    axes[0][0].legend(fontsize=7.6, frameon=False)

    fig.suptitle("Power by sample size", fontsize=11.5, fontweight="bold",
                 y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        return output_path
    return fig


def plot_curves(curves: pd.DataFrame, summary: pd.DataFrame,
                n_per_group: Optional[int] = None,
                log2fc: Optional[float] = None,
                seed: Optional[int] = None,
                output_path: Optional[str] = None, dpi: int = DPI):
    """ROC and precision-recall curves for one representative scenario.

    Threshold-free, so a method that ranks genes well but calibrates its
    p-values badly still looks good here -- which is exactly the distinction
    worth drawing next to the FDR panel.
    """
    if n_per_group is None:
        n_per_group = int(sorted(curves["n_per_group"].unique())[
            len(curves["n_per_group"].unique()) // 2])
    if log2fc is None:
        log2fc = float(sorted(curves["log2fc"].unique())[0])
    if seed is None:
        seed = int(sorted(curves["seed"].unique())[0])

    sub = curves[(curves["n_per_group"] == n_per_group)
                 & (np.isclose(curves["log2fc"], log2fc))
                 & (curves["seed"] == seed)]
    stats = summary[(summary["n_per_group"] == n_per_group)
                    & (np.isclose(summary["log2fc"], log2fc))
                    & (summary["seed"] == seed)].set_index("method")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=dpi)
    for method, grp in sub.groupby("method"):
        color = METHOD_COLORS.get(method, "#666666")
        label = METHOD_LABELS.get(method, method)
        auroc = stats["auroc"].get(method, np.nan)
        ap = stats["average_precision"].get(method, np.nan)
        axes[0].plot(grp["fpr"], grp["tpr"], lw=1.6, color=color,
                     label=f"{label} (AUC {auroc:.3f})")
        axes[1].plot(grp["recall"], grp["precision"], lw=1.6, color=color,
                     label=f"{label} (AP {ap:.3f})")

    axes[0].plot([0, 1], [0, 1], ls="--", lw=1.0, color="#999999")
    axes[0].set_xlabel("False positive rate", fontsize=9)
    axes[0].set_ylabel("True positive rate", fontsize=9)
    axes[0].set_title("ROC", fontsize=10)

    axes[1].set_xlabel("Recall", fontsize=9)
    axes[1].set_ylabel("Precision", fontsize=9)
    axes[1].set_title("Precision-recall", fontsize=10)

    for ax in axes:
        ax.legend(fontsize=7.6, frameon=False, loc="lower left")
        _style(ax)

    fig.suptitle(f"Ranking quality: n = {n_per_group} per group, "
                 f"log2FC = {log2fc:g}, seed {seed}",
                 fontsize=11.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        return output_path
    return fig


def plot_effect_accuracy(summary: pd.DataFrame,
                         output_path: Optional[str] = None, dpi: int = DPI):
    """Bias and error of the estimated log2 fold changes."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=dpi)
    methods = sorted(summary["method"].unique())
    width = 0.8 / max(1, len(methods))
    ns = sorted(summary["n_per_group"].unique())
    x = np.arange(len(ns))

    for i, method in enumerate(methods):
        sub = summary[summary["method"] == method]
        bias = sub.groupby("n_per_group")["lfc_bias"].mean().reindex(ns)
        rmse = sub.groupby("n_per_group")["lfc_rmse"].mean().reindex(ns)
        offset = (i - (len(methods) - 1) / 2) * width
        color = METHOD_COLORS.get(method, "#666666")
        axes[0].bar(x + offset, bias.to_numpy(), width, color=color,
                    label=METHOD_LABELS.get(method, method))
        axes[1].bar(x + offset, rmse.to_numpy(), width, color=color)

    axes[0].axhline(0.0, color="#333333", lw=0.9)
    axes[0].set_ylabel("Mean log2FC bias (estimate − truth)", fontsize=9)
    axes[0].set_title("Bias: shrinkage pulls estimates toward zero",
                      fontsize=10)
    axes[1].set_ylabel("log2FC RMSE", fontsize=9)
    axes[1].set_title("Total error", fontsize=10)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("Samples per group", fontsize=9)
        _style(ax)
    axes[0].legend(fontsize=7.6, frameon=False)

    fig.suptitle("Effect-size accuracy on truly differential genes",
                 fontsize=11.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        return output_path
    return fig
