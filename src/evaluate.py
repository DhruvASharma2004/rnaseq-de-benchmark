"""Benchmark metrics: does a differential-expression method do what it claims?

A method's p-values come with two implicit promises: that calling genes at
``padj < alpha`` holds the false discovery rate near ``alpha``, and that the
reported fold changes are close to the real ones. Both are checkable against
simulated truth, and both are checked here.

===========================  =================================================
``observed_fdr``             Share of calls that are actually null
``power``                    Share of truly-DE genes recovered
``null_p_rate``              Share of null genes with raw p < alpha (should
                             equal alpha if p-values are calibrated)
``auroc`` / ``average_precision``  Threshold-free ranking quality
``lfc_bias`` / ``lfc_rmse``  Accuracy of the effect-size estimates
``min_pvalue``               The smallest p-value the method could produce --
                             exposes the discreteness floor of rank tests
===========================  =================================================
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["evaluate_result", "roc_curve", "pr_curve"]


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the rank-sum identity, with ties averaged.

    Equivalent to the Mann-Whitney U statistic normalised to [0, 1]; computed
    directly so the benchmark does not depend on scikit-learn for a metric
    this small.
    """
    labels = labels.astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1)

    # Average ranks within tied groups.
    sorted_scores = scores[order]
    start = 0
    for i in range(1, scores.size + 1):
        if i == scores.size or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (step interpolation)."""
    labels = labels.astype(bool)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    hits = labels[order]
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, hits.size + 1)
    return float((precision * hits).sum() / labels.sum())


def roc_curve(scores: np.ndarray, labels: np.ndarray):
    """False-positive and true-positive rates across all thresholds."""
    labels = labels.astype(bool)
    order = np.argsort(-scores)
    hits = labels[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(~hits)
    n_pos, n_neg = max(1, int(labels.sum())), max(1, int((~labels).sum()))
    return fp / n_neg, tp / n_pos


def pr_curve(scores: np.ndarray, labels: np.ndarray):
    """Recall and precision across all thresholds."""
    labels = labels.astype(bool)
    order = np.argsort(-scores)
    hits = labels[order]
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, hits.size + 1)
    recall = tp / max(1, int(labels.sum()))
    return recall, precision


def evaluate_result(result: pd.DataFrame, truth: pd.DataFrame,
                    alpha: float = 0.05,
                    params: Optional[Dict] = None) -> Dict[str, float]:
    """Score one method's output against the simulation's ground truth.

    Parameters
    ----------
    result : pandas.DataFrame
        Output of :func:`src.methods.run_method`, indexed by gene.
    truth : pandas.DataFrame
        ``is_de`` and ``true_log2fc`` per gene.
    alpha : float
        Nominal FDR level the method was asked to control.
    params : dict, optional
        Simulation settings, copied into the output row for provenance.

    Returns
    -------
    dict
        One flat row of metrics, ready to concatenate across the benchmark.
    """
    aligned = result.reindex(truth.index)
    padj = aligned["padj"].to_numpy(dtype=float)
    pval = aligned["pvalue"].to_numpy(dtype=float)
    lfc = aligned["log2FoldChange"].to_numpy(dtype=float)
    is_de = truth["is_de"].to_numpy(dtype=bool)
    true_lfc = truth["true_log2fc"].to_numpy(dtype=float)

    called = np.nan_to_num(padj, nan=1.0) < alpha
    n_called = int(called.sum())
    true_pos = int((called & is_de).sum())
    false_pos = int((called & ~is_de).sum())

    # Observed FDR is undefined with no calls; reported as 0 by convention,
    # with n_called carried alongside so a vacuous 0 is never mistaken for
    # good behaviour.
    observed_fdr = (false_pos / n_called) if n_called else 0.0
    power = true_pos / max(1, int(is_de.sum()))

    null_p = pval[~is_de]
    null_p_rate = float(np.mean(np.nan_to_num(null_p, nan=1.0) < alpha)) \
        if null_p.size else float("nan")

    # Rank by significance: -log10(p) so that larger means more significant.
    with np.errstate(divide="ignore"):
        score = -np.log10(np.clip(np.nan_to_num(pval, nan=1.0), 1e-300, 1.0))

    de_mask = is_de & np.isfinite(lfc)
    lfc_bias = float(np.mean(lfc[de_mask] - true_lfc[de_mask])) \
        if de_mask.any() else float("nan")
    lfc_rmse = float(np.sqrt(np.mean((lfc[de_mask] - true_lfc[de_mask]) ** 2))) \
        if de_mask.any() else float("nan")

    row = {
        "n_genes": int(truth.shape[0]),
        "n_true_de": int(is_de.sum()),
        "alpha": float(alpha),
        "n_called": n_called,
        "true_positives": true_pos,
        "false_positives": false_pos,
        "observed_fdr": float(observed_fdr),
        "fdr_controlled": bool(observed_fdr <= alpha * 1.25 or n_called == 0),
        "power": float(power),
        "null_p_rate": null_p_rate,
        "auroc": _auroc(score, is_de),
        "average_precision": _average_precision(score, is_de),
        "lfc_bias": lfc_bias,
        "lfc_rmse": lfc_rmse,
        "min_pvalue": float(np.nanmin(pval)) if pval.size else float("nan"),
    }
    if params:
        row.update({k: v for k, v in params.items() if k not in row})
    return row
