"""Differential-expression methods behind one interface.

Three methods span the range a working analyst actually chooses between:

``deseq2``
    Negative-binomial GLM with empirical-Bayes dispersion shrinkage
    (``pydeseq2``, a Python port of DESeq2). The reference method.
``ttest``
    Welch's t-test on log2 counts-per-million. The quick approach, and the
    one most likely to misbehave at small sample sizes.
``wilcoxon``
    Rank-sum test on CPM. Distribution-free, but it cannot produce a p-value
    below a floor set by the sample size, which matters at n = 3.

Each returns the same frame -- one row per gene with ``log2FoldChange``,
``pvalue`` and BH-adjusted ``padj`` -- so the evaluation layer never needs to
know which method produced it.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict

import numpy as np
import pandas as pd

__all__ = ["METHODS", "run_method", "cpm", "benjamini_hochberg"]


def cpm(counts: pd.DataFrame, log: bool = True, prior: float = 1.0
        ) -> pd.DataFrame:
    """Counts per million, optionally log2-transformed.

    Normalising by library size is what makes samples comparable; ``prior``
    is added before the log so that zero counts stay finite.
    """
    lib = counts.sum(axis=1).to_numpy()[:, None]
    lib = np.where(lib == 0, 1.0, lib)
    out = counts.to_numpy() / lib * 1e6
    if log:
        out = np.log2(out + prior)
    return pd.DataFrame(out, index=counts.index, columns=counts.columns)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values.

    Implemented directly rather than imported so the FDR procedure under test
    is visible in this repository: the benchmark's whole subject is whether
    nominal FDR matches observed FDR, and that claim is weaker if the
    adjustment itself is a black box.
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    out = np.full(n, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out

    pf = p[finite]
    m = pf.size
    order = np.argsort(pf)
    ranked = pf[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    # Enforce monotonicity from the largest p-value downward.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty(m)
    restored[order] = adjusted
    out[finite] = restored
    return out


def _deseq2(counts: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dds = DeseqDataSet(counts=counts, metadata=metadata,
                           design="~condition", quiet=True)
        dds.deseq2()
        stats = DeseqStats(dds, contrast=["condition", "B", "A"], quiet=True)
        stats.summary()
    res = stats.results_df.reindex(counts.columns)
    return pd.DataFrame({"log2FoldChange": res["log2FoldChange"].to_numpy(),
                         "pvalue": res["pvalue"].to_numpy(),
                         "padj": res["padj"].to_numpy()},
                        index=counts.columns)


def _ttest(counts: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats as sps

    logcpm = cpm(counts, log=True)
    a = logcpm.loc[metadata["condition"] == "A"].to_numpy()
    b = logcpm.loc[metadata["condition"] == "B"].to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        res = sps.ttest_ind(b, a, axis=0, equal_var=False)
    pvalue = np.asarray(res.pvalue, dtype=float)
    # A gene with zero variance in both groups yields a nan p-value; it is
    # uninformative rather than significant, so it is treated as p = 1.
    pvalue = np.where(np.isfinite(pvalue), pvalue, 1.0)
    lfc = b.mean(axis=0) - a.mean(axis=0)

    return pd.DataFrame({"log2FoldChange": lfc, "pvalue": pvalue,
                         "padj": benjamini_hochberg(pvalue)},
                        index=counts.columns)


def _wilcoxon(counts: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats as sps

    logcpm = cpm(counts, log=True)
    a = logcpm.loc[metadata["condition"] == "A"].to_numpy()
    b = logcpm.loc[metadata["condition"] == "B"].to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sps.mannwhitneyu(b, a, axis=0, alternative="two-sided")
    pvalue = np.asarray(res.pvalue, dtype=float)
    pvalue = np.where(np.isfinite(pvalue), pvalue, 1.0)
    lfc = np.median(b, axis=0) - np.median(a, axis=0)

    return pd.DataFrame({"log2FoldChange": lfc, "pvalue": pvalue,
                         "padj": benjamini_hochberg(pvalue)},
                        index=counts.columns)


#: Method name -> callable(counts, metadata) -> results frame.
METHODS: Dict[str, Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]] = {
    "deseq2": _deseq2,
    "ttest": _ttest,
    "wilcoxon": _wilcoxon,
}


def run_method(name: str, counts: pd.DataFrame,
               metadata: pd.DataFrame) -> pd.DataFrame:
    """Run one named method and return its standard results frame.

    Raises
    ------
    KeyError
        If ``name`` is not a registered method.
    """
    if name not in METHODS:
        raise KeyError(f"Unknown method '{name}'. "
                       f"Available: {', '.join(sorted(METHODS))}")
    out = METHODS[name](counts, metadata)
    # DESeq2 sets padj to NaN for genes it filters out for low counts. Those
    # genes were tested but not called; treating them as padj = 1 keeps every
    # method's output on the same footing for evaluation.
    out["padj"] = out["padj"].fillna(1.0)
    out["pvalue"] = out["pvalue"].fillna(1.0)
    return out
