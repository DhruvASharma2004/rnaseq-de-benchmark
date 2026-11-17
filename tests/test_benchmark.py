"""Unit tests for the benchmark's simulator, methods, and metrics.

These check behaviour that matters statistically: that the simulator's truth
is internally consistent, that the Benjamini-Hochberg implementation matches a
reference, that the metrics agree with scikit-learn, and that a method with an
obvious signal recovers it.

Run from the repository root::

    pytest -q
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.evaluate import (_auroc, _average_precision,  # noqa: E402
                          evaluate_result, pr_curve, roc_curve)
from src.methods import (METHODS, benjamini_hochberg, cpm,  # noqa: E402
                         run_method)
from src.simulate import simulate_counts  # noqa: E402


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def test_simulate_shapes_and_truth_are_consistent():
    d = simulate_counts(n_genes=500, n_per_group=4, frac_de=0.2, seed=0)
    assert d.counts.shape == (8, 500)
    assert d.metadata.shape[0] == 8
    assert d.truth.shape[0] == 500
    assert d.n_de == 100
    # Truth and fold changes must agree exactly.
    assert bool(((d.truth["true_log2fc"] != 0) == d.truth["is_de"]).all())
    assert set(d.metadata["condition"]) == {"A", "B"}


def test_simulate_is_reproducible_and_seed_sensitive():
    a = simulate_counts(n_genes=300, n_per_group=3, seed=7)
    b = simulate_counts(n_genes=300, n_per_group=3, seed=7)
    c = simulate_counts(n_genes=300, n_per_group=3, seed=8)
    assert np.array_equal(a.counts.to_numpy(), b.counts.to_numpy())
    assert not np.array_equal(a.counts.to_numpy(), c.counts.to_numpy())


def test_simulate_counts_are_non_negative_integers():
    d = simulate_counts(n_genes=400, n_per_group=3, seed=2)
    vals = d.counts.to_numpy()
    assert vals.min() >= 0
    assert np.issubdtype(vals.dtype, np.integer)


def test_simulate_fold_changes_are_balanced_up_and_down():
    """A one-sided shift would be absorbed by size-factor normalisation."""
    d = simulate_counts(n_genes=4000, n_per_group=4, frac_de=0.3, seed=11)
    up = int((d.truth["true_log2fc"] > 0).sum())
    down = int((d.truth["true_log2fc"] < 0).sum())
    assert abs(up - down) < 0.2 * (up + down)


def test_simulate_applies_the_requested_effect():
    """Group B must shift by the true fold change, net of composition.

    The comparison is made relative to the median shift of the unchanged
    genes, because total-count normalisation induces a global offset (see
    :func:`test_balanced_fold_changes_still_inflate_the_library`).
    """
    d = simulate_counts(n_genes=1500, n_per_group=20, frac_de=0.2,
                        log2fc=3.0, lfc_spread=0.05, library_size_sd=0.01,
                        seed=5)
    norm = cpm(d.counts, log=True)
    a = norm.loc[d.metadata["condition"] == "A"].mean()
    b = norm.loc[d.metadata["condition"] == "B"].mean()
    observed = (b - a).to_numpy()
    truth = d.truth["true_log2fc"].to_numpy()

    baseline = float(np.median(observed[truth == 0]))
    assert observed[truth > 0].mean() - baseline > 1.5
    assert observed[truth < 0].mean() - baseline < -1.5


def test_balanced_fold_changes_still_inflate_the_library():
    """Equal numbers of up- and down-regulated genes do not cancel.

    Fold changes are multiplicative: doubling a gene adds its own mass again,
    while halving it removes only half. So a balanced design still raises
    total counts, and total-count (CPM) normalisation then makes unchanged
    genes look mildly down-regulated. This composition effect is why
    count-based methods estimate size factors from a median of ratios instead,
    and it is visible in the benchmark as a negative fold-change bias for the
    CPM-based methods but not for DESeq2.
    """
    d = simulate_counts(n_genes=2000, n_per_group=15, frac_de=0.2,
                        log2fc=3.0, lfc_spread=0.05, library_size_sd=0.0,
                        seed=9)
    lib = d.counts.sum(axis=1)
    lib_a = lib[d.metadata["condition"] == "A"].mean()
    lib_b = lib[d.metadata["condition"] == "B"].mean()
    assert lib_b > lib_a                       # the library grew

    norm = cpm(d.counts, log=True)
    a = norm.loc[d.metadata["condition"] == "A"].mean().to_numpy()
    b = norm.loc[d.metadata["condition"] == "B"].mean().to_numpy()
    null_shift = np.median((b - a)[d.truth["true_log2fc"].to_numpy() == 0])
    assert null_shift < 0                      # unchanged genes look lower


def test_simulate_rejects_bad_arguments():
    with pytest.raises(ValueError):
        simulate_counts(frac_de=1.5)
    with pytest.raises(ValueError):
        simulate_counts(n_per_group=1)
    with pytest.raises(ValueError):
        simulate_counts(n_genes=1)


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------

def test_benjamini_hochberg_matches_statsmodels():
    statsmodels = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(0)
    p = np.clip(rng.beta(0.3, 3.0, 800), 1e-12, 1.0)
    ref = statsmodels.multipletests(p, method="fdr_bh")[1]
    assert np.abs(benjamini_hochberg(p) - ref).max() < 1e-12


def test_benjamini_hochberg_is_monotone_and_bounded():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=300)
    adj = benjamini_hochberg(p)
    assert np.all(adj >= p - 1e-12)          # adjustment never decreases a p
    assert adj.min() >= 0.0 and adj.max() <= 1.0
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_benjamini_hochberg_handles_nan():
    p = np.array([0.01, np.nan, 0.5])
    adj = benjamini_hochberg(p)
    assert np.isnan(adj[1])
    assert np.isfinite(adj[0]) and np.isfinite(adj[2])


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def test_cpm_normalises_library_size():
    counts = pd.DataFrame([[10, 10], [100, 100]], columns=["g1", "g2"],
                          index=["s1", "s2"])
    out = cpm(counts, log=False)
    assert out.loc["s1", "g1"] == pytest.approx(500_000)
    assert out.loc["s2", "g1"] == pytest.approx(500_000)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_returns_the_standard_frame(method):
    d = simulate_counts(n_genes=300, n_per_group=4, seed=3)
    res = run_method(method, d.counts, d.metadata)
    assert list(res.columns) == ["log2FoldChange", "pvalue", "padj"]
    assert len(res) == 300
    assert res.index.equals(d.counts.columns)
    assert res["pvalue"].between(0, 1).all()
    assert res["padj"].between(0, 1).all()
    assert res.notna().all().all()


@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_recovers_an_obvious_signal(method):
    """With a large effect and plenty of samples, ranking must be good."""
    d = simulate_counts(n_genes=600, n_per_group=10, frac_de=0.1,
                        log2fc=3.0, seed=4)
    res = run_method(method, d.counts, d.metadata)
    score = -np.log10(np.clip(res["pvalue"].to_numpy(), 1e-300, 1.0))
    assert _auroc(score, d.truth["is_de"].to_numpy(bool)) > 0.85


def test_run_method_rejects_unknown_name():
    d = simulate_counts(n_genes=100, n_per_group=3, seed=0)
    with pytest.raises(KeyError):
        run_method("edger", d.counts, d.metadata)


def test_wilcoxon_pvalue_floor_at_small_n():
    """A rank test cannot produce a small p-value with few samples.

    This is the structural reason Wilcoxon makes no discoveries at n = 3,
    whatever the effect size, and it is a property of the test rather than of
    this implementation.
    """
    d = simulate_counts(n_genes=400, n_per_group=3, frac_de=0.3, log2fc=5.0,
                        seed=6)
    res = run_method("wilcoxon", d.counts, d.metadata)
    assert res["pvalue"].min() > 0.01
    assert int((res["padj"] < 0.05).sum()) == 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_auroc_and_average_precision_match_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(3):
        y = rng.integers(0, 2, 400).astype(bool)
        s = rng.normal(size=400) + y * 1.2
        assert _auroc(s, y) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-12)
        assert _average_precision(s, y) == pytest.approx(
            sk.average_precision_score(y, s), abs=1e-12)


def test_auroc_handles_ties_and_extremes():
    assert _auroc(np.array([1.0, 1.0, 1.0, 0.0]),
                  np.array([1, 1, 0, 0], dtype=bool)) == pytest.approx(0.75)
    perfect = _auroc(np.array([3.0, 2.0, 1.0, 0.0]),
                     np.array([1, 1, 0, 0], dtype=bool))
    assert perfect == pytest.approx(1.0)
    assert np.isnan(_auroc(np.array([1.0, 2.0]), np.array([0, 0], dtype=bool)))


def test_evaluate_result_computes_fdr_and_power_correctly():
    """Hand-built case: 4 calls, of which 1 is null -> FDR 0.25."""
    genes = [f"g{i}" for i in range(10)]
    truth = pd.DataFrame({"is_de": [True] * 5 + [False] * 5,
                          "true_log2fc": [1.0] * 5 + [0.0] * 5}, index=genes)
    padj = [0.001, 0.002, 0.003, 0.9, 0.9, 0.004, 0.9, 0.9, 0.9, 0.9]
    result = pd.DataFrame({"log2FoldChange": [1.0] * 10,
                           "pvalue": padj, "padj": padj}, index=genes)
    row = evaluate_result(result, truth, alpha=0.05)
    assert row["n_called"] == 4
    assert row["true_positives"] == 3
    assert row["false_positives"] == 1
    assert row["observed_fdr"] == pytest.approx(0.25)
    assert row["power"] == pytest.approx(0.6)


def test_evaluate_result_reports_zero_calls_without_claiming_accuracy():
    genes = [f"g{i}" for i in range(6)]
    truth = pd.DataFrame({"is_de": [True] * 3 + [False] * 3,
                          "true_log2fc": [1.0] * 3 + [0.0] * 3}, index=genes)
    result = pd.DataFrame({"log2FoldChange": [0.0] * 6, "pvalue": [1.0] * 6,
                           "padj": [1.0] * 6}, index=genes)
    row = evaluate_result(result, truth)
    assert row["n_called"] == 0
    assert row["observed_fdr"] == 0.0
    assert row["power"] == 0.0
    assert row["fdr_controlled"] is True       # vacuously, and n_called says so


def test_curves_are_monotone_and_bounded():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200).astype(bool)
    s = rng.normal(size=200) + y
    fpr, tpr = roc_curve(s, y)
    recall, precision = pr_curve(s, y)
    assert np.all(np.diff(fpr) >= -1e-12) and np.all(np.diff(tpr) >= -1e-12)
    assert fpr.max() <= 1.0 and tpr.max() <= 1.0
    assert np.all(np.diff(recall) >= -1e-12)
    assert precision.min() >= 0.0 and precision.max() <= 1.0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def test_simulate_run_evaluate_end_to_end():
    d = simulate_counts(n_genes=400, n_per_group=6, frac_de=0.15,
                        log2fc=2.5, seed=8)
    res = run_method("ttest", d.counts, d.metadata)
    row = evaluate_result(res, d.truth, alpha=0.05,
                          params={"method": "ttest"})
    assert row["method"] == "ttest"
    assert 0.0 <= row["observed_fdr"] <= 1.0
    assert 0.0 <= row["power"] <= 1.0
    assert row["auroc"] > 0.7


def test_snakemake_dag_resolves():
    """The workflow must parse and plan without executing anything."""
    proc = subprocess.run(
        [sys.executable, "-m", "snakemake", "-s", "workflow/Snakefile",
         "--configfile", "config/ci.yaml", "-n", "-q"],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
