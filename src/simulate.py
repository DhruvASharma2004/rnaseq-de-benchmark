"""Negative-binomial RNA-seq count simulator with known differential expression.

Measuring whether a differential-expression workflow controls its false
discovery rate requires knowing which genes are truly null, which no real
dataset tells you. This module generates counts whose truth is recorded by
construction, with the features that make real RNA-seq hard:

* **Negative-binomial counts** with a decreasing mean-dispersion trend, the
  empirical relationship reported for real libraries (Love et al. 2014).
* **Library-size variation** between samples, so normalisation matters.
* **A log-normal expression spectrum**, so most genes are lowly expressed and
  a few dominate the library.
* **Optional outlier samples**, to test robustness.

The ground truth is returned alongside the counts: which genes are
differentially expressed, and the exact log2 fold change used for each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["SimulatedDataset", "simulate_counts"]


@dataclass
class SimulatedDataset:
    """Counts plus the ground truth that generated them.

    Attributes
    ----------
    counts : pandas.DataFrame
        Samples (rows) by genes (columns), raw integer counts.
    metadata : pandas.DataFrame
        One row per sample with a ``condition`` column ("A" or "B").
    truth : pandas.DataFrame
        One row per gene: ``is_de`` (bool) and ``true_log2fc`` (0 for nulls).
    params : dict
        The settings used, carried through to the results for provenance.
    """

    counts: pd.DataFrame
    metadata: pd.DataFrame
    truth: pd.DataFrame
    params: dict

    @property
    def n_genes(self) -> int:
        return int(self.counts.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_de(self) -> int:
        return int(self.truth["is_de"].sum())


def simulate_counts(n_genes: int = 4000,
                    n_per_group: int = 6,
                    frac_de: float = 0.10,
                    log2fc: float = 1.0,
                    lfc_spread: float = 0.25,
                    dispersion_asymptote: float = 0.05,
                    dispersion_scale: float = 6.0,
                    library_size_sd: float = 0.25,
                    n_outlier_samples: int = 0,
                    min_mean_count: float = 1.0,
                    seed: int = 0) -> SimulatedDataset:
    """Simulate a two-group RNA-seq experiment with known truth.

    Parameters
    ----------
    n_genes : int
        Number of genes.
    n_per_group : int
        Samples per condition; the total is twice this.
    frac_de : float
        Fraction of genes made differentially expressed.
    log2fc : float
        Mean absolute log2 fold change applied to DE genes. Half are raised
        and half lowered, so the library is not globally shifted -- a global
        shift would be absorbed by size-factor normalisation and would make
        the benchmark easier than reality.
    lfc_spread : float
        Standard deviation of the per-gene fold change around ``log2fc``.
    dispersion_asymptote, dispersion_scale : float
        Mean-dispersion trend, ``disp(mu) = asymptote + scale / (mu + 10)``.
        Lowly expressed genes are noisier, as in real data.
    library_size_sd : float
        Log-normal standard deviation of per-sample sequencing depth.
    n_outlier_samples : int
        Samples given an inflated dispersion, simulating a bad library.
    min_mean_count : float
        Genes below this expected count are resampled upward, mimicking the
        low-count filtering applied before a real analysis.
    seed : int
        Seed; the same seed always produces the same dataset.

    Returns
    -------
    SimulatedDataset
    """
    if not 0.0 <= frac_de < 1.0:
        raise ValueError("frac_de must be in [0, 1)")
    if n_per_group < 2:
        raise ValueError("n_per_group must be at least 2 for a comparison")
    if n_genes < 2:
        raise ValueError("n_genes must be at least 2")

    rng = np.random.default_rng(seed)
    n_samples = 2 * n_per_group

    # Baseline expression: log-normal spectrum, floored so that every gene is
    # at least nominally detectable.
    base_mean = rng.lognormal(mean=4.0, sigma=1.6, size=n_genes)
    base_mean = np.maximum(base_mean, min_mean_count)

    # Ground truth: which genes move, and by how much.
    n_de = int(round(frac_de * n_genes))
    de_idx = rng.choice(n_genes, size=n_de, replace=False)
    true_lfc = np.zeros(n_genes)
    if n_de:
        magnitude = rng.normal(log2fc, lfc_spread, size=n_de)
        magnitude = np.maximum(np.abs(magnitude), 0.05)
        sign = rng.choice([-1.0, 1.0], size=n_de)       # balanced up/down
        true_lfc[de_idx] = sign * magnitude

    # Per-sample means: group B shifted by the true fold change.
    condition = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    group_factor = np.where(condition == "B", 1.0, 0.0)[:, None]
    mu = base_mean[None, :] * np.power(2.0, group_factor * true_lfc[None, :])

    # Library size: a multiplicative per-sample factor.
    size_factor = rng.lognormal(0.0, library_size_sd, size=n_samples)
    mu = mu * size_factor[:, None]

    # Mean-dispersion trend, inflated for designated outlier samples.
    dispersion = dispersion_asymptote + dispersion_scale / (base_mean + 10.0)
    disp_matrix = np.tile(dispersion, (n_samples, 1))
    if n_outlier_samples > 0:
        outliers = rng.choice(n_samples, size=min(n_outlier_samples,
                                                  n_samples), replace=False)
        disp_matrix[outliers, :] *= 4.0

    # Negative binomial in (n, p) form: n = 1/disp, p = 1/(1 + mu*disp).
    nb_n = 1.0 / disp_matrix
    nb_p = 1.0 / (1.0 + mu * disp_matrix)
    counts = rng.negative_binomial(nb_n, nb_p)

    gene_names = [f"gene_{i:05d}" for i in range(n_genes)]
    sample_names = [f"{c}_{i:02d}" for i, c in enumerate(condition)]

    counts_df = pd.DataFrame(counts, index=sample_names, columns=gene_names)
    metadata = pd.DataFrame({"condition": condition}, index=sample_names)
    truth = pd.DataFrame({"gene": gene_names,
                          "is_de": true_lfc != 0.0,
                          "true_log2fc": true_lfc}).set_index("gene")

    params = {"n_genes": n_genes, "n_per_group": n_per_group,
              "frac_de": frac_de, "log2fc": log2fc, "lfc_spread": lfc_spread,
              "library_size_sd": library_size_sd,
              "n_outlier_samples": n_outlier_samples, "seed": seed}

    return SimulatedDataset(counts=counts_df, metadata=metadata, truth=truth,
                            params=params)
