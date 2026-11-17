"""Snakemake wrapper: simulate one dataset."""
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src.simulate import simulate_counts  # noqa: E402

sm = snakemake  # noqa: F821  (injected by Snakemake)
sim = dict(sm.params.sim)

data = simulate_counts(n_genes=sim["n_genes"],
                       n_per_group=int(sm.wildcards.n),
                       frac_de=sim["frac_de"],
                       log2fc=float(sm.wildcards.lfc),
                       lfc_spread=sim["lfc_spread"],
                       library_size_sd=sim["library_size_sd"],
                       n_outlier_samples=sim["n_outlier_samples"],
                       seed=int(sm.wildcards.seed))

os.makedirs(os.path.dirname(sm.output.counts), exist_ok=True)
data.counts.to_csv(sm.output.counts)
data.metadata.to_csv(sm.output.metadata)
data.truth.to_csv(sm.output.truth)

with open(sm.log[0], "w") as fh:
    fh.write(f"simulated {data.n_samples} samples x {data.n_genes} genes, "
             f"{data.n_de} truly DE\n")
