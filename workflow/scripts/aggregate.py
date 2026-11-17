"""Snakemake wrapper: gather every metrics row into one table."""
import pandas as pd

sm = snakemake  # noqa: F821

frames = [pd.read_csv(path) for path in sm.input]
summary = pd.concat(frames, ignore_index=True)
summary = summary.sort_values(["method", "n_per_group", "log2fc", "seed"])
summary.to_csv(sm.output.summary, index=False)

with open(sm.log[0], "w") as fh:
    fh.write(f"aggregated {len(summary)} scenario-method rows\n")
