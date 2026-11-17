"""Snakemake wrapper: run one DE method on one dataset."""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src.methods import run_method  # noqa: E402

sm = snakemake  # noqa: F821

counts = pd.read_csv(sm.input.counts, index_col=0)
metadata = pd.read_csv(sm.input.metadata, index_col=0)

start = time.time()
result = run_method(sm.wildcards.method, counts, metadata)
elapsed = time.time() - start

os.makedirs(os.path.dirname(sm.output.result), exist_ok=True)
result.to_csv(sm.output.result)

with open(sm.log[0], "w") as fh:
    fh.write(f"{sm.wildcards.method}: {counts.shape[1]} genes, "
             f"{counts.shape[0]} samples, {elapsed:.2f}s\n")
