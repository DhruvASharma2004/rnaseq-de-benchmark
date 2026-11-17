"""Snakemake wrapper: score one result against ground truth."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src.evaluate import evaluate_result, pr_curve, roc_curve  # noqa: E402

sm = snakemake  # noqa: F821

result = pd.read_csv(sm.input.result, index_col=0)
truth = pd.read_csv(sm.input.truth, index_col=0)

params = {"n_per_group": int(sm.wildcards.n),
          "log2fc": float(sm.wildcards.lfc),
          "seed": int(sm.wildcards.seed),
          "method": sm.wildcards.method}

row = evaluate_result(result, truth, alpha=float(sm.params.alpha),
                      params=params)

os.makedirs(os.path.dirname(sm.output.metrics), exist_ok=True)
pd.DataFrame([row]).to_csv(sm.output.metrics, index=False)

# Threshold-free curves, thinned to keep the gathered files small.
aligned = result.reindex(truth.index)
pval = np.nan_to_num(aligned["pvalue"].to_numpy(dtype=float), nan=1.0)
with np.errstate(divide="ignore"):
    score = -np.log10(np.clip(pval, 1e-300, 1.0))
labels = truth["is_de"].to_numpy(dtype=bool)

fpr, tpr = roc_curve(score, labels)
recall, precision = pr_curve(score, labels)
step = max(1, len(fpr) // 400)
pd.DataFrame({"fpr": fpr[::step], "tpr": tpr[::step],
              "recall": recall[::step], "precision": precision[::step],
              **{k: v for k, v in params.items()}}
             ).to_csv(sm.output.curves, index=False)

with open(sm.log[0], "w") as fh:
    fh.write(f"{params['method']}: called {row['n_called']}, "
             f"observed FDR {row['observed_fdr']:.4f}, "
             f"power {row['power']:.4f}\n")
