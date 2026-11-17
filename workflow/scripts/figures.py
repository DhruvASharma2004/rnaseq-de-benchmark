"""Snakemake wrapper: render the benchmark figures."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src.plots import (plot_curves, plot_effect_accuracy,  # noqa: E402
                       plot_fdr_control, plot_power)

sm = snakemake  # noqa: F821

summary = pd.read_csv(sm.input.summary)
curves = pd.concat([pd.read_csv(p) for p in sm.input.curves],
                   ignore_index=True)

os.makedirs(os.path.dirname(sm.output.fdr), exist_ok=True)
plot_fdr_control(summary, alpha=float(sm.params.alpha),
                 output_path=sm.output.fdr)
plot_power(summary, output_path=sm.output.power)
plot_curves(curves, summary, output_path=sm.output.curves_fig)
plot_effect_accuracy(summary, output_path=sm.output.effect)

with open(sm.log[0], "w") as fh:
    fh.write("rendered 4 figures\n")
