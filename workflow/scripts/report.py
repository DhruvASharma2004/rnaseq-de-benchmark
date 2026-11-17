"""Snakemake wrapper: write the Markdown benchmark report."""
import pandas as pd

sm = snakemake  # noqa: F821

summary = pd.read_csv(sm.input.summary)
alpha = float(sm.params.alpha)
cfg = dict(sm.params.config)

by_method = (summary.groupby(["method", "n_per_group"])
             .agg(observed_fdr=("observed_fdr", "mean"),
                  power=("power", "mean"),
                  auroc=("auroc", "mean"),
                  n_called=("n_called", "mean"),
                  lfc_bias=("lfc_bias", "mean"))
             .round(4).reset_index())

breaches = summary[(summary["observed_fdr"] > alpha * 1.25)
                   & (summary["n_called"] > 0)]
silent = summary[summary["n_called"] == 0]

lines = [
    "# RNA-seq differential-expression benchmark",
    "",
    f"{len(summary)} scenario-method runs across "
    f"{summary['n_per_group'].nunique()} sample sizes, "
    f"{summary['log2fc'].nunique()} effect sizes and "
    f"{summary['seed'].nunique()} seeds, at nominal FDR {alpha:g}.",
    "",
    "## Mean performance by method and sample size",
    "",
    by_method.to_markdown(index=False),
    "",
    "## FDR control",
    "",
]

if len(breaches):
    lines += [f"{len(breaches)} run(s) exceeded the nominal FDR by more than "
              f"25%:", "",
              breaches[["method", "n_per_group", "log2fc", "seed",
                        "observed_fdr", "n_called"]].round(4)
              .to_markdown(index=False), ""]
else:
    lines += [f"No run exceeded the nominal FDR of {alpha:g} by more than "
              f"25%. Every method held its promise on these simulations.", ""]

if len(silent):
    lines += ["## Methods that made no calls", "",
              "An observed FDR of zero means nothing when a method calls "
              "nothing. These runs returned no genes at all:", "",
              silent[["method", "n_per_group", "log2fc", "seed",
                      "min_pvalue"]].round(6).to_markdown(index=False),
              "",
              "For a rank-based test the cause is structural: with small "
              "groups the smallest attainable p-value is bounded below by "
              "the number of possible rank orderings, and that bound does "
              "not survive multiple-testing correction.", ""]

lines += ["## Configuration", "", "```yaml",
          f"n_genes: {cfg['simulation']['n_genes']}",
          f"frac_de: {cfg['simulation']['frac_de']}",
          f"n_per_group: {cfg['scenarios']['n_per_group']}",
          f"log2fc: {cfg['scenarios']['log2fc']}",
          f"seed: {cfg['scenarios']['seed']}",
          f"methods: {cfg['methods']}",
          f"alpha: {cfg['alpha']}",
          "```", ""]

with open(sm.output.report, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
with open(sm.log[0], "w") as fh:
    fh.write(f"report written from {len(summary)} rows\n")
