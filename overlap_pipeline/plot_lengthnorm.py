"""
Raw vs length-normalized weakness-only coverage (one extra headline figure).

The point estimate and 95% bootstrap CI for both the raw weakness-only metric
(`coverage_critique_only`) and the length-normalized metric
(`coverage_lengthnorm`) are read straight from the canonical
`outputs/scores/bootstrap_ci_by_source.csv` produced by score_overlap.py, so
this figure is consistent with Table 4.1 by construction. For each source it
draws two bars side by side -- raw (solid) and length-normalized (hatched) --
making the "the gap is verbosity, not per-concern skill" finding visible: the
tall AI raw bars collapse to the human level once length is controlled.

Run: python plot_lengthnorm.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
CI_CSV = _HERE / "outputs" / "scores" / "bootstrap_ci_by_source.csv"
PLOTS_ROOT = _HERE / "outputs" / "plots"

# Palette mirrors plot_results.py (one colour per source + grey baseline/edges).
COLOR_AI = "#3785bb"     # gpt blue
COLOR_MED = "#fe8112"    # llama orange
COLOR_GRN = "#43aa43"    # qwen green
COLOR_GREY = "#a0a0a0"   # human grey
EDGE = "#3A3A3A"
GRID = "#B5B5B5"

SOURCE_COLOR = {"gpt": COLOR_AI, "llama": COLOR_MED, "qwen": COLOR_GRN,
                "human_internal": COLOR_GREY}
SOURCE_ORDER = ["gpt", "llama", "qwen", "human_internal"]
SOURCE_LABEL = {"gpt": "GPT", "llama": "Llama", "qwen": "Qwen",
                "human_internal": "Human\n(internal)"}

RAW_METRIC = "coverage_critique_only"
NORM_METRIC = "coverage_lengthnorm"


def _apply_base_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE)
    ax.spines["bottom"].set_color(EDGE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def _row(df, src, metric):
    """Return (point, ci_lo, ci_hi) for one source/metric from the bootstrap CSV."""
    r = df[(df["group_source"] == src) & (df["metric"] == metric)]
    if r.empty:
        return np.nan, np.nan, np.nan
    r = r.iloc[0]
    return float(r["point"]), float(r["ci_lo"]), float(r["ci_hi"])


def main():
    df = pd.read_csv(CI_CSV)
    sources = [s for s in SOURCE_ORDER if s in set(df["group_source"])]

    x = np.arange(len(sources))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    _apply_base_style(ax)

    for j, (metric, hatch, offset) in enumerate(
            [(RAW_METRIC, None, -0.5 * width), (NORM_METRIC, "////", 0.5 * width)]):
        points, lo_err, hi_err = [], [], []
        for src in sources:
            p, lo, hi = _row(df, src, metric)
            points.append(p)
            lo_err.append(p - lo)
            hi_err.append(hi - p)
        colors = [SOURCE_COLOR.get(s, COLOR_GREY) for s in sources]
        bars = ax.bar(x + offset, points, width,
                      color=colors, edgecolor=EDGE, linewidth=0.7,
                      hatch=hatch, alpha=0.55 if hatch else 1.0, zorder=3)
        ax.errorbar(x + offset, points, yerr=[lo_err, hi_err], fmt="none",
                    ecolor=EDGE, elinewidth=0.9, capsize=3, zorder=4)
        for b, p, hi in zip(bars, points, hi_err):
            if not np.isnan(p):
                ax.annotate(f"{p:.2f}", (b.get_x() + b.get_width() / 2, p + hi),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8.5)

    # Legend distinguishes the two metrics (texture), not the sources (colour).
    legend_handles = [
        Patch(facecolor=COLOR_GREY, edgecolor=EDGE, label="Raw weakness-only"),
        Patch(facecolor=COLOR_GREY, edgecolor=EDGE, hatch="////", alpha=0.55,
              label="Length-normalized"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper right")

    ax.set_ylabel("Weakness-only coverage")
    ax.set_ylim(0, 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_LABEL.get(s, s) for s in sources])
    fig.tight_layout()

    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS_ROOT / f"fig_lengthnorm_comparison.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_lengthnorm_comparison.pdf / .png to {PLOTS_ROOT}")


if __name__ == "__main__":
    main()
