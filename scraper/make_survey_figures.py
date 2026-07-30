"""
Generates survey-results figures for the thesis (AI-assisted peer review policies).

Counts below come from chapters/results.tex - update here and re-run to regenerate.
Output goes to bachelor_thesis/figures/ (pdf + png each).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# data (aggregated survey counts; keep in sync with chapters/results.tex)

POLICY_CATEGORIES = ["Prohibited", "Partial", "Active", "None"]
AI_NLP_POLICY = [13, 32, 1, 17]   # n = 63 AI/NLP conferences
MEDICAL_POLICY = [28, 18, 1, 1]   # n = 48 medical journals

# sub-tags within the "partially allowed" bucket (non-exclusive)
SUBTAGS = [
    "polish_ok",
    "readability_ok",
    "no_manuscript_upload",
    "no_review_upload",
    "disclosure_required",
]
AI_NLP_SUBTAG = [12, 14, 8, 2, 1]   # AI/NLP partial, n = 32
MEDICAL_SUBTAG = [1, 0, 15, 0, 3]   # Medical partial, n = 18

#style - paper colour scheme (shared with result figures)
COLOR_AI = "#3785bb"      # gpt blue (AI/NLP series)
COLOR_MED = "#fe8112"     # llama orange (Medical series)
COLOR_GRN = "#43aa43"     # qwen green
COLOR_GREY = "#a0a0a0"    # human grey
EDGE = "#3A3A3A"
GRID = "#B5B5B5"


def _figures_dir():
    """Walk up from this script to find bachelor_thesis/figures/ - works from any cwd."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "bachelor_thesis" / "figures"
        if cand.is_dir():
            return cand
    return here.parent


OUT_DIR = _figures_dir()


def _apply_base_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE)
    ax.spines["bottom"].set_color(EDGE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def _label_bars(ax, bars, fontsize=9):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{int(h)}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def _grouped_bar(categories, series_a, series_b, label_a, label_b,
                 ylabel, ymax, out_stem, figsize, bar_width=0.38,
                 xtick_rotation=0, xtick_ha="center"):
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=figsize)
    _apply_base_style(ax)

    bars_a = ax.bar(x - bar_width / 2, series_a, bar_width,
                    label=label_a, color=COLOR_AI,
                    edgecolor=EDGE, linewidth=0.7, zorder=3)
    bars_b = ax.bar(x + bar_width / 2, series_b, bar_width,
                    label=label_b, color=COLOR_MED,
                    edgecolor=EDGE, linewidth=0.7, zorder=3)

    _label_bars(ax, bars_a)
    _label_bars(ax, bars_b)

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=xtick_rotation, ha=xtick_ha)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_stem}.pdf / .png")


def make_policy_distribution():
    _grouped_bar(
        categories=POLICY_CATEGORIES,
        series_a=AI_NLP_POLICY,
        series_b=MEDICAL_POLICY,
        label_a="AI/NLP (n = 63)",
        label_b="Medical (n = 48)",
        ylabel="Number of venues",
        ymax=36,
        out_stem="fig_policy_distribution",
        figsize=(7.2, 4.3),
    )


def make_subtag_distribution():
    _grouped_bar(
        categories=SUBTAGS,
        series_a=AI_NLP_SUBTAG,
        series_b=MEDICAL_SUBTAG,
        label_a="AI/NLP partial (n = 32)",
        label_b="Medical partial (n = 18)",
        ylabel="Venues carrying tag",
        ymax=18,
        out_stem="fig_subtag_distribution",
        figsize=(8.4, 4.8),
        bar_width=0.40,
        xtick_rotation=18,
        xtick_ha="right",
    )


if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "sans-serif",
    })
    make_policy_distribution()
    make_subtag_distribution()
