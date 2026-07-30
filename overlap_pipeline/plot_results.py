"""
Overlap result figures (stage 4). Recomputes aggregate tables from Stage-2
match files via score_overlap and renders coverage plots (A-F) to outputs/plots/
as .pdf + .png (300 dpi). Run: python plot_results.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from score_overlap import (
    KIND_AI,
    OUT_ROOT as SCORES_ROOT,
    build_aggregates,
    compute_pair,
    item_level_rows,
    load_match_files,
    _load_status_buckets,
)


# critique-only: strengths are upward-biased, critiques are the defensible number
HEADLINE_METRIC = "coverage_critique_only"
HEADLINE_YLABEL = "Weakness-only coverage (recall of human weaknesses)"


def _human_internal_baseline(pair_df, metric=HEADLINE_METRIC):
    """Mean human-vs-human leave-one-out coverage on `metric` (the in-corpus reference)."""
    return pair_df.loc[pair_df["group_source"] == "human_internal", metric].mean()

_HERE = Path(__file__).resolve().parent
PLOTS_ROOT = _HERE / "outputs" / "plots"

# Palette (mirrors the survey figures; one colour per source + grey baseline).
COLOR_AI = "#3785bb"     # gpt blue
COLOR_MED = "#fe8112"    # llama orange
COLOR_GRN = "#43aa43"    # qwen green
COLOR_GREY = "#a0a0a0"   # human grey
EDGE = "#3A3A3A"
GRID = "#B5B5B5"

SOURCE_COLOR = {
    "gpt": COLOR_AI,
    "llama": COLOR_MED,
    "qwen": COLOR_GRN,
    "human_internal": COLOR_GREY,
}
SOURCE_ORDER = ["gpt", "llama", "qwen", "human_internal"]
SOURCE_LABEL = {"gpt": "GPT", "llama": "Llama", "qwen": "Qwen",
                "human_internal": "Human\n(internal)"}


def _apply_base_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE)
    ax.spines["bottom"].set_color(EDGE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, stem):
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS_ROOT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


def _ordered_sources(present):
    return [s for s in SOURCE_ORDER if s in present]


## figures

def fig_coverage_distribution(pair_df):
    sources = _ordered_sources(set(pair_df["group_source"]))
    if not sources:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    _apply_base_style(ax)
    for x, src in enumerate(sources):
        vals = pair_df.loc[pair_df["group_source"] == src, HEADLINE_METRIC].dropna()
        if vals.empty:
            continue
        ax.bar(x, vals.mean(), 0.6, yerr=vals.std() if len(vals) > 1 else None,
               color=SOURCE_COLOR.get(src, COLOR_GREY), edgecolor=EDGE,
               linewidth=0.7, zorder=3, capsize=4)
        jitter = (np.linspace(-0.18, 0.18, len(vals)) if len(vals) > 1 else [0])
        ax.scatter(np.full(len(vals), x) + jitter, vals, color=EDGE, s=14,
                   zorder=4, alpha=0.7)
    hbase = _human_internal_baseline(pair_df)
    if not np.isnan(hbase):
        ax.axhline(hbase, color="#B23A48", linestyle="--", linewidth=1.2,
                   zorder=2, label=f"Human internal {hbase:.2f}")
    ax.set_ylabel(HEADLINE_YLABEL)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([SOURCE_LABEL.get(s, s) for s in sources])
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig_coverage_distribution")


def _grouped_by_category(table, cat_col, stem, title_ylabel, rotate=0):
    """Grouped bars: category on x, one bar per source. Expects columns [cat_col, group_source, coverage]."""
    if table is None or table.empty:
        return
    cats = list(dict.fromkeys(table[cat_col].fillna("None")))
    sources = _ordered_sources(set(table["group_source"]))
    if not cats or not sources:
        return
    x = np.arange(len(cats))
    width = 0.8 / max(len(sources), 1)

    fig, ax = plt.subplots(figsize=(max(6.4, 1.1 * len(cats)), 4.2))
    _apply_base_style(ax)
    for i, src in enumerate(sources):
        sub = table[table["group_source"] == src]
        lookup = dict(zip(sub[cat_col].fillna("None"), sub["coverage"]))
        ys = [float(lookup.get(c, np.nan)) for c in cats]
        offset = (i - (len(sources) - 1) / 2) * width
        ax.bar(x + offset, ys, width, label=SOURCE_LABEL.get(src, src).replace("\n", " "),
               color=SOURCE_COLOR.get(src, COLOR_GREY), edgecolor=EDGE,
               linewidth=0.6, zorder=3)
    ax.set_ylabel(title_ylabel)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=rotate, ha="right" if rotate else "center")
    ax.legend(frameon=False, loc="upper right", ncol=len(sources))
    fig.tight_layout()
    _save(fig, stem)


def fig_source_comparison(pair_df):
    sources = _ordered_sources(set(pair_df["group_source"]))
    if not sources:
        return
    means = [pair_df.loc[pair_df["group_source"] == s, HEADLINE_METRIC].mean()
             for s in sources]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    _apply_base_style(ax)
    bars = ax.bar(range(len(sources)), means, 0.6,
                  color=[SOURCE_COLOR.get(s, COLOR_GREY) for s in sources],
                  edgecolor=EDGE, linewidth=0.7, zorder=3)
    for b, m in zip(bars, means):
        if not np.isnan(m):
            ax.annotate(f"{m:.2f}", (b.get_x() + b.get_width() / 2, m),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    hbase = _human_internal_baseline(pair_df)
    if not np.isnan(hbase):
        ax.axhline(hbase, color="#B23A48", linestyle="--", linewidth=1.3,
                   zorder=2, label=f"Human internal {hbase:.2f}")
        ax.legend(frameon=False, loc="upper right")
    ax.set_ylabel("Mean weakness-only coverage")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([SOURCE_LABEL.get(s, s) for s in sources])
    fig.tight_layout()
    _save(fig, "fig_source_comparison")


def fig_coverage_by_venue(pair_df):
    venues = [("ICLR26_rndm50", "ICLR", COLOR_AI),
              ("natureCom_nlpterms_31", "natureCom", COLOR_MED)]
    present = set(pair_df["group_source"])
    sources = _ordered_sources(present)
    if not sources or pair_df["dataset"].nunique() < 2:
        return  #nothing to split
    x = np.arange(len(sources))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    _apply_base_style(ax)
    for j, (ds, label, color) in enumerate(venues):
        means = [pair_df[(pair_df.group_source == s) & (pair_df.dataset == ds)]
                 ["coverage_baseline"].mean() for s in sources]
        offset = (j - 0.5) * width
        bars = ax.bar(x + offset, means, width, label=label, color=color,
                      edgecolor=EDGE, linewidth=0.7, zorder=3)
        for b, m in zip(bars, means):
            if not np.isnan(m):
                ax.annotate(f"{m:.2f}", (b.get_x() + b.get_width() / 2, m),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=8)
    ax.set_ylabel("Baseline coverage")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_LABEL.get(s, s) for s in sources])
    ax.legend(frameon=False, loc="upper right", title="Venue")
    fig.tight_layout()
    _save(fig, "fig_coverage_by_venue")


def fig_llm_vs_embedding():
    fp = SCORES_ROOT / "baseline_per_pair.csv"
    if not fp.exists():
        return
    df = pd.read_csv(fp)
    if "semantic_coverage" not in df.columns:
        return
    sub = df[["llm_coverage", "semantic_coverage", "source"]].dropna()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    _apply_base_style(ax)
    for src in _ordered_sources(set(sub["source"])):
        s = sub[sub["source"] == src]
        ax.scatter(s["llm_coverage"], s["semantic_coverage"],
                   color=SOURCE_COLOR.get(src, COLOR_GREY), edgecolor=EDGE,
                   s=36, zorder=3, label=SOURCE_LABEL.get(src, src).replace("\n", " "))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.0, zorder=2)
    ax.set_xlabel("LLM matcher coverage")
    ax.set_ylabel("Semantic matcher coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig_llm_vs_embedding")


def main():
    matches = load_match_files()
    if not matches:
        print("No match files; run run_full_matching.py then score_overlap.py first.")
        return
    status = _load_status_buckets()
    pair_df = pd.DataFrame([compute_pair(m, status) for m in matches])
    item_df = pd.DataFrame([r for m in matches for r in item_level_rows(m)])
    tables = build_aggregates(pair_df, item_df)

    print(f"rendering {len(matches)} match files -> {PLOTS_ROOT}")
    fig_coverage_distribution(pair_df)
    _grouped_by_category(tables.get("by_seriousness"), "seriousness",
                         "fig_coverage_seriousness",
                         "Coverage by weakness seriousness")
    _grouped_by_category(tables.get("by_type"), "type",
                         "fig_coverage_type", "Coverage by concern type", rotate=35)
    fig_source_comparison(pair_df)
    fig_coverage_by_venue(pair_df)
    fig_llm_vs_embedding()
    print("Done.")


if __name__ == "__main__":
    main()
