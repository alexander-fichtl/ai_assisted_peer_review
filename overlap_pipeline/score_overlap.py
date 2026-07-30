"""
Overlap scoring (stage 3 of 3). Pure compute, no API.

Reads Stage-2 match files under outputs/matches/ and computes overlap metrics,
breakdown tables, and CSVs. Human leave-one-out files provide an internal baseline.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from match_concerns import KIND_AI, KIND_H2H

_HERE = Path(__file__).resolve().parent
DATA_ROOT = _HERE / "data"
MATCHES_ROOT = _HERE / "outputs" / "matches"
OUT_ROOT = _HERE / "outputs" / "scores"

# Channel filters.
C_BASELINE = {"critique", "strength"}
C_CRITIQUE = {"critique"}
C_WITH_NEUTRAL = {"critique", "strength", "neutral"}

# Seriousness weights (edit freely; constants live here on purpose).
W_ESSENTIAL = 2.0
W_RECOMMENDED = 1.0
W_MINOR = 0.7
W_STRENGTH = 1.0
W_NEUTRAL = 0.5
W_DEFAULT_CRITIQUE = 1.0  # critique with no seriousness_category

# Exploratory "quantity" bonus per additional AI finding.
QUANTITY_BONUS = 0.1

# Granularity control (length normalization): shrink the AI review to ONE human
# reviewer's worth of concerns and recompute coverage, to test whether the AI's
# coverage lead is just an artifact of writing more concerns. 

# Paper-level (cluster) bootstrap for confidence intervals on the headline tables.
# The unit of resampling is the paper: the human-internal baseline has several
# leave-one-out comparisons per paper, so we first average those to one value per
# paper (per-paper aggregation), then resample papers with replacement. This makes
# the point estimate and its CI use the same clustering unit. Each AI source has
# exactly one review per paper, so per-paper == per-comparison for them.
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 7
BOOTSTRAP_SOURCES = ["gpt", "llama", "qwen", "human_internal"]
BOOTSTRAP_METRICS = [
    "coverage_critique_only", "coverage_baseline", "coverage_lengthnorm",
    "precision", "additional_rate",
]
BOOTSTRAP_DATASETS = ["ICLR26_rndm50", "natureCom_nlpterms_31"]


## Helpers

def _weight(item):
    ch = item.get("channel")
    if ch == "strength":
        return W_STRENGTH
    if ch == "neutral":
        return W_NEUTRAL
    if ch == "critique":
        return {
            "Essential": W_ESSENTIAL,
            "Recommended": W_RECOMMENDED,
            "Minor": W_MINOR,
        }.get(item.get("seriousness_category"), W_DEFAULT_CRITIQUE)
    return 0.0


def _coverage(items, matched_nums, channels):
    """Fraction of `items` in `channels` whose number is in `matched_nums`."""
    pool = [it for it in items if it.get("channel") in channels]
    if not pool:
        return None
    hit = sum(1 for it in pool if it.get("number") in matched_nums)
    return hit / len(pool)


def _weighted_coverage(items, matched_nums):
    den = sum(_weight(it) for it in items)
    if den == 0:
        return None
    num = sum(_weight(it) for it in items if it.get("number") in matched_nums)
    return num / den


def _count_in(items, channels):
    return sum(1 for it in items if it.get("channel") in channels)


def _lengthnorm_coverage(match, channels=C_CRITIQUE):
    """Coverage when the A-list (AI review) is shrunk to one reviewer's worth of items.

    Computed on the weakness (critique) channel by default, so it counts the same
    channel as the raw weakness-only metric and differs from it only in review length.

    K = round(|pooled human items| / number of distinct human reviewers) ~= the
    item count of a single typical reviewer of this paper. We report the coverage
    *averaged over all* size-K subsets of the AI review's concerns: the mean across
    every way of keeping K of them. This average has a closed form. A human concern
    with d matchers among the N AI concerns is covered by a fraction
    1 - C(N-d, K)/C(N, K) of the size-K subsets, and the metric averages that over the
    human concerns. It is exact and deterministic. If the A-list has <= K items (e.g.
    the human-vs-human leave-one-out case), the plain coverage is returned. This makes
    AI vs human comparable on review length.
    """
    A = [a for a in match["ai_items"] if a.get("channel") in channels]
    H = [h for h in match["human_items"] if h.get("channel") in channels]
    if not H or not A:
        return None
    hbase = {h["number"] for h in H}
    a_nums = {a["number"] for a in A}
    # AI concerns matching each human concern, within the same channel
    matchers = {}
    for m in match["matches"]:
        if m["human_number"] in hbase and m["ai_number"] in a_nums:
            matchers.setdefault(m["human_number"], set()).add(m["ai_number"])

    n_rev = len({h.get("reviewer_file") for h in match["human_items"]}) or 1
    N = len(a_nums)
    K = max(1, round(len(H) / n_rev))

    if N <= K:  # review at or below one reviewer's length: use plain coverage
        covered = sum(1 for h in hbase if matchers.get(h))
        return covered / len(H)

    # Average over all C(N, K) subsets: a human concern with d matchers is covered
    # by a fraction 1 - C(N-d, K)/C(N, K) of them.
    denom = math.comb(N, K)
    total = 0.0
    for h in hbase:
        d = len(matchers.get(h, ()))
        total += 1.0 - math.comb(N - d, K) / denom
    return total / len(H)


def _load_status_buckets():
    """ICLR paper_id -> status_bucket (accept/reject grouping). Empty if absent."""
    out = {}
    csv_path = DATA_ROOT / "ICLR26_rndm50" / "dataset.csv"
    if not csv_path.exists():
        return out
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pid = (row.get("id") or "").strip()
            if pid:
                out[pid] = (row.get("status_bucket") or "").strip() or "unknown"
    return out


## Per-pair metrics

def compute_pair(match, status_buckets):
    """One per_pair record from one parsed match file."""
    H = match["human_items"]   # list B (pooled humans, or pooled-other reviewers for h2h)
    A = match["ai_items"]      # list A (AI source, or held-out reviewer for h2h)
    matched_H = {m["human_number"] for m in match["matches"]}
    matched_A = {m["ai_number"] for m in match["matches"]}
    kind = match.get("comparison_kind", KIND_AI)

    cov_baseline = _coverage(H, matched_H, C_BASELINE)
    cov_critique = _coverage(H, matched_H, C_CRITIQUE)
    cov_neutral = _coverage(H, matched_H, C_WITH_NEUTRAL)
    cov_weighted = _weighted_coverage(H, matched_H)

    n_A_base = _count_in(A, C_BASELINE)
    matched_A_base = sum(1 for a in A if a.get("channel") in C_BASELINE
                         and a.get("number") in matched_A)
    precision = matched_A_base / n_A_base if n_A_base else None
    additional_count = n_A_base - matched_A_base
    additional_rate = additional_count / n_A_base if n_A_base else None

    # Weakness-only counts, for the pooled per-weakness precision reported below.
    n_A_crit = _count_in(A, C_CRITIQUE)
    matched_A_crit = sum(1 for a in A if a.get("channel") in C_CRITIQUE
                         and a.get("number") in matched_A)

    quantity = None
    if cov_baseline is not None:
        quantity = cov_baseline + QUANTITY_BONUS * additional_count

    return {
        "dataset": match["dataset"],
        "paper_id": match["paper_id"],
        "source": match["source"],
        "comparison_kind": kind,
        # group_source folds every leave-one-out reviewer under one label so the
        # AI sources and the human baseline sit side by side in aggregations.
        "group_source": match["source"] if kind == KIND_AI else "human_internal",
        "status_bucket": status_buckets.get(match["paper_id"], ""),
        "coverage_baseline": cov_baseline,
        "coverage_critique_only": cov_critique,
        "coverage_with_neutral": cov_neutral,
        "coverage_weighted": cov_weighted,
        "coverage_lengthnorm": _lengthnorm_coverage(match, C_CRITIQUE),  # weakness-only, matches primary metric
        "precision": precision,
        "additional_count": additional_count,
        "additional_rate": additional_rate,
        "quantity_exploratory": quantity,
        "n_H": len(H),
        "n_H_baseline": _count_in(H, C_BASELINE),
        "n_A": len(A),
        "n_A_baseline": n_A_base,
        "n_matched_H_baseline": sum(1 for h in H if h.get("channel") in C_BASELINE
                                    and h.get("number") in matched_H),
        "n_matched_A_baseline": matched_A_base,
        "n_A_critique": n_A_crit,
        "n_matched_A_critique": matched_A_crit,
        "n_matches": len(match["matches"]),
    }


def matched_pair_rows(match):
    """Explode one match file into per-pair inspection rows."""
    H_by = {h["number"]: h for h in match["human_items"]}
    A_by = {a["number"]: a for a in match["ai_items"]}
    rows = []
    for m in match["matches"]:
        a = A_by.get(m["ai_number"], {})
        h = H_by.get(m["human_number"], {})
        rows.append({
            "dataset": match["dataset"],
            "paper_id": match["paper_id"],
            "source": match["source"],
            "comparison_kind": match.get("comparison_kind", KIND_AI),
            "ai_number": m["ai_number"],
            "ai_text": a.get("text", ""),
            "human_number": m["human_number"],
            "human_text": h.get("text", ""),
            "human_reviewer": h.get("reviewer_file", ""),
            "channel": h.get("channel", ""),
            "type": h.get("type", ""),
            "seriousness": h.get("seriousness_category", ""),
            "justification": m.get("justification", ""),
        })
    return rows


def item_level_rows(match):
    """One row per human concern, with matched flag, for type/seriousness breakdowns."""
    matched_H = {m["human_number"] for m in match["matches"]}
    rows = []
    for h in match["human_items"]:
        rows.append({
            "dataset": match["dataset"],
            "paper_id": match["paper_id"],
            "source": match["source"],
            "comparison_kind": match.get("comparison_kind", KIND_AI),
            "group_source": match["source"] if match.get("comparison_kind", KIND_AI) == KIND_AI else "human_internal",
            "channel": h.get("channel"),
            "type": h.get("type"),
            "seriousness": h.get("seriousness_category"),
            "matched": h.get("number") in matched_H,
        })
    return rows


## Loading

def load_match_files(root=MATCHES_ROOT):
    return [json.loads(fp.read_text(encoding="utf-8"))
            for fp in sorted(root.rglob("*.json"))]


## Aggregation

_METRIC_COLS = [
    "coverage_baseline", "coverage_critique_only", "coverage_with_neutral",
    "coverage_weighted", "coverage_lengthnorm", "precision", "additional_rate",
    "quantity_exploratory",
]


def _mean_std(df, by, cols):
    g = df.groupby(by, dropna=False)
    agg = g[cols].agg(["mean", "std", "count"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    return agg.reset_index()


def _df_to_md(df, floatfmt="{:.3f}"):
    """Minimal markdown table (no tabulate dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return "" if math.isnan(v) else floatfmt.format(v)
        return str(v)
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join(
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    )
    return "\n".join([head, sep, body])


def build_aggregates(pair_df, item_df):
    tables = {}

    # 1. by group_source (gpt/llama/qwen/human_internal)
    tables["by_source"] = _mean_std(pair_df, ["group_source"], _METRIC_COLS)
    # 2. by dataset
    tables["by_dataset"] = _mean_std(pair_df, ["dataset"], _METRIC_COLS)
    # 3. source x dataset
    tables["by_source_dataset"] = _mean_std(pair_df, ["group_source", "dataset"], _METRIC_COLS)

    # 4. coverage by accept/reject (ICLR AI-vs-human only)
    iclr_ai = pair_df[(pair_df["dataset"] == "ICLR26_rndm50")
                      & (pair_df["comparison_kind"] == KIND_AI)]
    if not iclr_ai.empty:
        tables["by_status_bucket"] = _mean_std(
            iclr_ai, ["status_bucket", "group_source"], ["coverage_baseline"])

    # 6/7. coverage by human concern type and seriousness (item-level, AI-vs-human, weakness channel only).
    #      Concern-level recall pooled over papers/sources.
    crit = item_df[(item_df["comparison_kind"] == KIND_AI)
                   & (item_df["channel"].isin(C_CRITIQUE))]
    if not crit.empty:
        by_type = crit.groupby(["type", "group_source"], dropna=False)["matched"] \
            .agg(["mean", "count"]).reset_index().rename(columns={"mean": "coverage"})
        tables["by_type"] = by_type

        by_ser = crit.groupby(["seriousness", "group_source"], dropna=False)["matched"] \
            .agg(["mean", "count"]).reset_index().rename(columns={"mean": "coverage"})
        tables["by_seriousness"] = by_ser

    return tables


## Paper-level bootstrap confidence intervals

def _per_paper(pair_df, metric, group_source, dataset=None):
    """Mean per paper for (group_source, metric), averaging leave-one-out rows for human_internal."""
    sub = pair_df[pair_df["group_source"] == group_source]
    if dataset is not None:
        sub = sub[sub["dataset"] == dataset]
    sub = sub[["paper_id", metric]].dropna(subset=[metric])
    return sub.groupby("paper_id")[metric].mean()


def _bootstrap_ci(values, rng, n=BOOTSTRAP_SAMPLES):
    """Point estimate (mean) and 95% percentile bootstrap CI over the given values."""
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(vals.mean())
    idx = rng.integers(0, vals.size, size=(n, vals.size))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return point, float(lo), float(hi)


def build_bootstrap(pair_df):
    """Per-paper point estimates + 95% bootstrap CIs for the headline tables."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    src_rows = []
    for src in BOOTSTRAP_SOURCES:
        for metric in BOOTSTRAP_METRICS:
            s = _per_paper(pair_df, metric, src)
            point, lo, hi = _bootstrap_ci(s.values, rng)
            src_rows.append({"group_source": src, "metric": metric,
                             "point": point, "ci_lo": lo, "ci_hi": hi,
                             "n_papers": int(s.size)})
    by_source = pd.DataFrame(src_rows)

    venue_rows = []
    for src in BOOTSTRAP_SOURCES:
        for ds in BOOTSTRAP_DATASETS:
            s = _per_paper(pair_df, "coverage_baseline", src, dataset=ds)
            point, lo, hi = _bootstrap_ci(s.values, rng)
            venue_rows.append({"group_source": src, "dataset": ds,
                               "point": point, "ci_lo": lo, "ci_hi": hi,
                               "n_papers": int(s.size)})
    by_venue = pd.DataFrame(venue_rows)

    # Paired GPT - human length-normalized difference (same papers).
    g = _per_paper(pair_df, "coverage_lengthnorm", "gpt")
    h = _per_paper(pair_df, "coverage_lengthnorm", "human_internal")
    common = g.index.intersection(h.index)
    diff = (g.loc[common] - h.loc[common]).values
    dpoint, dlo, dhi = _bootstrap_ci(diff, rng)
    paired = {"point": dpoint, "ci_lo": dlo, "ci_hi": dhi, "n": int(len(common))}

    return {"by_source": by_source, "by_venue": by_venue, "paired_gpt_human_lengthnorm": paired}


def _ci_wide_md(boot_long, row_key, metrics):
    """Wide markdown: one row per source, cells 'point [lo, hi]' (two decimals)."""
    def cell(r):
        if r.empty or math.isnan(r["point"].iloc[0]):
            return ""
        v = r.iloc[0]
        return f"{v['point']:.2f} [{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]"
    head = "| " + row_key + " | " + " | ".join(metrics) + " |"
    sep = "| " + " | ".join("---" for _ in range(len(metrics) + 1)) + " |"
    lines = [head, sep]
    for src in BOOTSTRAP_SOURCES:
        cells = [src]
        for m in metrics:
            cells.append(cell(boot_long[(boot_long["group_source"] == src)
                                        & (boot_long["metric"] == m)]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pooled_weak_precision(pair_df):
    """Pooled per-weakness precision per source: matched weaknesses / all weaknesses.

    Item-weighted (every weakness counts once), so it answers "per weakness, how often
    does it match a human concern" -- the figure cited in the discussion. This differs
    from a per-paper mean of the ratio, which weights each paper equally.
    """
    rows = []
    for src in BOOTSTRAP_SOURCES:
        sub = pair_df[pair_df["group_source"] == src]
        tot = int(sub["n_A_critique"].sum())
        mat = int(sub["n_matched_A_critique"].sum())
        rows.append({"group_source": src,
                     "precision_critique_only_pooled": mat / tot if tot else float("nan"),
                     "n_matched_weak": mat, "n_weak": tot})
    return pd.DataFrame(rows)


## Driver

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    status_buckets = _load_status_buckets()

    matches = load_match_files()
    if not matches:
        print(f"no match files under {MATCHES_ROOT}, run run_full_matching.py first")
        return
    print(f"loaded {len(matches)} match files")

    pair_rows = [compute_pair(m, status_buckets) for m in matches]
    pair_df = pd.DataFrame(pair_rows)

    matched_rows = [r for m in matches for r in matched_pair_rows(m)]
    matched_df = pd.DataFrame(matched_rows)

    item_rows = [r for m in matches for r in item_level_rows(m)]
    item_df = pd.DataFrame(item_rows)

    # write per-pair + matched-pairs CSVs
    pair_df.to_csv(OUT_ROOT / "per_pair.csv", index=False)
    matched_df.to_csv(OUT_ROOT / "matched_pairs.csv", index=False)

    # aggregates
    tables = build_aggregates(pair_df, item_df)
    boot = build_bootstrap(pair_df)

    # tidy long-form aggregate.csv (concat of by_source / by_source_dataset)
    tables["by_source"].assign(grouping="by_source").to_csv(
        OUT_ROOT / "aggregate.csv", index=False)

    md_parts = ["# Overlap aggregates\n"]
    titles = {
        "by_source": "Coverage & precision by source (mean / std / n)",
        "by_source_dataset": "All metrics by source x dataset",
        "by_dataset": "All metrics by dataset",
        "by_status_bucket": "Baseline coverage by accept/reject (ICLR)",
        "by_seriousness": "Coverage by human critique seriousness",
        "by_type": "Coverage by human concern type",
    }
    for key in ["by_source", "by_source_dataset",
                "by_dataset", "by_status_bucket", "by_seriousness", "by_type"]:
        if key in tables:
            md_parts.append(f"\n## {titles[key]}\n")
            md_parts.append(_df_to_md(tables[key].round(3)))
            md_parts.append("")

    md_parts.append("\n## Per-weakness precision (pooled over all weakness concerns)\n")
    md_parts.append(_df_to_md(_pooled_weak_precision(pair_df).round(3)))
    md_parts.append("")

    # paper-level bootstrap CIs (these back thesis Tables 4.4 and 4.5)
    md_parts.append(f"\n## Paper-level bootstrap 95% CIs by source "
                    f"({BOOTSTRAP_SAMPLES} resamples, seed {BOOTSTRAP_SEED}, "
                    f"per-paper aggregation)\n")
    md_parts.append(_ci_wide_md(boot["by_source"], "group_source", BOOTSTRAP_METRICS))
    md_parts.append("")
    md_parts.append("\n## Paper-level bootstrap 95% CIs: baseline coverage by venue\n")
    for ds in BOOTSTRAP_DATASETS:
        sub = boot["by_venue"][boot["by_venue"]["dataset"] == ds].copy()
        sub["metric"] = "coverage_baseline"
        md_parts.append(f"\n### {ds}\n")
        md_parts.append(_ci_wide_md(sub, "group_source", ["coverage_baseline"]))
        md_parts.append("")
    p = boot["paired_gpt_human_lengthnorm"]
    md_parts.append("\n## Paired length-normalized difference (GPT minus human, same papers)\n")
    md_parts.append(f"mean difference = {p['point']:.3f}, "
                    f"95% CI [{p['ci_lo']:.3f}, {p['ci_hi']:.3f}], n = {p['n']} papers\n")

    (OUT_ROOT / "aggregate.md").write_text("\n".join(md_parts), encoding="utf-8")
    boot["by_source"].to_csv(OUT_ROOT / "bootstrap_ci_by_source.csv", index=False)
    boot["by_venue"].to_csv(OUT_ROOT / "bootstrap_ci_by_venue.csv", index=False)

    # --- console summary ---
    # Primary metric: critique-only coverage (strengths are upward-biased by
    # one-to-many matching of vague praise). Baseline shown alongside for context;
    # the in-corpus reference is the human-internal leave-one-out baseline.
    by_src = tables["by_source"]
    hi_row = by_src[by_src["group_source"] == "human_internal"]
    hi_crit = float(hi_row["coverage_critique_only_mean"].iloc[0]) if not hi_row.empty else float("nan")
    print("\ncoverage (critique only):")
    if not math.isnan(hi_crit):
        print(f"  human-internal baseline = {hi_crit:.3f}")
    for _, r in by_src.iterrows():
        n = int(r["coverage_critique_only_count"])
        mean = r["coverage_critique_only_mean"]
        std = r["coverage_critique_only_std"]
        std_s = f"{std:.3f}" if not (isinstance(std, float) and math.isnan(std)) else "  -  "
        print(f"  {r['group_source']} critique-only={mean:.3f} ± {std_s}  "
              f"baseline(crit+str)={r['coverage_baseline_mean']:.3f}  (n={n})")

    print("\nlength-normed (AI shrunk to one reviewer):")
    for _, r in by_src.iterrows():
        ln = r.get("coverage_lengthnorm_mean", float("nan"))
        bl = r["coverage_baseline_mean"]
        if not math.isnan(ln):
            print(f"  {r['group_source']} length-normed={ln:.3f}  (vs full baseline {bl:.3f})")

    print("\nper-weakness precision (pooled, matched weaknesses / all weaknesses):")
    for _, r in _pooled_weak_precision(pair_df).iterrows():
        print(f"  {r['group_source']} = {r['precision_critique_only_pooled']:.3f}  "
              f"({int(r['n_matched_weak'])}/{int(r['n_weak'])})")

    print(f"\nbootstrap 95% CIs:")
    for _, r in boot["by_source"].iterrows():
        if not math.isnan(r["point"]):
            print(f"  {r['group_source']} {r['metric']} "
                  f"{r['point']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  "
                  f"(n={int(r['n_papers'])})")
    pp = boot["paired_gpt_human_lengthnorm"]
    print(f"  paired length-norm GPT-human: {pp['point']:.3f} "
          f"[{pp['ci_lo']:.3f}, {pp['ci_hi']:.3f}]  (n={pp['n']})")

    print(f"\nwrote outputs to {OUT_ROOT}")


if __name__ == "__main__":
    main()
