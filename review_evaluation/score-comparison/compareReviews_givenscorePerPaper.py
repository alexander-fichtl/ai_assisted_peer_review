"""
"""

#####----- Imports --------------------------------------------------------------------------------------#####
import hashlib
import json
import re
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
#####----------------------------------------------------------------------------------------------------#####



#####----- Parameters -----------------------------------------------------------------------------------#####
DATASET_NAME = "ICLR26_rndm50"

###--- Paths
DIR_out = Path("review_evaluation/score-comparison/plots")
DIR_GENERATED_REVIEWS = Path("data/reviews_generated/ICLR")
DATASET_CSV = Path("data/final_datasets") / DATASET_NAME / "dataset.csv"   # raw dataset — no pre-generated CSVs needed

#Choose which LLM sources get visualized (folder names under data/generated_reviews/)
#If [], visualizes ALL sources found there
SOURCES_TO_VISUALIZE = [
    "gpt",
    "llama",
    "qwen"
]

###--- What to show
VAR_showHumanReviews = True
VAR_showLlmReviews = True
VAR_showDecisionShading = True   # Accept/reject background shading
VAR_showHumanIQR = True
VAR_showHumanMedian = True

#Labels
VAR_annotatedIDs = []   # These review_ids are ALWAYS annotated
SHOW_LABELS = False     # Show all review_id labels next to each point

#Visual
LLM_COLOR_CYCLE = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
GIVEN_SCORE_COL = "given score"
SCORE_RANGE = (1, 10)   # Fixed y-axis range; set to None for automatic
#####----------------------------------------------------------------------------------------------------#####



#####----- Human review parsing (from dataset.csv) ------------------------------------------------------#####
def _is_meta_review(note: Dict[str, Any]) -> bool:
    c = note.get("content") or {}
    if "metareview" in c:
        return True
    inv = (note.get("invitation") or "").lower()
    invs = " ".join(note.get("invitations") or []).lower()
    meta_tokens = ("meta_review", "metareview", "area_chair")
    return any(t in inv or t in invs for t in meta_tokens)

def _extract_human_reviewer_id(review_raw: Dict[str, Any]) -> str:
    for key in ("signatures", "writers", "readers"):
        for entry in review_raw.get(key, []) or []:
            if "Reviewer_" in entry:
                rid = entry.split("Reviewer_", 1)[-1]
                rid = "".join(ch for ch in rid if ch.isalnum())
                if rid:
                    return rid
    num = review_raw.get("number")
    if isinstance(num, int):
        return f"r{num}"
    rid = review_raw.get("id")
    if isinstance(rid, str) and rid.strip():
        s = rid.strip()[-8:]
        s = "".join(ch for ch in s if ch.isalnum())
        if s:
            return s
    digest = hashlib.sha256(json.dumps(review_raw, sort_keys=True).encode()).hexdigest()[:8]
    return f"h{digest}"

def _extract_numeric_rating_from_obj(obj) -> float | None:
    """Extract overall numeric rating from an OpenReview-style review dict."""
    if obj is None:
        return None
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            m = re.search(r"(\d+)", obj)
            return float(m.group(1)) if m else None

    if not isinstance(obj, dict):
        return None

    content = obj.get("content") or obj
    if isinstance(content, dict):
        for field in ["rating", "overall", "recommendation"]:
            if field not in content:
                continue
            v = content[field]
            if isinstance(v, dict):
                v = v.get("value")
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                m = re.search(r"(\d+)", v)
                if m:
                    return float(m.group(1))

    if isinstance(obj.get("rating"), (int, float, str)):
        m = re.search(r"(\d+)", str(obj["rating"]))
        return float(m.group(1)) if m else None

    return None

def load_human_scores_from_dataset_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read dataset.csv directly and extract one row per human review with its numeric rating.
    No pre-generated given_scores.csv needed.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"dataset.csv not found: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for base_row in reader:
            pid = (base_row.get("id") or base_row.get("paper_id") or "").strip()
            if not pid:
                continue

            decision = ""
            for col in ["status_bucket", "decision", "final_decision"]:
                if base_row.get(col, "").strip():
                    decision = base_row[col].strip()
                    break

            raw_reviews = base_row.get("reviews") or "[]"
            try:
                reviews = json.loads(raw_reviews)
            except Exception:
                reviews = []

            for r in reviews:
                if not isinstance(r, dict):
                    continue
                if _is_meta_review(r):
                    continue

                score = _extract_numeric_rating_from_obj(r)
                if score is None:
                    continue

                rid = _extract_human_reviewer_id(r)
                source_name = f"human_{rid}"
                review_id = f"{pid}_{source_name}"

                rows.append({
                    "paper_id": pid,
                    "review_id": review_id,
                    "human review": "yes",
                    "source": "human",
                    GIVEN_SCORE_COL: score,
                    "decision": decision,
                })

    df = pd.DataFrame(rows)
    print(f"[INFO] Loaded {len(df)} human review scores from {csv_path}")
    return df
#####----------------------------------------------------------------------------------------------------#####



#####----- LLM score extraction -------------------------------------------------------------------------#####
def _in_score_range(x: float) -> bool:
    return 0.0 <= x <= 10.0

def _extract_number_prefix(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", v.strip())
        return float(m.group(1)) if m else None
    return None

def _extract_openreview_overall_from_string(s: str):
    if not isinstance(s, str):
        return None
    t = s.replace("\r\n", "\n").replace("\r", "\n")

    m = re.search(r"(?i)\boverall\s*score\b\s*[:\-–—]\s*([0-9]+(?:\.[0-9]+)?)\b", t)
    if m:
        x = float(m.group(1))
        return x if _in_score_range(x) else None

    m = re.search(r"(?is)\b([0-9]+(?:\.[0-9]+)?)\s*:\s*[^\n]{0,60}?\b(?:accept|reject)\b", t)
    if m:
        x = float(m.group(1))
        return x if _in_score_range(x) else None

    m2 = re.search(r"(?is)\boverall\s*score\b(.{0,400})", t)
    if m2:
        seg = m2.group(1)
        m3 = re.search(r"(?is)\b([0-9]+(?:\.[0-9]+)?)\s*:\s*[^\n]{0,60}?\b(?:accept|reject)\b", seg)
        if m3:
            x = float(m3.group(1))
            return x if _in_score_range(x) else None
        m4 = re.search(r"(?is)\b([0-9]+(?:\.[0-9]+)?)\b", seg)
        if m4:
            x = float(m4.group(1))
            return x if _in_score_range(x) else None

    return None

def _extract_gpt5_q9_strict(text: str):
    m = re.search(r"Q9\s*\(.*?\).*?\*\*Rating:\s*([0-9]+(?:\.[0-9]+)?)\s*\*\*", text, re.DOTALL)
    return float(m.group(1)) if m else None

def _extract_overall_from_text(text: str):
    if not isinstance(text, str) or not text.strip():
        return None
    x = _extract_openreview_overall_from_string(text)
    if x is not None:
        return x
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    m = re.search(r"(?is)\boverall\b\s*(?:rating|score)?\s*[:\-–—]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/10)?", t)
    if m:
        x = float(m.group(1))
        return x if _in_score_range(x) else None
    m = re.search(r"(?is)\bQ9\b.*?(?:rating|score)\s*[:\-–—]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/10)?", t)
    if m:
        x = float(m.group(1))
        return x if _in_score_range(x) else None
    return None

def _extract_overall_from_json(obj):
    if obj is None:
        return None
    if isinstance(obj, str):
        return _extract_overall_from_text(obj)
    if not isinstance(obj, dict):
        return None

    content = obj.get("content")
    if isinstance(content, dict):
        rating = content.get("rating")
        if isinstance(rating, dict) and "value" in rating:
            v = rating["value"]
            if isinstance(v, (int, float)):
                x = float(v)
                return x if _in_score_range(x) else None
            if isinstance(v, str):
                x = _extract_openreview_overall_from_string(v)
                if x is not None:
                    return x
                x = _extract_number_prefix(v)
                if x is not None and _in_score_range(x):
                    return x
        if isinstance(rating, (int, float, str)):
            x = _extract_openreview_overall_from_string(str(rating))
            if x is not None:
                return x
            x = _extract_number_prefix(rating)
            return x if (x is not None and _in_score_range(x)) else None
        for k in ["review", "text", "comment", "final_review"]:
            if k in content and isinstance(content[k], str):
                x = _extract_overall_from_text(content[k])
                if x is not None:
                    return x

    if isinstance(obj.get("rating"), (int, float, str)):
        x = _extract_openreview_overall_from_string(str(obj["rating"]))
        if x is not None:
            return x
        x = _extract_number_prefix(obj["rating"])
        return x if (x is not None and _in_score_range(x)) else None

    for k in ["review", "text", "comment", "final_review"]:
        if k in obj and isinstance(obj[k], str):
            x = _extract_overall_from_text(obj[k])
            if x is not None:
                return x

    return None

def build_llm_review_dirs(root: Path, allowlist=None) -> dict:
    if not root.exists():
        print(f"[ERROR] DIR_GENERATED_REVIEWS not found: {root.resolve()}")
        return {}
    all_sources = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if allowlist is None or len(allowlist) == 0:
        chosen = all_sources
    else:
        chosen = list(allowlist)
        missing = [s for s in chosen if s not in all_sources]
        if missing:
            print(f"[WARN] Missing sources under {root}: {missing}")
    print(f"[INFO] Found sources: {all_sources}")
    print(f"[INFO] Chosen sources: {chosen}")
    return {name: (root / name) for name in chosen}

def load_llm_scores(root: Path, source_name: str) -> pd.DataFrame:
    rows = []
    if not root.exists():
        print(f"[{source_name}] Directory not found: {root}")
        return pd.DataFrame()

    for paper_dir in root.iterdir():
        if not paper_dir.is_dir():
            continue
        paper_id = paper_dir.name
        score = None

        if source_name == "gpt5.1thinking":
            txt_path = paper_dir / f"{paper_id}.txt"
            if txt_path.exists():
                text = txt_path.read_text(encoding="utf-8", errors="ignore")
                score = _extract_gpt5_q9_strict(text)
            if score is None:
                for jp in sorted(paper_dir.glob("**/*.json")):
                    try:
                        obj = json.loads(jp.read_text(encoding="utf-8", errors="ignore"))
                    except Exception:
                        continue
                    score = _extract_overall_from_json(obj)
                    if score is not None:
                        break
            if score is None:
                print(f"[{source_name}] No Q9 rating found for {paper_id}")
                continue
        else:
            p0 = paper_dir / "final_review_openreview.json"
            json_files = sorted(paper_dir.glob("**/*.json"))
            preferred_json = []
            if p0.exists():
                preferred_json.append(p0)
            preferred_json += [p for p in json_files if "openreview" in p.name.lower() and p not in preferred_json]
            preferred_json += [p for p in json_files if "final_review" in p.name.lower() and p not in preferred_json]
            preferred_json += [p for p in json_files if p not in preferred_json]

            for jp in preferred_json:
                try:
                    obj = json.loads(jp.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    continue
                score = _extract_overall_from_json(obj)
                if score is not None:
                    break

            if score is None:
                txt_files = sorted(paper_dir.glob("**/*.txt"))
                preferred = paper_dir / f"{paper_id}.txt"
                preferred_txt = ([preferred] if preferred.exists() else []) + [p for p in txt_files if p != preferred]
                for tp in preferred_txt:
                    try:
                        text = tp.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        continue
                    score = _extract_overall_from_text(text)
                    if score is not None:
                        break

            if score is None:
                print(f"[{source_name}] No overall rating found for {paper_id}")
                continue

        rows.append({
            "paper_id": str(paper_id),
            "review_id": f"{paper_id}_{source_name}",
            "human review": "no",
            GIVEN_SCORE_COL: float(score),
            "source": source_name,
            "decision": np.nan,
        })

    df = pd.DataFrame(rows)
    print(f"[INFO] {source_name}: parsed rows = {len(df)}")
    return df
#####----------------------------------------------------------------------------------------------------#####



#####----- Plotting -------------------------------------------------------------------------------------#####
def compute_human_iqr_and_median(df: pd.DataFrame, metric_col: str, pid_to_x: dict):
    humans = df[df["human review"] == "yes"].dropna(subset=[metric_col])
    if humans.empty:
        return None
    median_x, median_y, q25_y, q75_y = [], [], [], []
    for pid in sorted(pid_to_x.keys(), key=lambda p: pid_to_x[p]):
        vals = humans.loc[humans["paper_id"] == pid, metric_col].astype(float)
        if vals.empty:
            continue
        x = pid_to_x[pid]
        arr = vals.values
        median_x.append(x)
        median_y.append(float(np.median(arr)))
        q25_y.append(float(np.percentile(arr, 25)))
        q75_y.append(float(np.percentile(arr, 75)))
    if not median_x:
        return None
    return median_x, median_y, q25_y, q75_y

def apply_jitter_same_y(sub: pd.DataFrame, metric_col: str, step: float = 0.06, scale: float = 0.8) -> pd.DataFrame:
    sub = sub.copy()
    sub["x_jitter"] = sub["x"].astype(float)
    sub["_score_val"] = sub[metric_col].astype(float)
    for (pid, score), idx in sub.groupby(["paper_id", "_score_val"]).groups.items():
        idx = list(idx)
        n = len(idx)
        if n <= 1:
            continue
        base_x = sub.loc[idx[0], "x"]
        eff = step * scale
        start = -eff * (n - 1) / 2.0
        offsets = [start + i * eff for i in range(n)]
        sub.loc[idx, "x_jitter"] = [base_x + off for off in offsets]
    return sub.drop(columns="_score_val")

def make_given_score_plot(df: pd.DataFrame, dataset_name: str, show_humans: bool, show_llms: bool, show_labels: bool, annotate_ids, label_suffix: str = ""):
    if not show_humans and not show_llms:
        show_humans = True
        show_llms = True

    DIR_out.mkdir(parents=True, exist_ok=True)

    paper_ids = sorted(df["paper_id"].unique())
    pid_to_x = {pid: i for i, pid in enumerate(paper_ids)}

    df = df.copy()
    df["x"] = df["paper_id"].map(pid_to_x)
    df = df.dropna(subset=[GIVEN_SCORE_COL, "x"])

    llm_sources = sorted(src for src in df["source"].unique() if src != "human")
    llm_color_map = {src: LLM_COLOR_CYCLE[i % len(LLM_COLOR_CYCLE)] for i, src in enumerate(llm_sources)}

    human_color = "black"
    annotate_ids = set(str(r) for r in annotate_ids)

    pid_to_decision = {}
    if "decision" in df.columns:
        for pid, vals in df.groupby("paper_id")["decision"]:
            val = next((v for v in vals if isinstance(v, str) and v.strip()), None)
            if val is not None:
                pid_to_decision[pid] = val

    metric_col = GIVEN_SCORE_COL
    scores = df[metric_col].astype(float)
    if SCORE_RANGE is not None:
        ymin, ymax = SCORE_RANGE
    else:
        ymin = float(scores.min())
        ymax = float(scores.max())
        if ymin == ymax:
            ymin -= 0.5
            ymax += 0.5

    plt.figure(figsize=(20, 6))
    mask = pd.Series(False, index=df.index, dtype=bool)
    if show_humans:
        mask |= (df["human review"] == "yes")
    if show_llms:
        mask |= (df["human review"] == "no")

    sub = df[mask].copy()
    if sub.empty:
        print(f"No data to plot for {dataset_name}")
        plt.close()
        return

    sub = apply_jitter_same_y(sub, metric_col)
    ax = plt.gca()

    if VAR_showDecisionShading and pid_to_decision:
        for pid, x in pid_to_x.items():
            dec = str(pid_to_decision.get(pid, "")).strip().lower()
            if not dec:
                continue
            if "accept" in dec:
                color = "#CCFFCC"
            elif "reject" in dec:
                color = "#FFCCCC"
            else:
                continue
            ax.axvspan(x - 0.5, x + 0.5, color=color, alpha=0.2, zorder=0)

    if show_humans:
        h = sub[sub["human review"] == "yes"]
        if not h.empty:
            ax.scatter(h["x_jitter"], h[metric_col].astype(float), marker="o", s=40,
                       color=human_color, alpha=0.3, label="Score given by human review", zorder=4)

    if show_llms:
        for src in llm_sources:
            sys_points = sub[(sub["human review"] == "no") & (sub["source"] == src)]
            if sys_points.empty:
                continue
            color = llm_color_map[src]
            ax.scatter(sys_points["x_jitter"], sys_points[metric_col].astype(float), marker="x", s=40,
                       color=color, alpha=0.9, label=f"Score given by {src} review", zorder=5)
            line_data = sys_points.sort_values("x")
            ax.plot(line_data["x"], line_data[metric_col].astype(float), linestyle="-", linewidth=1.5,
                    color=color, alpha=0.7, zorder=3)

    stats = compute_human_iqr_and_median(df, metric_col, pid_to_x)
    if stats:
        median_x, median_y, q25_y, q75_y = stats
        if VAR_showHumanIQR:
            ax.fill_between(median_x, q25_y, q75_y, color="black", alpha=0.07,
                            label="IQR of human given scores", zorder=1)
        if VAR_showHumanMedian:
            ax.plot(median_x, median_y, linestyle="--", linewidth=1.5, color="black", alpha=0.4,
                    label="Median human given score", zorder=2)

    if show_labels or annotate_ids:
        yrange = ymax - ymin
        for _, row in sub.iterrows():
            rid = row["review_id"]
            if show_labels or rid in annotate_ids:
                ax.text(row["x_jitter"], float(row[metric_col]) + 0.02 * yrange, rid,
                        ha="center", va="bottom", rotation=90, fontsize=6, zorder=6)

    ax.set_ylim(ymin - 0.1, ymax + 0.1)
    ax.set_xlim(-0.5, len(paper_ids) - 0.5)
    ax.set_xlabel("paper_id")
    ax.set_ylabel("given score")
    ax.set_title(f"Scores given by human and LLM reviews per paper (dataset: {dataset_name})")
    ax.set_xticks(range(len(paper_ids)))
    ax.set_xticklabels(paper_ids, rotation=90, fontsize=6)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best", fontsize=8)

    out_path = DIR_out / f"givenScorePerPaper_{dataset_name}{label_suffix}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot to {out_path}")
#####----------------------------------------------------------------------------------------------------#####



#####----- Correlation ----------------------------------------------------------------------------------#####
def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each LLM source, compute Pearson and Spearman correlation of its given score
    with the per-paper human median and human mean given score.
    """
    humans = df[df["source"] == "human"].copy()
    humans[GIVEN_SCORE_COL] = humans[GIVEN_SCORE_COL].astype(float)
    human_median = humans.groupby("paper_id")[GIVEN_SCORE_COL].median().rename("human_median")
    human_mean   = humans.groupby("paper_id")[GIVEN_SCORE_COL].mean().rename("human_mean")
    human_agg = human_median.to_frame().join(human_mean)

    llm_sources = sorted(s for s in df["source"].unique() if s != "human")
    records = []
    for src in llm_sources:
        llm = df[df["source"] == src][["paper_id", GIVEN_SCORE_COL]].copy()
        llm[GIVEN_SCORE_COL] = llm[GIVEN_SCORE_COL].astype(float)
        merged = llm.merge(human_agg, on="paper_id", how="inner")
        n = len(merged)
        if n < 3:
            print(f"[CORR] {src}: only {n} paired papers — skipping")
            continue
        y = merged[GIVEN_SCORE_COL].values
        row = {"source": src, "n_papers": n}
        for agg_label, col in [("median", "human_median"), ("mean", "human_mean")]:
            x = merged[col].values
            pearson_r, pearson_p   = stats.pearsonr(x, y)
            spearman_r, spearman_p = stats.spearmanr(x, y)
            row[f"pearson_r_{agg_label}"]  = round(pearson_r,  4)
            row[f"pearson_p_{agg_label}"]  = round(pearson_p,  4)
            row[f"spearman_r_{agg_label}"] = round(spearman_r, 4)
            row[f"spearman_p_{agg_label}"] = round(spearman_p, 4)
            print(f"[CORR/{agg_label:6s}] {src:20s}  n={n:3d}  "
                  f"Pearson r={pearson_r:+.4f} (p={pearson_p:.4f})  "
                  f"Spearman r={spearman_r:+.4f} (p={spearman_p:.4f})")
        records.append(row)

    corr_df = pd.DataFrame(records)
    if not corr_df.empty:
        out_csv = DIR_out / f"correlations_{DATASET_NAME}.csv"
        corr_df.to_csv(out_csv, index=False)
        print(f"[CORR] Saved to {out_csv}")
    return corr_df
#####----------------------------------------------------------------------------------------------------#####



#####----- Main -----------------------------------------------------------------------------------------#####
def main():
    # Load human scores directly from dataset.csv — no pre-generated CSVs needed
    df = load_human_scores_from_dataset_csv(DATASET_CSV)

    # Load and merge LLM scores from generated review folders
    llm_review_dirs = build_llm_review_dirs(DIR_GENERATED_REVIEWS, SOURCES_TO_VISUALIZE)
    for source_name, root in llm_review_dirs.items():
        llm_df = load_llm_scores(root, source_name=source_name)
        if not llm_df.empty:
            df = df[~((df["human review"] == "no") & (df["source"] == source_name))].copy()
            df = pd.concat([df, llm_df], ignore_index=True, sort=False)

    # Keep only selected sources + human
    if SOURCES_TO_VISUALIZE:
        allowed = set(SOURCES_TO_VISUALIZE) | {"human"}
        df = df[df["source"].isin(allowed)].copy()

    print("\n--- Correlations: LLM given score vs. human median given score ---")
    compute_correlations(df)
    print()

    make_given_score_plot(
        df=df,
        dataset_name=DATASET_NAME,
        show_humans=VAR_showHumanReviews,
        show_llms=VAR_showLlmReviews,
        show_labels=SHOW_LABELS,
        annotate_ids=VAR_annotatedIDs,
    )

    first20_ids = set(sorted(df["paper_id"].unique())[:20])
    make_given_score_plot(
        df=df[df["paper_id"].isin(first20_ids)],
        dataset_name=DATASET_NAME,
        show_humans=VAR_showHumanReviews,
        show_llms=VAR_showLlmReviews,
        show_labels=SHOW_LABELS,
        annotate_ids=VAR_annotatedIDs,
        label_suffix="_first20",
    )

main()
#####----------------------------------------------------------------------------------------------------#####