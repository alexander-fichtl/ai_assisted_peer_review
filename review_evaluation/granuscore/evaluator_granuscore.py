"""
Library for GranuScore comparison experiments.
Data loading, scoring, filtering, saving, and plotting.
Call run_comparison().
"""


#####----- Imports ----------------------------------------------------------------------------------------------------------#####
import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import matplotlib.colors as mcolors

from typing import Any

from granuscore import GranuScore
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Variables --------------------------------------------------------------------------------------------------------#####
TextDict = dict[str, dict[str, str]]        # {review_id: {"paper_id": ..., "text": ...}}
ScoredDict = dict[str, dict[str, Any]]      # same + "granu_score", "granu_details", ...

_granu_score = GranuScore()
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Data loading  ----------------------------------------------------------------------------------------------------------#####
def load_naturecom_generated(reviewer: str, data_dir: Path, key_sep: str = "__") -> TextDict:
    """Load from data_dir/<reviewer>/<paper_id>/final_review.txt or <paper_id>.txt"""
    reviewer_dir = data_dir / reviewer
    if not reviewer_dir.exists():
        raise FileNotFoundError(f"Reviewer folder not found: {reviewer_dir}")

    out: TextDict = {}
    for paper_dir in sorted(reviewer_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        pid = paper_dir.name

        # Try final_review.txt first, then <paper_id>.txt
        txt_path = paper_dir / "final_review.txt"
        if not txt_path.exists():
            txt_path = paper_dir / f"{pid}.txt"
        if not txt_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        out[f"{reviewer}{key_sep}{pid}"] = {"paper_id": pid, "text": text}
    return out

def load_naturecom_human(human_dir: Path) -> TextDict:
    """Load from human_dir/reviews_<paper_id>.json (first_remarks only)."""
    if not human_dir.exists():
        raise FileNotFoundError(f"Human reviews folder not found: {human_dir}")

    out: TextDict = {}
    for json_path in sorted(human_dir.glob("reviews_*.json")):
        pid = json_path.stem.removeprefix("reviews_")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for item in data.get("first_remarks", []):
            if not isinstance(item, dict):
                continue
            if str(item.get("invalid", "")).strip().lower() in {"yes", "y", "true", "1"}:
                continue
            reviewer_id = str(item.get("reviewer_id", "")).strip()
            text = (item.get("remarks_to_the_author") or "").strip()
            if not reviewer_id or not text:
                continue
            rid = _unique_key(out, f"{pid}_human_{reviewer_id}")
            out[rid] = {"paper_id": pid, "text": text}
    return out

def load_openreview_generated(reviewer: str, data_dir: Path, key_sep: str = "__", *, json_filename: str = "final_review_openreview.json", write_txt: bool = True) -> TextDict:
    """Load from OpenReview JSON; optionally write final_review.txt alongside it."""
    reviewer_dir = data_dir / reviewer
    if not reviewer_dir.exists():
        raise FileNotFoundError(f"Reviewer folder not found: {reviewer_dir}")

    out: TextDict = {}
    for paper_dir in sorted(p for p in reviewer_dir.iterdir() if p.is_dir()):
        json_path = paper_dir / json_filename
        if not json_path.exists():
            continue

        data = json.loads(json_path.read_text(encoding="utf-8"))
        pid = str(data.get("pdfname") or paper_dir.name)
        text = parse_openreview_to_plaintext(data.get("content") or {})
        if not text.strip():
            continue

        if write_txt:
            (paper_dir / "final_review.txt").write_text(text, encoding="utf-8")

        out[f"{reviewer}{key_sep}{pid}"] = {"paper_id": pid, "text": text}
    return out

def load_openreview_human(csv_path: Path, *, require_human_flag: bool = True) -> TextDict:
    """Load human reviews from CSV (columns: review_id, paper_id, review, [human review])."""
    if not csv_path.exists():
        raise FileNotFoundError(f"ICLR human review CSV not found: {csv_path}")

    out: TextDict = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not row:
                continue
            if require_human_flag:
                flag = (row.get("human review") or "").strip().lower()
                if flag and flag not in {"yes", "y", "true", "1"}:
                    continue

            review_id = (row.get("review_id") or "").strip()
            pid = (row.get("paper_id") or "").strip()
            text = (row.get("review") or "").strip()
            if not review_id or not pid or not text:
                continue

            review_id = _unique_key(out, review_id)
            out[review_id] = {"paper_id": pid, "text": text}
    return out

def load_openreview_human_from_dataset(dataset_csv: Path) -> TextDict:
    """
    Load human OpenReview reviews directly from final_datasets/<name>/dataset.csv.
    Uses column:
      - id
      - reviews   (JSON list of review notes)
    """
    if not dataset_csv.exists():
        raise FileNotFoundError(f"ICLR dataset CSV not found: {dataset_csv}")

    out: TextDict = {}
    with dataset_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if not row:
                continue

            pid = (row.get("id") or "").strip()
            raw_reviews = row.get("reviews") or ""
            if not pid or not raw_reviews.strip():
                continue

            try:
                reviews = json.loads(raw_reviews)
            except Exception:
                continue

            if not isinstance(reviews, list):
                continue

            for item in reviews:
                if not isinstance(item, dict):
                    continue
                if _is_openreview_meta_review(item):
                    continue

                reviewer_id = _extract_openreview_reviewer_id(item)
                text = parse_openreview_to_plaintext(item.get("content") or {})
                if not text.strip():
                    continue

                rid = _unique_key(out, f"{pid}_human_{reviewer_id}")
                out[rid] = {"paper_id": pid, "text": text}

    return out
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Data preprocessing  ----------------------------------------------------------------------------------------------------------#####
def parse_openreview_to_plaintext(content: dict[str, Any]) -> str:
    _OR_PREFERRED_ORDER = ["summary", "strengths", "weaknesses", "questions", "soundness", "presentation", "contribution", "rating", "confidence", "code_of_conduct", "flag_for_ethics_review"]
    _OR_LABEL_MAP = {"summary": "Summary", "strengths": "Strengths", "weaknesses": "Weaknesses", "questions": "Suggestions/Questions", "soundness": "Soundness", "presentation": "Presentation", "contribution": "Contribution", "rating": "Overall", "confidence": "Confidence", "code_of_conduct": "Conduct", "flag_for_ethics_review": "Ethics"}
    _OR_SECTION_KEYS = {"summary", "strengths", "weaknesses", "questions"}
    _OR_SCORE_KEYS = {"soundness", "presentation", "contribution", "rating", "confidence", "code_of_conduct", "flag_for_ethics_review"}

    if not isinstance(content, dict):
        return ""

    ordered = [k for k in _OR_PREFERRED_ORDER if k in content]
    ordered += sorted(k for k in content if k not in _OR_PREFERRED_ORDER)

    def _val(entry):
        return entry.get("value") if isinstance(entry, dict) and "value" in entry else entry

    def _block(v):
        if v is None:
            return ""
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, (int, float, bool)):
            return str(v)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            items = [x.strip() for x in v if x and x.strip()]
            return "\n".join(f"- {x}" for x in items) if items else ""
        return json.dumps(v, ensure_ascii=False, indent=2) if isinstance(v, (list, dict)) else str(v).strip()

    def _oneline(v):
        if isinstance(v, list):
            for x in v:
                if isinstance(x, str) and x.strip():
                    return x.strip()
        return " ".join(_block(v).split())

    blocks = []
    for k in ordered:
        val = _val(content.get(k))
        if val is None or (isinstance(val, str) and not val.strip()):
            continue
        title = _OR_LABEL_MAP.get(k, k.replace("_", " ").strip().title())
        if k in _OR_SECTION_KEYS:
            t = _block(val)
            if t:
                blocks.append(f"{title}:\n{t}")
        elif k in _OR_SCORE_KEYS:
            t = _oneline(val)
            if t:
                blocks.append(f"{title}: {t}")
        else:
            t = _block(val)
            if t:
                blocks.append(f"{title}:\n{t}")
    return "\n\n".join(blocks).strip()

def split_iclr_sections(text: str) -> dict[str, str]:
    """Split an ICLR-style review into {summary, strengths, weaknesses, questions}."""
    _SCORE_STOP_RE = re.compile(r"^\s*(soundness|presentation|contribution|overall|rating|confidence|conduct|ethics)\b\s*:?\s*", flags=re.IGNORECASE)

    _HEADING_MAP = {
        "summary": {"summary"},
        "strengths": {"strengths", "strength"},
        "weaknesses": {"weaknesses", "weakness"},
        "questions": {
            "questions", "question",
            "suggestions/questions", "suggestions / questions",
            "suggestions/questions for the authors",
            "questions and suggestions for the authors",
        },
    }

    empty = {"summary": "", "strengths": "", "weaknesses": "", "questions": ""}
    if not text:
        return empty

    def clean(line):
        line = line.lstrip("\ufeff\u200b\u200c\u200d")
        line = re.sub(r"^\s*#{1,6}\s*", "", line)
        return line.strip()

    def norm(s):
        return re.sub(r"\s+", " ", s.strip().lower()).rstrip(":").strip()

    def canonical(heading):
        h = norm(heading)
        for canon, variants in _HEADING_MAP.items():
            if h in variants:
                return canon
        return None

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    hits: list[tuple[int, str, str]] = []

    for i, raw in enumerate(lines):
        s = clean(raw)
        if not s or _SCORE_STOP_RE.match(s):
            continue
        if ":" in s:
            head, tail = s.split(":", 1)
            c = canonical(head)
            if c:
                hits.append((i, c, tail.strip()))
        else:
            c = canonical(s)
            if c:
                hits.append((i, c, ""))

    if not hits:
        return empty

    hits.sort(key=lambda x: x[0])
    sections = dict(empty)

    for idx, (start_i, canon, inline) in enumerate(hits):
        end_i = hits[idx + 1][0] if idx + 1 < len(hits) else len(lines)
        chunk_lines = ([inline] if inline else []) + lines[start_i + 1 : end_i]
        cleaned = []
        for cl in chunk_lines:
            if _SCORE_STOP_RE.match(clean(cl)):
                break
            cleaned.append(cl)
        chunk = "\n".join(cleaned).strip()
        if chunk:
            sections[canon] = (sections[canon].rstrip() + "\n\n" + chunk).strip() if sections[canon] else chunk

    return sections
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Granuscore ----------------------------------------------------------------------------------------------------------#####
def score_reviews(reviews: TextDict, *, detailed: bool = True) -> ScoredDict:
    """Score each review with GranuScore (default settings)."""
    out: ScoredDict = {}
    for rid, item in reviews.items():
        base: dict[str, Any] = {"paper_id": item["paper_id"], "review_id": rid, "text": item["text"]}

        if detailed:
            res = _granu_score(item["text"], detailed_infos=True)
            pooled = _safe_float(res.get("pooled_score"))
            atomic = {str(k): float(v) for k, v in (res.get("atomic_scores") or {}).items()}
            base["granu_score"] = pooled
            base["granu_details"] = {"answer": res.get("answer"), "pooled_score": pooled, "atomic_scores": atomic}
        else:
            base["granu_score"] = _safe_float(_granu_score(item["text"], detailed_infos=False))

        out[rid] = base
    return out

def score_sections(reviews: TextDict, sections: list[str] | None = None) -> dict[str, dict[str, float]]:
    """Mean GranuScore per (section, paper_id). Uses GranuScore defaults."""
    sections = sections or ["summary", "strengths", "weaknesses", "questions"]
    bucket: dict[str, dict[str, list[float]]] = {s: {} for s in sections}

    for item in reviews.values():
        pid, txt = item.get("paper_id", ""), item.get("text", "")
        if not pid or not txt:
            continue
        sec_texts = split_iclr_sections(txt)
        for sec in sections:
            st = (sec_texts.get(sec) or "").strip()
            if not st:
                continue
            try:
                score = _safe_float(_granu_score(st, detailed_infos=False))
            except Exception:
                score = None
            if score is not None and math.isfinite(score):
                bucket[sec].setdefault(pid, []).append(score)

    return {
        sec: {pid: sum(vals) / len(vals) for pid, vals in bucket[sec].items() if vals}
        for sec in sections
    }
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Helper functions ----------------------------------------------------------------------------------------------------------#####
def filter_to_common_papers(*textdicts: TextDict) -> tuple[list[TextDict], set[str]]:
    """Keep only entries whose paper_id exists in ALL provided textdicts."""
    pid_sets = [{str(v.get("paper_id", "")) for v in td.values() if v.get("paper_id")} for td in textdicts]
    common = pid_sets[0]
    for s in pid_sets[1:]:
        common &= s
    return [{k: v for k, v in td.items() if str(v.get("paper_id", "")) in common} for td in textdicts], common

def apply_limit(textdict: TextDict, limit: int | None) -> TextDict:
    """Deterministic 'first N' by (paper_id, review_id)."""
    if limit is None:
        return textdict
    items = sorted(textdict.items(), key=lambda kv: (kv[1].get("paper_id", ""), kv[0]))
    return dict(items[:limit])

def mean_granu_score(scored: ScoredDict) -> tuple[float | None, int]:
    """(mean, count) of finite granu_scores."""
    vals = [float(v["granu_score"]) for v in scored.values()
            if v.get("granu_score") is not None and math.isfinite(float(v["granu_score"]))]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None

def _unique_key(d: dict, base: str, suffix: str = "") -> str:
    key = base
    if key in d and suffix:
        key = f"{base}_{suffix}"
    if key in d:
        n = 2
        while f"{key}_dup{n}" in d:
            n += 1
        key = f"{key}_dup{n}"
    return key

def _darken(color, factor=0.6):
    r, g, b = mcolors.to_rgb(color)
    return (max(0, r * factor), max(0, g * factor), max(0, b * factor))

###--- Openreview stuff
def _is_openreview_meta_review(note: dict[str, Any]) -> bool:
    c = note.get("content") or {}
    if "metareview" in c:
        return True

    inv = (note.get("invitation") or "").lower()
    invs = " ".join(note.get("invitations") or []).lower()

    meta_tokens = ("meta_review", "metareview", "area_chair")
    return any(t in inv for t in meta_tokens) or any(t in invs for t in meta_tokens)

def _extract_openreview_reviewer_id(review_raw: dict[str, Any]) -> str:
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
        s = "".join(ch for ch in rid.strip() if ch.isalnum())
        if s:
            return s[-8:] if len(s) > 8 else s

    return "unknown"

###--- Save stuff
def save_json(data: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def _wrap_summary(scored, *, avg_generated, avg_human, n_generated, n_human, group):
    diff = float(avg_generated - avg_human) if (avg_generated is not None and avg_human is not None) else None
    return {
        "avg_diff_generated_minus_human": diff,
        "avg_generated": avg_generated,
        "avg_human": avg_human,
        "n_generated_used_for_avg": n_generated,
        "n_human_used_for_avg": n_human,
        "group": group,
        "reviews": scored,
    }

def _save_fig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Plots ----------------------------------------------------------------------------------------------------------#####
def plot_single(scored_gen: ScoredDict, scored_human: ScoredDict, out_path: Path, *, title: str = "", annotate_ids: set[str] | None = None, show_labels: bool = False):
    """Per-paper scatter: one generated line + human points with IQR band."""
    paper_ids, pid_to_x = _paper_axis(scored_gen, scored_human)
    gen_pts = _gen_points(scored_gen, paper_ids, pid_to_x)
    human_pts, human_by_pid = _human_points(scored_human, paper_ids, pid_to_x)

    fig, ax = _make_figure(len(paper_ids))
    _draw_human_band(ax, human_by_pid, pid_to_x, paper_ids)
    _draw_human_scatter(ax, human_pts)
    if gen_pts:
        pts = sorted(gen_pts, key=lambda t: t[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker="x", ls="-", lw=2.5, alpha=0.8, label="Generated reviews", zorder=5)

    _finish_axes(ax, paper_ids, title)
    _annotate(ax, human_pts + gen_pts, annotate_ids, show_labels)
    _save_fig(fig, out_path)

def plot_multi(scored_by_label: dict[str, ScoredDict], scored_human: ScoredDict, out_path: Path, *, title: str = "", annotate_ids: set[str] | None = None, show_labels: bool = False):
    """Per-paper scatter with multiple generated lines."""
    all_scored = dict(scored_human)
    for sd in scored_by_label.values():
        all_scored.update(sd)
    paper_ids, pid_to_x = _paper_axis_from_all(all_scored)
    human_pts, human_by_pid = _human_points(scored_human, paper_ids, pid_to_x)

    fig, ax = _make_figure(len(paper_ids))
    _draw_human_band(ax, human_by_pid, pid_to_x, paper_ids)
    _draw_human_scatter(ax, human_pts)

    for label in sorted(scored_by_label):
        pts = sorted(_gen_points(scored_by_label[label], paper_ids, pid_to_x), key=lambda t: t[0])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker="x", ls="-", lw=2.5, alpha=0.8, label=f"Generated: {label}", zorder=5)

    _finish_axes(ax, paper_ids, title)
    _save_fig(fig, out_path)

def plot_sections(means_human, means_generated, out_path: Path, *, title: str = "", break_threshold: float = 5.0):
    """Section-level mean lines with optional broken y-axis for outliers."""
    sections = ["summary", "strengths", "weaknesses", "questions"]
    paper_ids = sorted(
        set().union(*(means_human.get(s, {}).keys() for s in sections))
        & set().union(*(means_generated.get(s, {}).keys() for s in sections))
    )
    if not paper_ids:
        print("No common papers for section plot. Skipping:", out_path)
        return

    colors = {"summary": "#0095ff", "strengths": "#00ff00", "weaknesses": "#ff8000", "questions": "#f700ff"}
    sec_label = lambda s: "Suggestions/Questions" if s == "questions" else s.title()
    xs = np.arange(len(paper_ids), dtype=float)

    all_vals = []
    for sec in sections:
        for pid in paper_ids:
            for src in (means_human, means_generated):
                v = src.get(sec, {}).get(pid)
                if v is not None and math.isfinite(float(v)):
                    all_vals.append(float(v))
    if not all_vals:
        return

    low = [v for v in all_vals if v < break_threshold]
    high = [v for v in all_vals if v >= break_threshold]
    use_broken = bool(high) and bool(low)
    w = max(12.0, min(28.0, 0.35 * max(1, len(paper_ids)) + 8.0))

    def _plot_lines(ax_main, ax_secondary=None):
        for sec in sections:
            base = colors.get(sec, "#333")
            dark = _darken(base)
            y_h = np.array([means_human.get(sec, {}).get(p, np.nan) for p in paper_ids])
            y_g = np.array([means_generated.get(sec, {}).get(p, np.nan) for p in paper_ids])
            ax_main.plot(xs, y_h, lw=2.0, color=base, alpha=0.85, label=f"Human - {sec_label(sec)}")
            ax_main.plot(xs, y_g, lw=2.5, color=dark, alpha=0.95, label=f"Generated - {sec_label(sec)}")
            if ax_secondary:
                ax_secondary.plot(xs, y_h, lw=2.0, color=base, alpha=0.85)
                ax_secondary.plot(xs, y_g, lw=2.5, color=dark, alpha=0.95)

    if not use_broken:
        fig, ax = plt.subplots(figsize=(w, 6.5))
        _plot_lines(ax)
        _finish_axes(ax, paper_ids, title, ylabel="Mean GranuScore (per section)")
        _save_fig(fig, out_path)
        return

    # Broken y-axis
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(w, 7.5),
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.05},
    )
    _plot_lines(ax_top, ax_bot)

    pad_lo = 0.08 * ((max(low) - min(low)) or 1.0)
    pad_hi = max(0.05, 0.08 * ((max(high) - min(high)) or 1.0))
    ax_bot.set_ylim(min(low) - pad_lo, max(low) + pad_lo)
    ax_top.set_ylim(min(high) - pad_hi, max(high) + pad_hi)

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False)
    ax_top.xaxis.tick_top()

    d = 0.008
    for a, y in [(ax_top, 0), (ax_bot, 1)]:
        kw = dict(transform=a.transAxes, color="k", clip_on=False)
        a.plot((-d, +d), (y - d, y + d), **kw)
        a.plot((1 - d, 1 + d), (y - d, y + d), **kw)

    fig.suptitle(title)
    ax_bot.set_xlabel("paper_id")
    ax_bot.set_ylabel("Mean GranuScore (per section)")
    ax_bot.set_xticks(range(len(paper_ids)))
    ax_bot.set_xticklabels(paper_ids, rotation=90, fontsize=6)
    ax_top.grid(axis="y", alpha=0.2)
    ax_bot.grid(axis="y", alpha=0.2)
    ax_top.legend(loc="upper right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_fig(fig, out_path)

###--- Helper functions
def _paper_axis(*scored_dicts):
    pids = sorted(set(v["paper_id"] for sd in scored_dicts for v in sd.values()))
    return pids, {p: i for i, p in enumerate(pids)}

def _paper_axis_from_all(combined):
    pids = sorted(set(v["paper_id"] for v in combined.values()))
    return pids, {p: i for i, p in enumerate(pids)}

def _gen_points(scored, paper_ids, pid_to_x):
    by_pid = {}
    for rid, item in scored.items():
        pid, y = item["paper_id"], _safe_float(item.get("granu_score"))
        if y is not None and math.isfinite(y) and pid not in by_pid:
            by_pid[pid] = (pid_to_x[pid], y, rid)
    return [by_pid[p] for p in paper_ids if p in by_pid]

def _human_points(scored, paper_ids, pid_to_x):
    by_pid: dict[str, list[float]] = {}
    points = []
    for rid, item in scored.items():
        pid, y = item["paper_id"], _safe_float(item.get("granu_score"))
        if y is None or not math.isfinite(y):
            continue
        by_pid.setdefault(pid, []).append(y)
        points.append((float(pid_to_x[pid]), y, rid))
    return points, by_pid

def _make_figure(n_papers):
    w = max(12.0, min(26.0, 0.35 * max(1, n_papers) + 8.0))
    return plt.subplots(figsize=(w, 6.0))

def _draw_human_band(ax, human_by_pid, pid_to_x, paper_ids):
    if not human_by_pid:
        return
    xs, med, q25, q75 = [], [], [], []
    for pid in paper_ids:
        scores = human_by_pid.get(pid, [])
        if not scores:
            continue
        arr = np.asarray(scores)
        xs.append(pid_to_x[pid])
        med.append(float(np.median(arr)))
        q25.append(float(np.percentile(arr, 25)))
        q75.append(float(np.percentile(arr, 75)))
    if xs:
        ax.fill_between(xs, q25, q75, color="grey", alpha=0.15, label="Human IQR (25–75%)", zorder=1)
        ax.plot(xs, med, ls="--", lw=1.5, color="grey", alpha=0.8, label="Human median", zorder=2)

def _draw_human_scatter(ax, points):
    if points:
        ax.scatter([p[0] for p in points], [p[1] for p in points],
                   marker="o", s=35, alpha=0.3, color="black", label="Human reviews", zorder=4)

def _finish_axes(ax, paper_ids, title, ylabel="GranuScore"):
    ax.set_xlabel("paper_id")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-0.5, len(paper_ids) - 0.5)
    ax.set_xticks(range(len(paper_ids)))
    ax.set_xticklabels(paper_ids, rotation=90, fontsize=6)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best", fontsize=9)

def _annotate(ax, points, annotate_ids, show_labels):
    annotate_ids = annotate_ids or set()
    if not show_labels and not annotate_ids:
        return
    all_y = [p[1] for p in points]
    yrange = (max(all_y) - min(all_y)) if all_y else 1.0
    if yrange == 0:
        yrange = 1.0
    for x, y, rid in points:
        if show_labels or rid in annotate_ids:
            ax.text(x, y + 0.02 * yrange, rid, ha="center", va="bottom",
                    rotation=90, fontsize=6, zorder=6)
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Main -------------------------------------------------------------------------------------------------------------#####
def run_comparison(generated_textdicts: dict[str, TextDict], human_textdict: TextDict, output_dir: Path, *, limit: int | None = None, filter_common: bool = True, do_section_analysis: bool = False, annotated_ids: list[str] | None = None, show_labels: bool = False):
    """
    generated_textdicts: {label: textdict}  — one entry = single reviewer, many = multi.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(generated_textdicts.keys())
    is_multi = len(labels) > 1
    annotate_set = set(annotated_ids or [])
    limit_tag = f"_LIMIT{limit}" if limit is not None else ""

    # Filter to common papers
    if filter_common:
        all_dicts = [generated_textdicts[l] for l in labels] + [human_textdict]
        filtered, common = filter_to_common_papers(*all_dicts)
        for i, l in enumerate(labels):
            generated_textdicts[l] = filtered[i]
        human_textdict = filtered[-1]

    # Apply limit
    human_textdict = apply_limit(human_textdict, limit)
    for l in labels:
        generated_textdicts[l] = apply_limit(generated_textdicts[l], limit)

    # Score
    print("Scoring human reviews...")
    scored_human = score_reviews(human_textdict, detailed=True)
    avg_hum, n_hum = mean_granu_score(scored_human)

    scored_by_label: dict[str, ScoredDict] = {}
    avg_by_label: dict[str, tuple[float | None, int]] = {}
    for l in labels:
        print(f"Scoring generated: {l}")
        scored_by_label[l] = score_reviews(generated_textdicts[l], detailed=True)
        avg_by_label[l] = mean_granu_score(scored_by_label[l])

    # Stats
    gen_means = [m for m, _ in avg_by_label.values() if m is not None]
    avg_gen = sum(gen_means) / len(gen_means) if gen_means else None

    stats = {
        "avg_generated": avg_gen,
        "avg_human": avg_hum,
        "avg_diff_generated_minus_human": (
            float(avg_gen - avg_hum) if avg_gen is not None and avg_hum is not None else None
        ),
        "n_human_used_for_avg": n_hum,
        "avg_per_reviewer": {l: {"avg": v[0], "n": v[1]} for l, v in avg_by_label.items()},
    }

    # Save JSONs
    save_json(
        _wrap_summary(scored_human, avg_generated=avg_gen, avg_human=avg_hum,
                      n_generated=sum(v[1] for v in avg_by_label.values()),
                      n_human=n_hum, group="human"),
        output_dir / f"scored_human{limit_tag}.json",
    )
    for l in labels:
        avg_l, n_l = avg_by_label[l]
        save_json(
            _wrap_summary(scored_by_label[l], avg_generated=avg_l, avg_human=avg_hum,
                          n_generated=n_l, n_human=n_hum, group="generated"),
            output_dir / f"scored_generated_{l}{limit_tag}.json",
        )
    save_json(stats, output_dir / "stats.json")

    # Per-paper figure
    fig_title = f"GranuScore per paper{f' | LIMIT={limit}' if limit else ''}"
    if is_multi:
        plot_multi(scored_by_label, scored_human,
                   output_dir / f"granuscore_per_paper{limit_tag}.png",
                   title=fig_title, annotate_ids=annotate_set, show_labels=show_labels)
    else:
        plot_single(scored_by_label[labels[0]], scored_human,
                    output_dir / f"granuscore_per_paper{limit_tag}.png",
                    title=fig_title, annotate_ids=annotate_set, show_labels=show_labels)

    # First-20-papers variant
    all_pids = sorted(set(v["paper_id"] for sd in [scored_human, *scored_by_label.values()] for v in sd.values()))
    first20_pids = set(all_pids[:20])
    scored_human_20 = {k: v for k, v in scored_human.items() if v["paper_id"] in first20_pids}
    scored_by_label_20 = {l: {k: v for k, v in sd.items() if v["paper_id"] in first20_pids} for l, sd in scored_by_label.items()}
    fig_title_20 = fig_title + " (first 20)"
    if is_multi:
        plot_multi(scored_by_label_20, scored_human_20,
                   output_dir / f"granuscore_per_paper{limit_tag}_first20.png",
                   title=fig_title_20, annotate_ids=annotate_set, show_labels=show_labels)
    else:
        plot_single(scored_by_label_20[labels[0]], scored_human_20,
                    output_dir / f"granuscore_per_paper{limit_tag}_first20.png",
                    title=fig_title_20, annotate_ids=annotate_set, show_labels=show_labels)

    # Section analysis
    if do_section_analysis:
        print("Running section analysis...")
        means_hum = score_sections(human_textdict)
        means_gen = score_sections(generated_textdicts[labels[0]])
        plot_sections(
            means_hum, means_gen,
            output_dir / f"granuscore_sections{limit_tag}.png",
            title=f"Section means ({labels[0]}){f' | LIMIT={limit}' if limit else ''}",
        )

    print(f"Done. Saved in {output_dir}")
#####------------------------------------------------------------------------------------------------------------------------#####