"""
Non-LLM baseline matchers for the thesis comparison subsection. No API, runs locally.

Matches human vs AI concerns using lexical (token Jaccard) and semantic (cosine)
similarity, then compares coverage and set agreement against the LLM matcher.

    python baseline_overlap.py
    python baseline_overlap.py --tau-lexical 0.25 --tau-semantic 0.55
    pip install sentence-transformers   # to enable semantic
"""

import argparse
from pathlib import Path
import re

import pandas as pd
from match_concerns import KIND_AI, MATCH_CHANNELS
from score_overlap import C_BASELINE, MATCHES_ROOT, OUT_ROOT, load_match_files

_TOKEN_RE = re.compile(r"[a-z0-9]+")


## similarity functions

def _tokens(text):
    return set(_TOKEN_RE.findall((text or "").lower()))


def token_jaccard(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class SemanticMatcher:
    """Lazy sentence-transformers cosine matcher; no-op if package missing."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.available = False
        self._model = None
        self._model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self._SentenceTransformer = SentenceTransformer
            self.available = True
        except Exception:
            self._SentenceTransformer = None

    def _ensure(self):
        if self._model is None:
            self._model = self._SentenceTransformer(self._model_name)

    def cosine_matrix(self, texts_a, texts_b):
        """Return an (|a| x |b|) cosine-similarity matrix, or None if unavailable."""
        if not self.available:
            return None
        from sentence_transformers import util
        self._ensure()
        ea = self._model.encode(texts_a, convert_to_tensor=True, normalize_embeddings=True)
        eb = self._model.encode(texts_b, convert_to_tensor=True, normalize_embeddings=True)
        return util.cos_sim(ea, eb).cpu().numpy()


## coverage helpers

def _matchable(items):
    return [it for it in items if it.get("channel") in MATCH_CHANNELS]


def coverage_lexical(ai_items, human_items, tau):
    A, H = _matchable(ai_items), _matchable(human_items)
    if not H:
        return None, set()
    matched = set()
    for h in H:
        for a in A:
            if a.get("channel") != h.get("channel"):
                continue
            if token_jaccard(a.get("text", ""), h.get("text", "")) >= tau:
                matched.add(h["number"])
                break
    return len(matched) / len(H), matched


def coverage_semantic(ai_items, human_items, tau, matcher):
    A, H = _matchable(ai_items), _matchable(human_items)
    if not H or not A:
        return (None, set()) if not H else (0.0, set())
    sim = matcher.cosine_matrix([a.get("text", "") for a in A],
                                [h.get("text", "") for h in H])
    if sim is None:
        return None, set()
    matched = set()
    for j, h in enumerate(H):
        for i, a in enumerate(A):
            if a.get("channel") != h.get("channel"):
                continue
            if sim[i, j] >= tau:
                matched.add(h["number"])
                break
    return len(matched) / len(H), matched


def _llm_matched_humans(match):
    """LLM coverage and matched human numbers from the stage-2 file."""
    H = [it for it in match["human_items"] if it.get("channel") in C_BASELINE]
    matched = {m["human_number"] for m in match["matches"]}
    matched_base = {n for n in matched if any(h["number"] == n for h in H)}
    cov = len(matched_base) / len(H) if H else None
    return cov, matched_base


def _jaccard_sets(a, b):
    if not a and not b:
        return None
    return len(a & b) / len(a | b) if (a | b) else None


## driver

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tau-lexical", type=float, default=0.20)
    ap.add_argument("--tau-semantic", type=float, default=0.50)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    matches = [m for m in load_match_files() if m.get("comparison_kind", KIND_AI) == KIND_AI]
    if not matches:
        print(f"no match files under {MATCHES_ROOT}, run run_full_matching.py first")
        return

    sem = SemanticMatcher(args.model)
    sem_status = f"on ({args.model})" if sem.available else "off (pip install sentence-transformers)"
    print(f"loaded {len(matches)} match files, semantic: {sem_status}")

    rows = []
    for m in matches:
        llm_cov, llm_set = _llm_matched_humans(m)
        lex_cov, lex_set = coverage_lexical(m["ai_items"], m["human_items"], args.tau_lexical)
        row = {
            "dataset": m["dataset"], "paper_id": m["paper_id"], "source": m["source"],
            "llm_coverage": llm_cov,
            "lexical_coverage": lex_cov,
            "lexical_vs_llm_jaccard": _jaccard_sets(lex_set, llm_set),
        }
        if sem.available:
            sem_cov, sem_set = coverage_semantic(m["ai_items"], m["human_items"],
                                                 args.tau_semantic, sem)
            row["semantic_coverage"] = sem_cov
            row["semantic_vs_llm_jaccard"] = _jaccard_sets(sem_set, llm_set)
        rows.append(row)

    df = pd.DataFrame(rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_ROOT / "baseline_per_pair.csv", index=False)

    lines = ["# Baseline (non-LLM) matchers vs the LLM matcher\n",
             f"tau_lexical={args.tau_lexical}, tau_semantic={args.tau_semantic}, semantic={sem_status}\n",
             f"Pairs: {len(df)}\n",
             "| method | mean coverage | corr(LLM) | mean Jaccard(LLM) |",
             "| --- | --- | --- | --- |"]

    def _corr(col):
        sub = df[["llm_coverage", col]].dropna()
        return sub["llm_coverage"].corr(sub[col]) if len(sub) > 1 else float("nan")

    print("\ncoverage by method:")
    print(f"  llm:      {df['llm_coverage'].mean():.3f}")
    lines.append(f"| llm | {df['llm_coverage'].mean():.3f} | 1.000 | 1.000 |")
    for method, cov_col, jac_col in [
        ("lexical",  "lexical_coverage",  "lexical_vs_llm_jaccard"),
        ("semantic", "semantic_coverage", "semantic_vs_llm_jaccard"),
    ]:
        if cov_col not in df:
            continue
        mc, cr, jc = df[cov_col].mean(), _corr(cov_col), df[jac_col].mean()
        print(f"  {method}\t{mc:.3f}  r={cr:.3f}  J={jc:.3f}")
        lines.append(f"| {method} | {mc:.3f} | {cr:.3f} | {jc:.3f} |")

    (OUT_ROOT / "baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote baseline_per_pair.csv and baseline_summary.md to {OUT_ROOT}")


if __name__ == "__main__":
    main()
