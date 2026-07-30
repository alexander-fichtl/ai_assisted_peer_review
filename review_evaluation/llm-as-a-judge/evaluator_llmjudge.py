"""
Library for LLM-as-a-judge review evaluation.
1. Runner loads reviews into a flat list: [{paper_id, review_id, is_human, source, text}]
2. Runner calls judge_reviews() with that list + papers directory + output directory
3. This lib loads each paper (.md), sends paper+review to GPT, writes CSV + JSON results
"""


#####----- Imports --------------------------------------------------------------------------------------#####
from __future__ import annotations
import os
import json
import csv
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from openai import OpenAI
#####----------------------------------------------------------------------------------------------------#####



#####----- OpenAI ---------------------------------------------------------------------------------------#####
def _client_singleton() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable.")
    return OpenAI(api_key=api_key)
#####----------------------------------------------------------------------------------------------------#####



#####----- Prompts & Schema -----------------------------------------------------------------------------#####
SYSTEM_INSTRUCTIONS = """
You will be given a PAPER and a REVIEW of that paper.
Rate the QUALITY OF THE REVIEW only (do NOT rate the paper). 
Do not be generic. Be review-specific and precise. Be critical if appropriate. 
Discriminate clearly between strong and weak reviews according to human preference. Avoid clustering most scores around the middle.

Agree or disagree with the following statements (5-item Likert):

Understanding: The review demonstrates an adequate understanding of the paper.
- Review makes comments that are detailed and specific to the paper
- It is OK for the reviewer to lack expertise in certain aspects of the paper as long as it is explicitly or implicitly indicated in the review

Coverage: The review covers all the required aspects.
- Review adequately comments on Soundness, Presentation, and Contribution of the paper
- Review adequately comments on Strengths and Weaknesses of the paper

Evidence Support: Evaluations made in the review are well supported.
- Objective arguments are grounded in the paper's content (e.g., specific results/comparisons) and are correct
- Subjective arguments are accompanied with reasoning
- A review that brings additional useful information (e.g., counter examples or uncited references which do a part of the claimed work) is especially strong

Constructiveness: The review provides constructive feedback to authors.
- Whenever possible, critical comments (especially subjective) are accompanied with actionable items on how to improve the paper
- Review is unbiased and written in a polite manner

Conciseness: The review communicates efficiently without sacrificing substance.
- Clear, direct writing with minimal repetition or digressions
- Focuses on paper-relevant points; avoids unnecessary verbosity

Provide an overall score for the quality of the review:
- 1 = Very low E.g., a generic review that is applicable to any paper / a short and dismissive review.
- 2 = Low E.g., a review with serious flaws on multiple aspects.
- 3 = Fair E.g., a review with serious flaws on one aspect / a review without serious flaws but with limited insights for Authors/Area Chairs.
- 4 = Good E.g., acceptable review, but nothing stands out. Moderately helpful for decision-making.
- 5 = Very good E.g., a helpful review that stands out on some aspects and provides useful insights.
- 6 = Excellent E.g., a very insightful review that stands out on all aspects.
- 7 = Exceptional E.g., an excellent review that helps authors to non-trivially improve the paper / brings a unique piece of information that is crucial for the decision.

Also produce an 'open_feedback' field: 1-5 sentences of free-form feedback to reviewers about the review's quality and how to improve it.
"""

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["version", "overall_score", "subscores", "rationales", "confidence", "open_feedback"],
    "properties": {
        "version": {"type": "string", "const": "review-judge-1.1"},
        "overall_score": {"type": "integer", "minimum": 1, "maximum": 7},
        "subscores": {
            "type": "object",
            "required": ["understanding", "coverage", "evidence_support", "conciseness", "constructiveness"],
            "properties": {
                "understanding":     {"type": "integer", "minimum": 1, "maximum": 5},
                "coverage":          {"type": "integer", "minimum": 1, "maximum": 5},
                "evidence_support":  {"type": "integer", "minimum": 1, "maximum": 5},
                "constructiveness":  {"type": "integer", "minimum": 1, "maximum": 5},
                "conciseness":       {"type": "integer", "minimum": 1, "maximum": 5},
            },
        },
        "rationales": {
            "type": "object",
            "required": ["understanding", "coverage", "evidence_support", "constructiveness", "conciseness", "overall"],
            "properties": {
                "understanding":     {"type": "string", "maxLength": 2000},
                "coverage":          {"type": "string", "maxLength": 2000},
                "evidence_support":  {"type": "string", "maxLength": 2000},
                "constructiveness":  {"type": "string", "maxLength": 2000},
                "conciseness":       {"type": "string", "maxLength": 2000},
                "overall":           {"type": "string", "maxLength": 2000},
            },
        },
        "confidence": {"type": "string", "enum": ["Low", "Medium", "High"]},
        "open_feedback": {"type": "string", "minLength": 50, "maxLength": 2000},
    },
}

TOOL_DEF = [{
    "type": "function",
    "function": {
        "name": "emit_judgement",
        "description": "Return a structured judgement of the REVIEW against the PAPER using the rubric.",
        "parameters": SCHEMA,
    },
}]

FEWSHOT_EXAMPLES = [
    {
        "paper": (
            "This paper reevaluates seven years of algorithmic fairness methods on tabular benchmarks. "
            "Using a unified 'unprocessing' framework, it compares thousands of model-constraint "
            "configurations and finds that a simple postprocessing baseline essentially recovers the "
            "fairness-accuracy Pareto frontier achieved by more complex approaches."
        ),
        "review": (
            "Summary: There have been many proposals in the recent literature to train fair ML models. "
            "This paper evaluates thousands of such models, and finds that a simple postprocessing "
            "technique achieves the fairness-accuracy Pareto frontier.\n\n"
            "Soundness: 4 excellent\nPresentation: 4 excellent\nContribution: 4 excellent\n\n"
            "Strengths:\nThis type of comprehensive benchmarking of thousands of models adds a ton of "
            "value to the algorithmic fairness literature. I think the result that a simple "
            "postprocessing step achieves the Pareto frontier is very significant. I applaud the "
            "authors for taking on this task.\n\nWeaknesses:\nNone\n\nQuestions:\nNone\n\n"
            "Rating: 8: accept, good paper\nConfidence: 3"
        ),
        "arguments": {
            "version": "review-judge-1.1",
            "overall_score": 1,
            "subscores": {"understanding": 2, "coverage": 1, "evidence_support": 1, "constructiveness": 1, "conciseness": 3},
            "rationales": {
                "understanding": "The reviewer paraphrases the main headline but does not engage with the unprocessing framework, methodological checks, or detailed empirical findings.",
                "coverage": "No substantive discussion of soundness, limitations, or comparative insights.",
                "evidence_support": "Positive claims are not supported by references to specific experiments, datasets, or figures.",
                "constructiveness": "No suggestions or concerns; provides no guidance to authors or program chairs.",
                "conciseness": "Short and readable, but primarily due to missing substance rather than efficient communication.",
                "overall": "A very short, weak, generic review whose enthusiasm is not justified by concrete analysis; minimally useful for decision-making.",
            },
            "confidence": "High",
            "open_feedback": "This review is overly short and generic. Ground your recommendation in concrete aspects of the methodology and results, and discuss both strengths and potential limitations.",
        },
    },
    {
        "paper": (
            "The paper 'Closing the Curious Case of Neural Text Degeneration' provides a theoretical explanation "
            "for truncation-based decoding heuristics by linking them to the softmax bottleneck, and introduces a "
            "basis-aware truncation-style method that better aligns sampling with the true support distribution."
        ),
        "review": (
            "Summary: This paper aims to give a formal justification for why truncation based sampling approaches "
            "work well in language generation. They link this phenomenon to the softmax bottleneck.\n\n"
            "Soundness: 3 good\nPresentation: 4 excellent\nContribution: 3 good\n\n"
            "Strengths:\nThe work offers a theoretical explanation for why certain ad-hoc methods used during "
            "language generator decoding work well. This is a valuable insight to the NLG community.\n"
            "The work then develops a sampling algorithm based on this theoretical explanation.\n\n"
            "Weaknesses:\nThe theoretical portion is at times difficult to understand due to notational choices. "
            "The method does not appear to have empirical performance benefits and is computationally expensive. "
            "Figure 7 only consists of 3 points, which hardly feels like enough evidence.\n\n"
            "Questions:\nSince the matrix W is static, can (3) not be solved ahead of time? "
            "Why is typical sampling omitted from the discussion? "
            "In theorem 1, these factors are the collective probability over/underestimation across all tokens, right? "
            "I don't think that the softmax function satisfies the additivity property required of a linear map. "
            "Can these principles explain the degeneracy during greedy decoding? "
            "How do these results align with Meister et al. 2023?\n\n"
            "Rating: 8: accept, good paper\nConfidence: 4"
        ),
        "arguments": {
            "version": "review-judge-1.1",
            "overall_score": 7,
            "subscores": {"understanding": 5, "coverage": 5, "evidence_support": 5, "constructiveness": 5, "conciseness": 5},
            "rationales": {
                "understanding": "Demonstrates deep understanding of the theoretical motivation, softmax bottleneck, and proposed method.",
                "coverage": "Systematically addresses relevant parts.",
                "evidence_support": "Points are grounded in specific notation, theorems, figures, and related work.",
                "constructiveness": "Provides clear, actionable suggestions to clarify theory, terminology, and experiments.",
                "conciseness": "Detailed but focused; very little gratuitous verbosity.",
                "overall": "An excellent, insightful review that strongly supports authors and area chairs.",
            },
            "confidence": "High",
            "open_feedback": "This review is technically sharp and highly constructive. Keep structuring feedback around precise references to the paper and proposing concrete fixes.",
        },
    },
]
#####----------------------------------------------------------------------------------------------------#####



#####----- Helper functions -----------------------------------------------------------------------------#####
def _load_generated_review_text(paper_dir: Path, paper_id: str) -> str | None:
    """Try final_review.txt, then <paper_id>.txt in a paper directory."""
    for name in ["final_review.txt", f"{paper_id}.txt"]:
        p = paper_dir / name
        if p.exists():
            text = p.read_text(encoding="utf-8").strip()
            if text:
                return text
    return None

def _is_meta(note: dict) -> bool:
    c = note.get("content") or {}
    if "metareview" in c:
        return True

    inv = (note.get("invitation") or "").lower()
    inv_norm = (note.get("_normalized_invitation") or "").lower()
    invs = " ".join(note.get("invitations") or []).lower()
    title = str(c.get("title") or "").lower()

    meta_tokens = ("meta_review", "metareview", "area_chair")
    if any(t in inv for t in meta_tokens):
        return True
    if any(t in inv_norm for t in meta_tokens):
        return True
    if any(t in invs for t in meta_tokens):
        return True
    if "meta-review" in title or "metareview" in title:
        return True
    return False

def _get_reviewer_code(hr: dict, fallback: str) -> str:
    for key in ("signatures", "writers", "readers"):
        vals = hr.get(key) or []
        if isinstance(vals, str):
            vals = [vals]
        for v in vals:
            if isinstance(v, str) and "Reviewer_" in v:
                code = v.split("Reviewer_", 1)[-1].split("/", 1)[0].strip()
                if code:
                    return code
    return fallback

def _openreview_to_text(review_obj: dict) -> str:
    c = (review_obj or {}).get("content") or {}
    parts = []

    def add(label, key):
        v = c.get(key)
        if isinstance(v, dict):
            v = v.get("value")
        if v is not None and str(v).strip():
            parts.append(f"{label}:\n{v}")

    add("Summary", "summary")
    add("Strengths", "strengths")
    add("Weaknesses", "weaknesses")
    add("Suggestions/Questions", "questions")

    for label, key in [
        ("Soundness", "soundness"), ("Presentation", "presentation"),
        ("Contribution", "contribution"), ("Overall", "rating"),
        ("Confidence", "confidence"), ("Conduct", "code_of_conduct"),
    ]:
        v = c.get(key)
        if isinstance(v, dict):
            v = v.get("value")
        if v is not None and str(v).strip():
            parts.append(f"{label}: {v}")

    v = c.get("flag_for_ethics_review")
    if isinstance(v, dict):
        v = v.get("value")
    if isinstance(v, list):
        v = ", ".join(str(x) for x in v if str(x).strip())
    if v is not None and str(v).strip():
        parts.append(f"Ethics: {v}")

    return "\n\n".join(parts)

def _build_fewshot_messages() -> list[dict]:
    msgs = []
    for i, ex in enumerate(FEWSHOT_EXAMPLES):
        tool_call_id = f"fewshot_{i+1}"
        args_json = json.dumps(ex["arguments"], ensure_ascii=False)

        msgs.append({
            "role": "user",
            "content": f"PAPER:\n{ex['paper'].strip()}\n\nREVIEW:\n{ex['review'].strip()}",
        })
        msgs.append({
            "role": "assistant",
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {"name": "emit_judgement", "arguments": args_json},
            }],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "emit_judgement",
            "content": args_json,
        })
    return msgs
#####----------------------------------------------------------------------------------------------------#####



#####----- Data loading ---------------------------------------------------------------------------------#####
def load_paper_md(paper_id: str, papers_dir: Path) -> str:
    """Load paper from papers_dir/<paper_id>/<paper_id>.md"""
    md_path = papers_dir / paper_id / f"{paper_id}.md"
    if not md_path.exists():
        raise FileNotFoundError(f"Paper .md not found: {md_path}")
    return md_path.read_text(encoding="utf-8")

def load_reviews_from_iclr_csv(csv_path: Path, generated_dirs: dict[str, Path] | None = None) -> list[dict]:
    """
    Load reviews from an ICLR dataset.csv.

    Human reviews: from the 'reviews' column (JSON array of OpenReview objects).
    Generated reviews: from generated_dirs {model_name: path_to_model_dir}.
      Each dir has <paper_id>/final_review.txt or <paper_id>.txt or final_review_openreview.json.

    Returns flat list of {paper_id, review_id, is_human, source, text}.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    reviews = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            paper_id = (row.get("id") or row.get("paper_id") or "").strip()
            if not paper_id:
                continue

            # Human reviews from OpenReview JSON
            for i, hr in enumerate(json.loads(row.get("reviews") or "[]")):
                if _is_meta(hr):
                    continue
                if str(hr.get("invalid", "")).strip().lower() in {"yes", "y", "true", "1"}:
                    continue
                rid = _get_reviewer_code(hr, fallback=str(i + 1))
                text = _openreview_to_text(hr)
                if text.strip():
                    reviews.append({
                        "paper_id": paper_id,
                        "review_id": f"{paper_id}_human_{rid}",
                        "is_human": True,
                        "source": "human",
                        "text": text,
                    })

    # Generated reviews from dirs
    if generated_dirs:
        for model_name, model_dir in generated_dirs.items():
            if not model_dir.exists():
                print(f"Warning: generated dir not found: {model_dir}")
                continue
            # Get paper IDs we have human reviews for
            paper_ids = {r["paper_id"] for r in reviews}
            for pid in paper_ids:
                text = _load_generated_review_text(model_dir / pid, pid)
                if text:
                    reviews.append({
                        "paper_id": pid,
                        "review_id": f"{pid}_{model_name}",
                        "is_human": False,
                        "source": model_name,
                        "text": text,
                    })

    return reviews

def load_reviews_for_naturecom(
    human_dir: Path,
    generated_dirs: dict[str, Path],
) -> list[dict]:
    """
    Load NatureCom reviews.

    Human: human_dir/reviews_<paper_id>.json
    Generated: generated_dirs[model_name]/<paper_id>/final_review.txt or <paper_id>.txt

    Returns flat list of {paper_id, review_id, is_human, source, text}.
    """
    reviews = []

    # Human reviews
    if human_dir.exists():
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
                reviews.append({
                    "paper_id": pid,
                    "review_id": f"{pid}_human_{reviewer_id}",
                    "is_human": True,
                    "source": "human",
                    "text": text,
                })

    # Generated reviews
    for model_name, model_dir in generated_dirs.items():
        if not model_dir.exists():
            print(f"Warning: generated dir not found: {model_dir}")
            continue
        for paper_dir in sorted(model_dir.iterdir()):
            if not paper_dir.is_dir():
                continue
            pid = paper_dir.name
            text = _load_generated_review_text(paper_dir, pid)
            if text:
                reviews.append({
                    "paper_id": pid,
                    "review_id": f"{pid}_{model_name}",
                    "is_human": False,
                    "source": model_name,
                    "text": text,
                })

    return reviews
#####----------------------------------------------------------------------------------------------------#####



#####----- Functions ------------------------------------------------------------------------------------#####
def evaluate_review(paper_text: str, review_text: str, *, model: str = "gpt-5", fewshot: bool = False) -> dict:
    """Send paper + review to GPT and get structured judgement back."""
    client = _client_singleton()
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]

    if fewshot:
        messages += _build_fewshot_messages()

    messages += [
        {"role": "user", "content": f"PAPER:\n{paper_text}"},
        {"role": "user", "content": f"REVIEW:\n{review_text}"},
    ]

    chat = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOL_DEF,
        tool_choice={"type": "function", "function": {"name": "emit_judgement"}},
    )

    args_str = chat.choices[0].message.tool_calls[0].function.arguments
    return json.loads(args_str)

def judge_reviews(reviews: list[dict], papers_dir: Path, output_dir: Path, *, model: str = "gpt-5", fewshot: bool = False, skip_existing: bool = True) -> None:
    """
    Judge all reviews and write results.

    reviews:      [{paper_id, review_id, is_human, source, text}]
    papers_dir:   directory containing <paper_id>/<paper_id>.md
    output_dir:   where to write judgements.csv + judgements/<paper_id>/<review_id>.json
    skip_existing: if True, skip reviews that already have a JSON in judgements/
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = output_dir / "judgements"
    json_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "judgements.csv"
    fieldnames = [
        "review_id", "paper_id", "human review", "source", "review",
        "overall score", "understanding score", "understanding rationale",
        "coverage score", "coverage rationale", "evidence_support score",
        "evidence support rationale", "constructiveness score",
        "constructiveness rationale", "conciseness score", "conciseness rationale",
        "confidence", "open_feedback",
    ]

    # Collect existing results (from CSV — so resume skips only what this run already wrote)
    existing_rows: dict[str, dict] = {}  # review_id -> csv row
    if skip_existing and csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rid = row.get("review_id", "").strip()
                if rid:
                    existing_rows[rid] = row
        if existing_rows:
            print(f"Found {len(existing_rows)} existing judgements in CSV, will skip those.")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        #Write existing results back to CSV
        for row in existing_rows.values():
            writer.writerow({fn: row.get(fn, "") for fn in fieldnames})

        #Judge new reviews
        to_judge = [r for r in reviews if r["review_id"] not in existing_rows]
        print(f"Judging {len(to_judge)} new reviews ({len(reviews) - len(to_judge)} skipped).")

        for i, rev in enumerate(to_judge):
            pid = rev["paper_id"]
            rid = rev["review_id"]
            print(f"  [{i+1}/{len(to_judge)}] {rid}")

            try:
                paper_text = load_paper_md(pid, papers_dir)
            except FileNotFoundError as e:
                print(f"    Skipping: {e}")
                continue

            try:
                res = evaluate_review(paper_text, rev["text"], model=model, fewshot=fewshot)
            except Exception as e:
                print(f"    Failed: {e}")
                continue

            s, r = res.get("subscores") or {}, res.get("rationales") or {}
            writer.writerow({
                "review_id": rid,
                "paper_id": pid,
                "human review": "yes" if rev["is_human"] else "no",
                "source": rev["source"],
                "review": rev["text"],
                "overall score": res.get("overall_score", ""),
                "understanding score": s.get("understanding", ""),
                "understanding rationale": r.get("understanding", ""),
                "coverage score": s.get("coverage", ""),
                "coverage rationale": r.get("coverage", ""),
                "evidence_support score": s.get("evidence_support", ""),
                "evidence support rationale": r.get("evidence_support", ""),
                "constructiveness score": s.get("constructiveness", ""),
                "constructiveness rationale": r.get("constructiveness", ""),
                "conciseness score": s.get("conciseness", ""),
                "conciseness rationale": r.get("conciseness", ""),
                "confidence": res.get("confidence", ""),
                "open_feedback": res.get("open_feedback", ""),
            })

            #Per-review JSON
            rev_dir = json_dir / pid
            rev_dir.mkdir(parents=True, exist_ok=True)
            (rev_dir / f"{rid}.json").write_text(
                json.dumps({
                    "paper_id": pid,
                    "review_id": rid,
                    "is_human": rev["is_human"],
                    "source": rev["source"],
                    "review_text": rev["text"],
                    "judgement": res,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(f"[DONE] Wrote {csv_path}")
#####----------------------------------------------------------------------------------------------------#####