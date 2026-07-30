"""
Concern extraction for AI-vs-human review overlap measurement (stage 1 of 3).

Decomposes one peer review into a structured list of items, each classified by
channel (critique / strength / neutral) and the seven-field concern schema.

Uses OpenAI Structured Outputs (strict json_schema) so the model returns JSON
matching the schema directly. Loads OPENAI_API_KEY from overlap_pipeline/.env.

    from extract_concerns import extract_concerns, save_extraction
    result = extract_concerns(review_text, review_id="...", paper_id="...")
    save_extraction(result, Path("outputs/.../human.json"))
"""

import json
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from pathlib import Path
from openai import OpenAI

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")


## configuration

MODEL_NAME = "gpt-5.5"


## taxonomies

CHANNELS = ["critique", "neutral", "strength"]

CONCERN_TYPES = [
    "Experimental design",
    "Methodology & soundness",
    "Data quality & validation",
    "Reproducibility",
    "Related work",
    "Clarity & writing",
    "Significance & contribution",
    "Ethics & broader impact",
    "Other",
]

RECOMMENDATION_CATEGORIES = [
    "New experiment",
    "New analysis",
    "Textual revision",
    "Clarification",
    "Additional citations",
    "Methodology change",
    "Data",
    "None",
]

SERIOUSNESS_CATEGORIES = [
    "Essential",
    "Recommended",
    "Minor",
]


## json schema for structured outputs
# strict mode: additionalProperties=false, all props in required, nullable fields use ["type","null"]

EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "number",
                    "channel",
                    "text",
                    "type",
                    "recommendation",
                    "recommendation_category",
                    "seriousness_explanation",
                    "seriousness_category",
                ],
                "properties": {
                    "number": {"type": "integer"},
                    "channel": {"type": "string", "enum": CHANNELS},
                    "text": {"type": "string"},
                    "type": {"type": "string", "enum": CONCERN_TYPES},
                    "recommendation": {"type": ["string", "null"]},
                    "recommendation_category": {
                        "type": ["string", "null"],
                        "enum": RECOMMENDATION_CATEGORIES + [None],
                    },
                    "seriousness_explanation": {"type": ["string", "null"]},
                    "seriousness_category": {
                        "type": ["string", "null"],
                        "enum": SERIOUSNESS_CATEGORIES + [None],
                    },
                },
            },
        },
    },
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "concern_extraction",
        "strict": True,
        "schema": EXTRACTION_SCHEMA,
    },
}


## system prompt

SYSTEM_PROMPT = """You are an expert analyst of academic peer reviews. Your task is to decompose a single peer review into a structured list of individual concerns and recommendations and classify each one.

INPUT
You will be given the text of one peer review of an academic paper. The review may be structured (with sections like Summary, Strengths, Weaknesses, Questions) or free-form prose. The paper itself is NOT provided. You are decomposing what the reviewer wrote, not fact-checking it against the paper.

OUTPUT
Reply with a JSON object that matches the provided schema. The object has a single field "items" containing a list of decomposed items.

CHANNEL CLASSIFICATION (every item gets exactly one)
- "critique": the reviewer identifies a problem, weakness, missing piece, error, methodological doubt, or thing the author should fix. Includes questions that imply a gap or a concern (e.g., "How does this handle X?" when X is a plausible failure mode).
- "strength": the reviewer praises something the paper does well (clear writing, strong results, interesting idea, sound methodology, etc.).
- "neutral": neither praise nor critique. Use this for a descriptive or factual observation without judgement that appears OUTSIDE the summary (for example a neutral restatement embedded in Weaknesses or Questions, or an informational question that does not imply a gap). Do NOT emit neutral items for the review's Summary section, which is skipped entirely (see the summary rule below). Editorial score lines like "Soundness: 4" are skipped entirely.

DECOMPOSITION RULES
1. SKIP THE SUMMARY ENTIRELY. Do not emit any item that comes from the review's summary. For structured reviews this is the field labelled "Summary"; for narrative journal reviews it is an opening paragraph that merely restates or paraphrases the paper without evaluation. By venue guidelines a summary should be description only, so it contributes no concerns, recommendations, or strengths. Exception: if a summary paragraph contains an explicit evaluative judgement (praise or critique), emit only that evaluative point and ignore the descriptive remainder.
2. Each item makes EXACTLY ONE point. Do not bundle two distinct concerns or two distinct strengths into one item.
3. Each item must be understandable on its own. Spell out implicit references (e.g., replace "this issue" with the specific issue).
4. Extract only what the reviewer actually wrote. Do not invent points the reviewer did not raise. Do not soften or sharpen the framing.
5. If a single sentence contains multiple distinct points, split it into multiple items.
6. If multiple sentences make the same point with different wording, merge them into one item with a single combined formulation.
7. Skip boilerplate: greetings, signoffs, conference-system scaffolding, numerical score lines (e.g., "Soundness: 4", "Confidence: 3", "Rating: 6").
8. Keep the item text concise (one to three sentences). The verbatim quote is not required; a faithful paraphrase is fine as long as the point is preserved.

TYPE CLASSIFICATION (every item gets exactly one)
- Experimental design: experiments, baselines, ablations, controls, hyperparameter sweeps, evaluation setup
- Methodology & soundness: correctness of the proposed method, theorems, proofs, claims, logical structure, mathematical rigor
- Data quality & validation: dataset choice, splits, biases, leakage, label noise, sample size
- Reproducibility: code or data availability, hyperparameter reporting, deterministic seeding, environment details
- Related work: coverage of cited work, accuracy of citations, missing references, mischaracterization of prior work
- Clarity & writing: readability, structure, notation, figure quality, English/grammar
- Significance & contribution: novelty, impact, scope, relevance to the venue, generality of the contribution
- Ethics & broader impact: ethical concerns, dual-use risk, fairness, environmental cost, societal harm
- Other: anything that does not fit a category above (use sparingly)

RECOMMENDATION FIELDS (only for channel="critique"; set to null for strength and neutral)
- recommendation (free text or null): the concrete action the reviewer EXPLICITLY recommends the authors take. Use the reviewer's stated action, paraphrased only for self-containment, in one or two sentences. If the reviewer flagged a problem without proposing an action, set this to null. Do NOT infer or invent recommendations; do not fill this field just because a fix seems obvious.
- recommendation_category (one of the listed options, OR null): pick a category only when `recommendation` is non-null. If `recommendation` is null, set this to null as well. Categories:
  - New experiment: run a new experiment or ablation
  - New analysis: re-analyse existing results, run additional statistical tests
  - Textual revision: rewrite, restructure, or improve the prose of a section
  - Clarification: explain or define something more clearly (notation, claim, motivation)
  - Additional citations: cite missing related work
  - Methodology change: alter the proposed method itself
  - Data: collect more data, change the dataset, fix labels
  - None: only when the reviewer explicitly recommends something but it does not fit any category above (very rare)

SERIOUSNESS FIELDS (only for channel="critique"; set to null for strength and neutral)
- seriousness_explanation (free text, one or two sentences): WHY the critique matters at the seriousness level you assign.
- seriousness_category (one of three):
  - Essential: the critique must be addressed for the paper's stated conclusions to hold or for acceptance to be justified. Without addressing it, a major claim of the paper is unsupported.
  - Recommended: addressing the critique would substantially improve the paper, but the core contribution survives if it is not addressed.
  - Minor: small improvement; the paper is fine without addressing it.

NUMBERING
Number items 1, 2, 3, ... in the order they appear in the review text.

GENERAL GUIDANCE
- Err on the side of more items rather than fewer, but do not split a single coherent point into trivial fragments.
- Be precise about channel: a sentence that praises one thing while critiquing another should yield two items, one strength and one critique.
- If the review is very short or contains only a numerical score and no text, return an empty items list.

EXTRACTION SCHEMA (use the same order in json):
""" + str(EXTRACTION_SCHEMA) + ""


## extraction

@dataclass
class ExtractionResult:
    review_id: str
    paper_id: str
    review_text: str
    items: list
    model: str
    raw_content: str = ""  # raw JSON string from the model
    usage: dict = field(default_factory=dict)


def _client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(f"OPENAI_API_KEY not set, add it to {_HERE / '.env'}")
    return OpenAI(api_key=key)


def extract_concerns(review_text, *, review_id="unknown", paper_id="unknown",
                     model=MODEL_NAME, temperature=None):
    """Decompose one review into classified items."""
    if not review_text or not review_text.strip():
        return ExtractionResult(
            review_id=review_id,
            paper_id=paper_id,
            review_text=review_text or "",
            items=[],
            model=model,
        )

    client = _client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"REVIEW:\n\n{review_text.strip()}"},
    ]

    call_kwargs = {
        "model": model,
        "messages": messages,
        "response_format": RESPONSE_FORMAT,
    }
    if temperature is not None:
        call_kwargs["temperature"] = temperature

    resp = client.chat.completions.create(**call_kwargs)
    msg = resp.choices[0].message

    # with strict json_schema, the JSON is in msg.content
    raw_content = msg.content or ""
    if not raw_content.strip():
        refusal = getattr(msg, "refusal", None)
        raise RuntimeError(f"empty response for {review_id} (refusal={refusal!r})")

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"bad JSON from model for {review_id}: {e}\n{raw_content[:300]!r}") from e

    items = data.get("items", [])

    usage = {}
    if resp.usage:
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }

    return ExtractionResult(
        review_id=review_id,
        paper_id=paper_id,
        review_text=review_text,
        items=items,
        model=model,
        raw_content=raw_content,
        usage=usage,
    )


## saving

def save_extraction(result, out_path):
    """Save extraction result as a clean JSON file at out_path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "review_id": result.review_id,
        "paper_id": result.paper_id,
        "model": result.model,
        "n_items": len(result.items),
        "channel_counts": {
            c: sum(1 for it in result.items if it.get("channel") == c)
            for c in CHANNELS
        },
        "usage": result.usage,
        "items": result.items,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
