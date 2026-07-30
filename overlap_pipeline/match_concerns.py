"""
Concern matching for AI-vs-human review overlap measurement (stage 2 of 3).

Source-agnostic: matches list A against list B, so the same match_review serves
both AI-vs-human (A=AI concerns, B=pooled human concerns) and human-vs-human
leave-one-out (A=held-out reviewer, B=everyone else).

Only critique<->critique and strength<->strength pairs are matched (MATCH_CHANNELS).
Neutral items are stored on the result for stage 3 coverage variants but never
fed to the model.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")


## configuration

MODEL_NAME = "gpt-5.5"

MATCH_CHANNELS = ("critique", "strength")

KIND_AI = "ai_vs_human"
KIND_H2H = "human_vs_human"


## json schema for structured outputs

MATCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["matches"],
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["justification", "ai_number", "human_number"],
                "properties": {
                    "justification": {"type": "string"},
                    "ai_number": {"type": "integer"},
                    "human_number": {"type": "integer"},
                },
            },
        },
    },
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "concern_matching",
        "strict": True,
        "schema": MATCH_SCHEMA,
    },
}


## system prompt

SYSTEM_PROMPT = """You are an expert meta-reviewer. You are given the full text of an academic paper, a list of HUMAN review concerns about that paper, and a list of AI review concerns about the same paper. Each concern has a number, a channel (critique or strength), a type, and a short text.

TASK
Identify every pair (AI concern, human concern) that expresses the SAME underlying point about the paper, even if the two are worded differently or stated at a different level of granularity. Use the paper text to disambiguate: two differently-phrased concerns count as the same point only if they are really about the same aspect of this specific paper.

RULES
1. POLARITY: match a critique only to a critique, and a strength only to a strength. NEVER pair a critique with a strength.
2. SAME POINT: a pair is valid when both concerns target the same substantive issue, gap, claim, contribution, or strength of the paper. A vague thematic overlap (e.g. both mention "experiments") is NOT enough. They must be making the same specific point.
3. GRANULARITY: matches may be many-to-many. One AI concern may match several human concerns (e.g. an AI concern that bundles two points a human raised separately), and one human concern may be matched by several AI concerns. Emit each valid pair separately.
4. CONFIDENCE: only emit a pair you are reasonably confident shares the same point. When in doubt, do not emit the pair. Unmatched concerns simply do not appear in the output.
5. Give a one-sentence justification per pair, naming the shared point.
6. If nothing matches, return an empty list.

OUTPUT
Reply with a JSON object matching the provided schema: a single field "matches" containing a list of {"ai_number", "human_number", "justification"}. "ai_number" must be a number from the AI list; "human_number" must be a number from the human list.

RESPONSE FORMAT (use the same order in json):
""" + str(MATCH_SCHEMA) + ""


## item helpers

def filter_channels(items, channels=MATCH_CHANNELS):
    """Keep only items whose channel is in channels."""
    allowed = set(channels)
    return [it for it in items if it.get("channel") in allowed]


def build_human_union(items_by_reviewer, channels=None):
    """Flatten several reviewers' item lists into one numbered union.

    Pass channels to restrict; default None keeps all (including neutral) so
    stage 3 can compute with-neutral coverage. No cross-reviewer dedup.
    Each item gets: number (1..M), reviewer_file, original_number, channel,
    type, seriousness_category, recommendation, text.
    """
    allowed = set(channels) if channels is not None else None
    flat = []
    n = 0
    for reviewer_file in sorted(items_by_reviewer):
        for it in items_by_reviewer[reviewer_file]:
            if allowed is not None and it.get("channel") not in allowed:
                continue
            n += 1
            flat.append({
                "number": n,
                "reviewer_file": reviewer_file,
                "original_number": it.get("number"),
                "channel": it.get("channel"),
                "type": it.get("type"),
                "seriousness_category": it.get("seriousness_category"),
                "recommendation": it.get("recommendation"),
                "text": it.get("text", ""),
            })
    return flat


def renumber_items(items, *, source_file=None, channels=None):
    """Assign flat numbers 1..N for the A list (used for the held-out reviewer in h2h)."""
    allowed = set(channels) if channels is not None else None
    out = []
    n = 0
    for it in items:
        if allowed is not None and it.get("channel") not in allowed:
            continue
        n += 1
        out.append({
            "number": n,
            "reviewer_file": source_file,
            "original_number": it.get("number"),
            "channel": it.get("channel"),
            "type": it.get("type"),
            "seriousness_category": it.get("seriousness_category"),
            "recommendation": it.get("recommendation"),
            "text": it.get("text", ""),
        })
    return out


def _format_concern_list(items):
    """Render a numbered concern list for the prompt (number, channel, type, text)."""
    lines = []
    for it in items:
        num = it.get("number")
        ch = it.get("channel", "?")
        typ = it.get("type", "?")
        text = " ".join(str(it.get("text", "")).split())
        lines.append(f"[{num}] ({ch}; {typ}) {text}")
    return "\n".join(lines) if lines else "(none)"


## matching

@dataclass
class MatchResult:
    dataset: str
    paper_id: str
    source: str                     # gpt|llama|qwen, or held-out reviewer suffix for h2h
    matches: list
    ai_items: list                  # list-A snapshot
    human_items: list               # list-B flattened union with mapping fields
    model: str
    comparison_kind: str = KIND_AI
    raw_content: str = ""
    usage: dict = field(default_factory=dict)


def _client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(f"OPENAI_API_KEY not set, add it to {_HERE / '.env'}")
    return OpenAI(api_key=key)


def match_review(paper_text, ai_items, human_items, *, dataset, paper_id, source,
                 comparison_kind=KIND_AI, model=MODEL_NAME, temperature=None):
    """Match list A (ai_items) against list B (human_items) for one paper.

    Full lists are stored on the result for stage 3; only critique+strength
    items are shown to the model. Empty matches list is a valid outcome.
    """
    ai_shown = filter_channels(ai_items, MATCH_CHANNELS)
    human_shown = filter_channels(human_items, MATCH_CHANNELS)

    if not ai_shown or not human_shown:
        return MatchResult(
            dataset=dataset, paper_id=paper_id, source=source,
            matches=[], ai_items=ai_items, human_items=human_items,
            model=model, comparison_kind=comparison_kind,
        )

    client = _client()
    user_content = (
        f"PAPER (may be truncated):\n{paper_text.strip()}\n\n"
        f"=== HUMAN CONCERNS ===\n{_format_concern_list(human_shown)}\n\n"
        f"=== AI CONCERNS ===\n{_format_concern_list(ai_shown)}\n\n"
        f"Return the matches as JSON per the schema."
    )

    call_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "response_format": RESPONSE_FORMAT,
    }
    if temperature is not None:
        call_kwargs["temperature"] = temperature

    resp = client.chat.completions.create(**call_kwargs)
    msg = resp.choices[0].message
    raw_content = msg.content or ""
    if not raw_content.strip():
        refusal = getattr(msg, "refusal", None)
        raise RuntimeError(f"empty response for {dataset}/{paper_id}/{source} (refusal={refusal!r})")

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"bad JSON from matcher for {paper_id}/{source}: {e}\n{raw_content[:300]!r}") from e

    matches = data.get("matches", [])

    # drop pairs with out-of-range numbers or cross-polarity channel
    ai_by_num = {it["number"]: it for it in ai_items}
    hu_by_num = {it["number"]: it for it in human_items}
    clean = []
    for m in matches:
        a, h = m.get("ai_number"), m.get("human_number")
        if a not in ai_by_num or h not in hu_by_num:
            continue
        if ai_by_num[a].get("channel") != hu_by_num[h].get("channel"):
            continue
        clean.append({
            "ai_number": a,
            "human_number": h,
            "justification": m.get("justification", ""),
        })

    usage = {}
    if resp.usage:
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }

    return MatchResult(
        dataset=dataset, paper_id=paper_id, source=source,
        matches=clean, ai_items=ai_items, human_items=human_items,
        model=model, comparison_kind=comparison_kind,
        raw_content=raw_content, usage=usage,
    )


## saving

def save_match(result, out_path):
    """Persist a MatchResult as self-contained JSON for stage 3."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": result.dataset,
        "paper_id": result.paper_id,
        "source": result.source,
        "comparison_kind": result.comparison_kind,
        "model": result.model,
        "n_matches": len(result.matches),
        "n_ai_items": len(result.ai_items),
        "n_human_items": len(result.human_items),
        "usage": result.usage,
        "matches": result.matches,
        "ai_items": result.ai_items,
        "human_items": result.human_items,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
