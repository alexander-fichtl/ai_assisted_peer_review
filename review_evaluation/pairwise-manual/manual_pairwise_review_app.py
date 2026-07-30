#python -m streamlit run pairwise_review.py

"""
Pairwise evaluation of peer reviews:
- Displays the link to a reviewed paper.
- Displays two different generated (or human) reviews for that paper.
- Displays a feedback form and saves the user input.
- Exports the user feedback as CSV. 
(This CSV is loaded again in the next session to avoid displaying the same turn for a user twice.)
"""

#####----- Imports ----------------------------------------------------------------------------------------------------------#####
import csv
import json
import datetime as dt
import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import re
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Variables --------------------------------------------------------------------------------------------------------#####
ID_user = "ID-01"
VAR_reviewers = ["llama4_improvedPrompt_512cut", "llama4_improvedPrompt_fewshot_2000cut", "human"]
# "gpt5.1thinking", 
DATASET_NAME = "ICLR25_rndm50"

##-- Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]  
PATH_feedbackCSV = PROJECT_ROOT / "data" / "manual_eval" / f"feedback_{DATASET_NAME}_{ID_user}.csv"
PATH_openreviewSubmissions = PROJECT_ROOT / "data" / "openreview_basesets" / "ICLR.cc_2025_Conference" / "submissions_jsons"
PATH_generatedReviews = PROJECT_ROOT / "data" / "generated_reviews"
PATH_dataset = PROJECT_ROOT / "data" / "final_datasets" / "ICLR25_rndm50" / "index.json"
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Paper Database ---------------------------------------------------------------------------------------------------#####
@dataclass
class Paper:
    paper_id: str
    title: str
    url: str
    pdf_url: str
    reviews: Dict[str, Dict[str, str]]  # review_id -> {"source": ..., "md": ...}

def _sanitize_body_md(text: str) -> str:
    if not text:
        return ""
    t = str(text)

    # Demote Markdown headings that blow up the UI
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s+(.*)$", r"**\1**", t)          # "# Title" -> "**Title**"
    t = re.sub(r"(?m)^(.*)\n[=]{3,}\s*$", r"**\1**", t)             # Setext "===="
    t = re.sub(r"(?m)^(.*)\n[-]{3,}\s*$", r"**\1**", t)             # Setext "----"

    return t.strip()

def _score_num(x, overall_score=False):
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        if overall_score:
            m = re.search(r"Overall Score:\s*([0-9]+)", x, re.IGNORECASE)
            if m:
                return int(m.group(1))
        m = re.search(r"\b([0-9]+)\b", x)
        return int(m.group(1)) if m else None
    return None

def _score_disp(x, overall_score=False):
    n = _score_num(x, overall_score=overall_score)
    return str(n) if n is not None else "N/A"

def _strip_leading_heading(s: str, keywords: tuple[str, ...]) -> str:
    if not isinstance(s, str):
        return ""
    lines = [ln.rstrip() for ln in s.strip().splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].lstrip().startswith("#"):
        first = lines[0].lstrip("#").strip().lower()
        if any(k in first for k in keywords):
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)
    return "\n".join(lines).strip()

def _fmt_block(label: str, raw: str, keywords: tuple[str, ...]) -> str:
    body = _sanitize_body_md(_strip_leading_heading(raw or "", keywords))
    return f"**{label}:**\n\n{body}" if body else f"**{label}:**\n\nN/A"

def _format_review_openreview(summary, soundness, presentation, contribution, strengths, weaknesses, questions, ethics, rating, confidence, code_of_conduct=None):
    ethics_txt = "\n".join(ethics) if isinstance(ethics, list) else (ethics or "")
    ethics_txt = ethics_txt.strip() if isinstance(ethics_txt, str) else ""
    raw = "" if code_of_conduct is None else str(code_of_conduct).strip()
    coc_txt = {"no": "No", "yes": "Yes"}.get(raw.lower(), raw)

    blocks = [
        f"**Summary:**\n\n{_sanitize_body_md(summary or '') or 'N/A'}",
        "---",
        "  \n".join([  # <- markdown hard line breaks
            f"**Soundness:** {_score_disp(soundness)}",
            f"**Presentation:** {_score_disp(presentation)}",
            f"**Contribution:** {_score_disp(contribution)}",
            f"**Rating:** {_score_disp(rating, overall_score=True)}",
            f"**Confidence:** {_score_disp(confidence)}",
        ]),
        "---",
        _fmt_block("Strengths", strengths, ("strength",)),
        "---",
        _fmt_block("Weaknesses", weaknesses, ("weakness",)),
        "---",
        _fmt_block("Questions", questions, ("question", "suggestion")),
    ]

    if ethics_txt:
        blocks += ["---", f"Flag For Ethics Review:\n{ethics_txt}"]
    if coc_txt:
        blocks += ["---", f"Code Of Conduct:\n{coc_txt}"]

    return "\n\n".join(blocks).strip()

def create_paperBase() -> Dict[str, Paper]:
    ##-- Load dataset index (paper ids)
    index = json.loads(PATH_dataset.read_text(encoding="utf-8"))
    paper_ids = index["paper_ids"]

    ##-- Create paper set
    papers: Dict[str, Paper] = {}
    for paper_id in paper_ids:
        title = f""
        url_forum = f"https://openreview.net/forum?id={paper_id}"
        url_pdf = f""
        reviews: Dict[str, Dict[str, str]] = {}

        ##-- Openreview submission json (for title/url/pdf + human reviews)
        sub_path = PATH_openreviewSubmissions / f"{paper_id}.json"
        if sub_path.exists():
            sub = json.loads(sub_path.read_text(encoding="utf-8"))
            content = sub.get("content", {}) or {}
            title = (content.get("title", {}) or {}).get("value", title)

            url_forum = f"https://openreview.net/forum?id={sub.get('forum', paper_id)}"
            url_pdf = (lambda p: f"https://openreview.net{p}" if isinstance(p, str) and p.startswith("/pdf/") else "")(((content.get("pdf") or {}).get("value", "")))

            if "human" in VAR_reviewers:
                details = sub.get("details", {}) or {}
                direct_replies = details.get("directReplies", []) or []

                for note in direct_replies:
                    invitations = note.get("invitations", []) or []
                    if not any("Official_Review" in inv for inv in invitations):
                        continue

                    n_content = note.get("content", {}) or {}

                    summary = (n_content.get("summary", {}) or {}).get("value", "")
                    soundness = (n_content.get("soundness", {}) or {}).get("value", None)
                    presentation = (n_content.get("presentation", {}) or {}).get("value", None)
                    contribution = (n_content.get("contribution", {}) or {}).get("value", None)
                    strengths = (n_content.get("strengths", {}) or {}).get("value", "")
                    weaknesses = (n_content.get("weaknesses", {}) or {}).get("value", "")
                    questions = (n_content.get("questions", {}) or {}).get("value", "")
                    ethics = (n_content.get("flag_for_ethics_review", {}) or {}).get("value", [])
                    rating = (n_content.get("rating", {}) or {}).get("value", None)
                    confidence = (n_content.get("confidence", {}) or {}).get("value", None)

                    signatures = note.get("signatures", []) or []
                    sig0 = signatures[0] if signatures else ""
                    m = re.search(r"Reviewer_([A-Za-z0-9]+)", sig0)
                    token = (m.group(1).lower() if m else str(note.get("number", "")).strip().lower()) or "x"

                    review_id = f"{paper_id}_human_{token}"

                    md = _format_review_openreview(
                        summary=summary,
                        soundness=soundness,
                        presentation=presentation,
                        contribution=contribution,
                        strengths=strengths,
                        weaknesses=weaknesses,
                        questions=questions,
                        ethics=ethics,
                        rating=rating,
                        confidence=confidence,
                        code_of_conduct=(n_content.get("code_of_conduct", {}) or {}).get("value", None),
                    )
                    reviews[review_id] = {"source": "human", "md": md}

        ##-- Generated reviews
        for reviewer in VAR_reviewers:
            if reviewer == "human":
                continue

            gen_path = PATH_generatedReviews / reviewer / paper_id / "final_review_openreview.json"
            if not gen_path.exists():
                print(f"ERROR: Path {gen_path} does not exist.")
                continue

            gen = json.loads(gen_path.read_text(encoding="utf-8"))
            g_content = gen.get("content", {}) or {}

            summary = (g_content.get("summary", {}) or {}).get("value", "")
            soundness = (g_content.get("soundness", {}) or {}).get("value", None)
            presentation = (g_content.get("presentation", {}) or {}).get("value", None)
            contribution = (g_content.get("contribution", {}) or {}).get("value", None)
            strengths = (g_content.get("strengths", {}) or {}).get("value", "")
            weaknesses = (g_content.get("weaknesses", {}) or {}).get("value", "")
            questions = (g_content.get("questions", {}) or {}).get("value", "")
            ethics = (g_content.get("flag_for_ethics_review", {}) or {}).get("value", [])
            rating = (g_content.get("rating", {}) or {}).get("value", None)
            confidence = (g_content.get("confidence", {}) or {}).get("value", None)

            review_id = f"{paper_id}_{reviewer}"

            md = _format_review_openreview(
                summary=summary,
                soundness=soundness,
                presentation=presentation,
                contribution=contribution,
                strengths=strengths,
                weaknesses=weaknesses,
                questions=questions,
                ethics=ethics,
                rating=rating,
                confidence=confidence,
                code_of_conduct=(g_content.get("code_of_conduct", {}) or {}).get("value", None),
            )
            reviews[review_id] = {"source": reviewer, "md": md}

        ##-- Keep only papers that have at least 2 reviews
        if len(reviews) >= 2:
            papers[paper_id] = Paper(
                paper_id=paper_id,
                title=str(title),
                url=str(url_forum),
                pdf_url=str(url_pdf),
                reviews=reviews,
            )

    ##-- Stats
    counts_per_reviewer = {r: 0 for r in VAR_reviewers}
    for p in papers.values():
        for rid, meta in p.reviews.items():
            counts_per_reviewer[meta["source"]] = counts_per_reviewer.get(meta["source"], 0) + 1

    print(f"[paperBase] papers_with_>=2_reviews: {len(papers)}")
    print(f"[paperBase] review_counts_by_source: {counts_per_reviewer}")

    return papers
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Helper Functions -------------------------------------------------------------------------------------------------#####
def _strip_toggle_heading(text: str, names: tuple[str, ...]) -> str:
    t = _sanitize_body_md(text or "").strip()
    if not t:
        return ""
    for n in names:
        # removes: "Summary:", "**Summary:**", "Summary\n", "**Summary**\n", with optional colon
        t2 = re.sub(rf"(?is)^\s*(\*\*)?\s*{re.escape(n)}\s*(\*\*)?\s*:?\s*(\n+| )", "", t, count=1).strip()
        if t2 != t:
            t = t2
            break
    return t.strip()

def _ethics_tag(ethics_body: str) -> str:
    first = (ethics_body or "").strip().splitlines()[0].strip() if (ethics_body or "").strip() else ""
    low = first.lower()
    if re.search(r"\byes\b", low):
        return "Yes"
    if re.search(r"\bno\b", low):
        return "No"
    return first[:40] + ("..." if len(first) > 40 else "") or "N/A"

def _split_review_sections(md: str) -> dict:
    parts = [p.strip() for p in re.split(r"\n\s*---\s*\n", (md or "").strip()) if p.strip()]
    out = {"summary": "", "scores": "", "strengths": "", "weaknesses": "", "questions": "", "ethics": "", "coc": "", "rest": []}

    for p in parts:
        low = p.lstrip().lower()

        if low.startswith("**summary:**") or low.startswith("summary:"):
            out["summary"] = _strip_toggle_heading(p, ("Summary",))

        elif low.startswith("**strengths:**") or low.startswith("strengths:"):
            out["strengths"] = _strip_toggle_heading(p, ("Strengths", "Strength"))

        elif low.startswith("**weaknesses:**") or low.startswith("weaknesses:"):
            out["weaknesses"] = _strip_toggle_heading(p, ("Weaknesses", "Weakness"))

        elif low.startswith("**questions:**") or low.startswith("questions:"):
            out["questions"] = _strip_toggle_heading(p, ("Questions", "Question", "Suggestions", "Suggestion"))

        elif low.startswith("flag for ethics review:"):
            out["ethics"] = _strip_toggle_heading(p, ("Flag For Ethics Review",))

        elif low.startswith("code of conduct:"):
            out["coc"] = _strip_toggle_heading(p, ("Code Of Conduct",))

        elif low.startswith("**soundness:**") or "**soundness:**" in low:
            out["scores"] = p 

        else:
            out["rest"].append(p)

    return out

def _drop_label(md: str, label: str) -> str:
    s = (md or "").strip()
    s = re.sub(rf"(?is)^\s*\*\*{re.escape(label)}:\*\*\s*", "", s)  # **Summary:**
    s = re.sub(rf"(?is)^\s*{re.escape(label)}:\s*", "", s)          # Summary:
    return s.strip()
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Streamlit App ----------------------------------------------------------------------------------------------------#####
def main():
    ##-- Initialize session state
    if "paperBase" not in st.session_state:
        st.session_state.paperBase = create_paperBase()
    if "turns" not in st.session_state:
        st.session_state.turns = []  # list of (paper_id, rid_a, rid_b)
    if "turn_i" not in st.session_state:
        st.session_state.turn_i = 0

    ##-- Load existing feedback to exclude already shown pairs
    seen = set()
    if PATH_feedbackCSV.exists():
        df_seen = pd.read_csv(PATH_feedbackCSV)
        for _, r in df_seen.iterrows():
            pid = str(r["paper_id"])
            a = str(r["pair_a"]); b = str(r["pair_b"])
            a, b = sorted([a, b])
            seen.add((pid, a, b))


    ##-- Build unseen turns once per session
    if not st.session_state.turns:
        turns = []
        for pid, paper in st.session_state.paperBase.items():
            rids = sorted(paper.reviews.keys())
            for a, b in itertools.combinations(rids, 2):
                key = (pid, *sorted([a, b]))
                if key in seen:
                    continue
                turns.append((pid, a, b))

        random.shuffle(turns)
        st.session_state.turns = turns
        st.session_state.turn_i = 0

    ##-- Setup streamlit app
    st.set_page_config(page_title="Pairwise Evaluation of Peer Reviews", layout="wide")
    st.title("Pairwise Evaluation of Peer Reviews")

    ##-- No turns left
    if st.session_state.turn_i >= len(st.session_state.turns):
        st.success("No unseen review pairs left for this user.")
        if PATH_feedbackCSV.exists():
            st.download_button(
                "Download CSV",
                data=PATH_feedbackCSV.read_bytes(),
                file_name=PATH_feedbackCSV.name,
                mime="text/csv",
            )
        return

    ##-- Current turn
    pid, rid_a, rid_b = st.session_state.turns[st.session_state.turn_i]
    paper = st.session_state.paperBase[pid]
    left_rid, right_rid = (rid_a, rid_b) if random.random() < 0.5 else (rid_b, rid_a)
    left = paper.reviews[left_rid]
    right = paper.reviews[right_rid]

    ##-- Display header (titel, paper links, etc.)
    c_1, c_or, c_pdf = st.columns([4, 1, 1])
    with c_pdf:
        if paper.pdf_url.strip():
            st.link_button("PDF", paper.pdf_url)
    with c_or:
        st.link_button("OpenReview", paper.url)
    with c_1:
        st.markdown(
            f"""
            <div style="text-align:left; font-size:0.8rem; color:grey; line-height:1.2;">
                paper_id: {paper.paper_id}<br>
                pair {st.session_state.turn_i+1}/{len(st.session_state.turns)}
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(f"## {paper.title}")

    st.subheader("Reviews")
    cols = st.columns(2)
    #with cols[0]:
    #    st.markdown(f"**Left:** `{left_rid}`  —  source: `{left['source']}`")
    #    st.markdown(left["md"])
    #with cols[1]:
    #    st.markdown(f"**Right:** `{right_rid}`  —  source: `{right['source']}`")
    #    st.markdown(right["md"])
    with cols[0]:
        L = _split_review_sections(left["md"])

        with st.expander("Summary", expanded=True):
            st.markdown(_drop_label(L["summary"], "Summary") or "N/A")

        st.markdown(L["scores"] or "")

        with st.expander("Strengths", expanded=True):
            st.markdown(_drop_label(L["strengths"], "Strengths") or "N/A")

        with st.expander("Weaknesses", expanded=True):
            st.markdown(_drop_label(L["weaknesses"], "Weaknesses") or "N/A")

        with st.expander("Questions", expanded=True):
            st.markdown(_drop_label(L["questions"], "Questions") or "N/A")

        if (L["ethics"] or "").strip():
            tag = _ethics_tag(L["ethics"])
            with st.expander(f"Flag For Ethics Review: {tag}", expanded=False):
                st.markdown(L["ethics"])

        if (L["coc"] or "").strip():
            st.markdown(f"**Code Of Conduct:** {L['coc']}")
            #with st.expander("Code Of Conduct", expanded=False):
            #    st.markdown(L["coc"])
            #st.divider()

        for extra in L["rest"]:
            st.markdown(extra)

    with cols[1]:
        R = _split_review_sections(right["md"])

        with st.expander("Summary", expanded=True):
            st.markdown(_drop_label(R["summary"], "Summary") or "N/A")

        st.markdown(R["scores"] or "")

        with st.expander("Strengths", expanded=True):
            st.markdown(_drop_label(R["strengths"], "Strengths") or "N/A")

        with st.expander("Weaknesses", expanded=True):
            st.markdown(_drop_label(R["weaknesses"], "Weaknesses") or "N/A")

        with st.expander("Questions", expanded=True):
            st.markdown(_drop_label(R["questions"], "Questions") or "N/A")

        if (R["ethics"] or "").strip():
            tag = _ethics_tag(R["ethics"])
            with st.expander(f"Flag For Ethics Review: {tag}", expanded=False):
                st.markdown(R["ethics"])
            #st.divider()

        if (R["coc"] or "").strip():
            st.markdown(f"**Code Of Conduct:** {R['coc']}")
            #with st.expander("Code Of Conduct", expanded=False):
            #    st.markdown(R["coc"])
            #st.divider()

        for extra in R["rest"]:
            st.markdown(extra)
            #st.divider()

    st.markdown("---")
    st.subheader("Feedback")

    with st.form("feedback_form"):
        fcols = st.columns(2)

        with fcols[0]:
            st.markdown("### Left review")
            overall_l          = st.number_input("Overall score Left (1–7)", min_value=1, max_value=7, value=4, step=1)
            understanding_l    = st.number_input("Understanding Left (1–5)", min_value=1, max_value=5, value=4, step=1)
            coverage_l         = st.number_input("Coverage Left (1–5)", min_value=1, max_value=5, value=4, step=1)
            evidence_support_l = st.number_input("Evidence support Left (1–5)", min_value=1, max_value=5, value=4, step=1)
            constructiveness_l = st.number_input("Constructiveness Left (1–5)", min_value=1, max_value=5, value=4, step=1)
            conciseness_l      = st.number_input("Conciseness Left (1–5)", min_value=1, max_value=5, value=4, step=1)
            granuality_l       = st.number_input("Granuality Left (1–5)", min_value=1, max_value=5, value=4, step=1)
            comment_l          = st.text_area("Comments Left", height=120)

        with fcols[1]:
            st.markdown("### Right review")
            overall_r          = st.number_input("Overall score Right (1–7)", min_value=1, max_value=7, value=4, step=1)
            understanding_r    = st.number_input("Understanding Right (1–5)", min_value=1, max_value=5, value=4, step=1)
            coverage_r         = st.number_input("Coverage Right (1–5)", min_value=1, max_value=5, value=4, step=1)
            evidence_support_r = st.number_input("Evidence support Right (1–5)", min_value=1, max_value=5, value=4, step=1)
            constructiveness_r = st.number_input("Constructiveness Right (1–5)", min_value=1, max_value=5, value=4, step=1)
            conciseness_r      = st.number_input("Conciseness Right (1–5)", min_value=1, max_value=5, value=4, step=1)
            granuality_r       = st.number_input("Granuality quality Right (1–5)", min_value=1, max_value=5, value=4, step=1)
            comment_r          = st.text_area("Comments Right", height=120)

        winner = st.radio("Winner", ["Left", "Right", "Tie"], horizontal=True)
        submitted = st.form_submit_button("Save & Next")


    if submitted:
        PATH_feedbackCSV.parent.mkdir(parents=True, exist_ok=True)
        pair_a, pair_b = sorted([left_rid, right_rid])
        pair_id = f"{pid}__{pair_a}__{pair_b}"
        winner_id = "tie" if winner == "Tie" else (left_rid if winner == "Left" else right_rid)

        row = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "user_id": ID_user,
            "paper_id": pid,

            "pair_id": pair_id,
            "pair_a": pair_a,
            "pair_b": pair_b,

            "review_id_left": left_rid,
            "review_id_right": right_rid,

            #left scores
            "overall_score_left": overall_l,
            "understanding_left": understanding_l,
            "coverage_left": coverage_l,
            "evidence_support_left": evidence_support_l,
            "constructiveness_left": constructiveness_l,
            "conciseness_left": conciseness_l,
            "granuality_left": granuality_l,
            "comments_left": comment_l,

            #right scores
            "overall_score_right": overall_r,
            "understanding_right": understanding_r,
            "coverage_right": coverage_r,
            "evidence_support_right": evidence_support_r,
            "constructiveness_right": constructiveness_r,
            "conciseness_right": conciseness_r,
            "granuality_right": granuality_r,
            "comments_right": comment_r,

            "winner": winner_id,
        }

        write_header = not PATH_feedbackCSV.exists()
        with PATH_feedbackCSV.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

        # advance
        st.session_state.turn_i += 1
        st.rerun()

    # --- download CSV any time
    if PATH_feedbackCSV.exists():
        st.download_button(
            "Download CSV",
            data=PATH_feedbackCSV.read_bytes(),
            file_name=PATH_feedbackCSV.name,
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
#####------------------------------------------------------------------------------------------------------------------------#####