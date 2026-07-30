"""
Extracts the first round of author remarks from Nature Communication's public peerreviews.
"""

#####----- Imports ----------------------------------------------------------------------------------------------------------#####
import re
import json
from pathlib import Path
from typing import Dict, List, Any
#####------------------------------------------------------------------------------------------------------------------------#####



#####------------------------------------------------------------------------------------------------------------------------#####
#Manually checked/fixed papers are skipped (i.e., NOT overwritten)
DONE_IDS_OLD = {
    "s41467-024-45563-x",
    "s41467-024-45102-8",
    "s41467-024-45914-8",
    "s41467-024-46947-9",
    "s41467-024-47824-1",
    "s41467-024-48005-w",
    "s41467-024-48115-5",
    "s41467-024-49173-5",
    "s41467-024-50043-3",
    "s41467-024-48069-8",
    "s41467-024-49780-2",
    "s41467-024-50310-3",
    "s41467-024-50903-y",
    "s41467-024-51844-2",
    "s41467-024-51749-0",
    "s41467-024-52293-7", 
    "s41467-024-52417-z",
    "s41467-024-53190-9",
    "s41467-024-53759-4",
    "s41467-024-54069-5",
    "s41467-024-54071-x",
    "s41467-024-51465-9",
    "s41467-024-54457-x",
    "s41467-024-54639-7",
    "s41467-024-55158-1",
    "s41467-025-55920-z", 
    "s41467-025-56162-9",
    "s41467-024-55399-0",
    "s41467-025-56254-6"
}


DONE_IDS = {
    "s41467-024-49798-6",
    "s41467-024-51669-z",
    "s41467-024-53287-1",
    "s41467-024-53457-1",
    "s41467-024-53742-z",
    "s41467-024-53768-3",
    "s41467-024-55146-5",
    "s41467-025-55920-z",
    "s41467-025-56625-z",
    "s41467-025-57426-0",
    "s41467-025-57473-7",
    "s41467-025-58872-6",
    "s41467-025-59797-w",
    "s41467-025-59822-y",
    "s41467-025-60989-7", #mentions names
    "s41467-025-61040-5",
    "s41467-025-61309-9",
    "s41467-025-61599-z",
    "s41467-025-61684-3",
    "s41467-025-61754-6",
    "s41467-025-63160-4",
    "s41467-025-63846-9",
    "s41467-025-64105-7",
    "s41467-025-64227-y",
    "s41467-025-64574-w",
    "s41467-025-64681-8",
    "s41467-025-64894-x",
    "s41467-025-66049-4",
    "s41467-025-66507-z",
    "s41467-025-66620-z",
    "s41467-025-66644-5",
    "s41467-025-67178-6",
    "s41467-025-67791-5"
}
#Missing generated json: "s41467-025-55920-z",
#Invalid: "s41467-025-59797-w", "s41467-025-59822-y"
#####------------------------------------------------------------------------------------------------------------------------#####



#####------------------------------------------------------------------------------------------------------------------------#####
#Matches headings like "Reviewer #1 (Remarks to the Author)" or "Reviewer #2:\nRemarks to Authors:" (case-insensitive),
#optionally with newlines between the "Reviewer #1" line and the "(Remarks to the Author)" line:
# "Reviewer #1 (Remarks to the Author)"
# "Reviewer #2\n\n(Remarks to the Authors)"
# "# Reviewer #3\n(Remarks to the Author)"
# "Reviewer #2:\n\nRemarks to the Authors:"

# Notes:
# - Optional markdown heading level before "Reviewer" (can be ignored i think?)
# - Literal "#" before the number
# - Any reviewer number (1, 2, 3, ...)
# - Allows "Author" or "Authors"
# - Optional newlines between "Reviewer #<n>" and "(Remarks ...)"
RE_REVIEWER = re.compile(
    r"""
    ^\s*(?:\#{1,6}\s*)?                 # optional markdown heading before 'Reviewer'
    Reviewer
    (?:\s*\#)?\s*                       # optional '#'
    (\d+)\b                             # reviewer number
    \s*[:.\-]?\s*                       # optional punctuation after number

    (?:\r?\n[ \t]*)*                    # optional blank/whitespace-only lines
    (?:\#{1,6}\s*)?                     # optional markdown heading before remarks line

    \(?\s*Remarks\s+to\s+(?:the\s+)?    # optional 'the'
    Author(?:s|\(s\))?                  # Author / Authors / Author(s)
    \s*\)?\s*                           # optional ')'
    :?\s*                               # optional ':'
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)
#Reviewer #1 (Remarks to the Author)

#"...ents:\n\nReviewer #1\n\n## (Remarks to the Author)"

#"\n\nReviewer #1 (Remarks to the Author):"

#Where first-round remarks end (heading level optional)
# Examples seen:
# "Point-by-point responses"
# "Response Letter"
# "## Manuscript NCOMMS-23-28663 Response to Reviewers"
# "Authors’ Response to Reviews of..."
# "Detailed Responses to Reviewers’ Comments"
# "RESPONSE TO REVIEWERS’ COMMENTS"
RE_FIRST_ROUND_END = re.compile(
    r'^\s*(?:#{1,6}\s*)?(?:'
    r'Point[-\s]*by[-\s]*point\s+responses\b'
    r'|Response\s+Letter\b'
    r'|.*\bResponse\s+to\s+Reviewers\b.*'
    r'|Authors[’\']?\s*Response\b.*'
    r'|Detailed\s+Responses?\b.*'
    r'|RESPONSE\s+TO\s+REVIEWERS[’\']?\s*COMMENTS\b.*'
    r')',
    re.IGNORECASE | re.MULTILINE,
)
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Functions --------------------------------------------------------------------------------------------------------#####
def extract_first_round(text: str) -> Dict[str, Any]:
    """Extract first-round 'Remarks to the Author(s)' blocks from a Nature Com peer-review .md."""

    # Best-effort cutoff before author response section
    m_end = RE_FIRST_ROUND_END.search(text)
    limit = m_end.start() if m_end else len(text)

    first_round_text = text[:limit]
    cut_after = text[limit:]  # includes the cut marker heading (if present)

    matches = list(RE_REVIEWER.finditer(first_round_text))

    # Everything before the first reviewer remarks header
    if matches:
        cut_before = first_round_text[:matches[0].start()].rstrip()
    else:
        cut_before = first_round_text.rstrip()

    first_remarks: List[Dict[str, Any]] = []

    for i, m in enumerate(matches):
        reviewer_id = m.group(1)

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(first_round_text)

        block = first_round_text[start:end].strip()
        if block:
            first_remarks.append(
                {
                    "reviewer_id": reviewer_id,
                    "remarks_to_the_author": block,
                }
            )

    return {
        "cut_before": cut_before,
        "cut_after": cut_after,
        "first_remarks": first_remarks,
    }

def process_md_file(md_path: Path) -> None:
    """Convert one <paper-id>.md file to reviews_<paper-id>.json in the same folder."""
    text = md_path.read_text(encoding="utf-8")
    data = extract_first_round(text)

    #Get paper-id from filename, e.g. s41467-024-45102-8 from s41467-024-45102-8.md
    paper_id = md_path.stem
    out_path = md_path.parent / f"reviews_{paper_id}.json"

    #Overwrite by default; DONE ids are skipped in process_folder()
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"Wrote {out_path}")


def process_folder(root: Path) -> None:
    """Convert only files with schema: <root>/<paper-id>/<paper-id>.md"""
    done_ids = set(DONE_IDS)

    for paper_dir in root.iterdir():
        if not paper_dir.is_dir():
            continue

        paper_id = paper_dir.name

        #Skip papers you already fixed manually
        if paper_id in done_ids:
            print(f"Skip DONE {paper_id}")
            continue

        md_path = paper_dir / f"{paper_id}.md"
        if md_path.is_file():
            process_md_file(md_path)
#####------------------------------------------------------------------------------------------------------------------------#####



folder = Path("data/processed_reviews") / "natureCom"
process_folder(folder)
