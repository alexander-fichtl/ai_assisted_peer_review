"""
Input: JSON produced by the Docling preprocessing pipeline (document.save_as_json with ImageRefMode.REFERENCED)
- vLLM OpenAI-compatible server backend (Qwen3-VL)
- Single-pass Nature Communications style review
Outputs: final_review.json, final_review.html, final_review.txt, audit.log
"""



#####----- Imports ----------------------------------------------------------------------------------------------------------#####
###--- Standard library
import base64
import copy
import gc
import json
import os
import re
import traceback
from io import BytesIO
from pathlib import Path
import requests
from typing import Any, Dict, List, Optional, Tuple

###--- Third-party
import numpy as np
from PIL import Image
import torch
import torch._dynamo
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Variables --------------------------------------------------------------------------------------------------------#####
CACHE_DIR = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
HF_TOKEN = os.environ.get("HF_TOKEN")

###--- Prompting
VAR_excludedSections = ["References", "Reference", "Acknowledgements", "Acknowledgement", "Author contributions", "Competing interests", "Additional information", "Appendix", "Supplementary", "Funding", "Corresponding authors", "Data availability", "Code availability"]
VAR_repeatPrevQA = True
VAR_fewshot = False
VAR_2ndPass = False
VAR_imagestandards = True
VAR_peerreview = True
VAR_researchethics = True

###--- Technical
torch._dynamo.config.capture_scalar_outputs = True
VAR_gpus = 4
VAR_quantized = False
VAR_maxContext = 64_000
VAR_tokensPerAnswer = 10000
VAR_tokensPerAnswer_2ndPass = 10000
VAR_model = "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"
VAR_setup = "vllm"
VAR_temperature = 0.6

###--- Server
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
VLLM_TIMEOUT = int(os.environ.get("VLLM_TIMEOUT", "600"))

###--- PATHS
PATH_output = Path("reviews_qwen_natureCom")
PATH_input = Path("processed_pdfs_natureCom/processed_pdfs_natureCom")   #Docling preprocessing outputs (folder with JSON + images)
PATH_imagestandards = Path("context/context_natureCom/imagestandards.txt")
PATH_peerreview = Path("context/context_natureCom/peerreview.txt")
PATH_researchethics = Path("context/context_natureCom/researchethics.txt")

###--- Images
VAR_maxImages = 5 #Set to 0 for ignoring images
VAR_maxImageSize = 4_000_000   # ~4MB
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Prompts ----------------------------------------------------------------------------------------------------------#####
PROMPT_system = (
    "You are a top-tier academic reviewer. Follow all rules:\
        1) Use only the provided context. Do not hallucinate.\
        2) No chain-of-thought in the final answer.\
        3) Write in natural flowing prose like a human expert reviewer. Do not use section headers or a rigid structure. Weave validity, significance, methodology, and suggestions naturally into your text. Be concise, relevant, specific, and non-repetitive. No boilerplate. Not unnecessarily long.\
        4) No restating of the questions or instructions.\
        3) Cite evidence from context. E.g. the section title or figure id or table id. You can also quote short sentence parts.\
        5) For listing strengths/weaknesses/suggestions prefer numbered lists (1., 2., …). Keep the amount of points reasonable. Only keep the least generic points.\
        8) You can incorporate Markdown and Latex into your review.\
        9) A total review has typically at least 300 and maximal 2000 words. Prefer your reviews to have between 500 and 1500 words."""
)

PROMPT_reviewInstruction = """Criteria for publication
Your review is vital in helping our editors decide if the manuscript meets the journal's criteria for publication, and we ask you to keep the following factors in mind when you write your report:
- The quality of the data — whether they are technically sound, obtained with appropriate techniques, analysed and interpreted carefully, and presented in sufficient detail.
- The level of support for the conclusions — whether sufficiently strong evidence is provided for the authors' claims and all appropriate controls have been included.
- The potential significance of the results — whether these results will be important to the field and advance understanding in a way that will move the field forward. (Note that posting of preprints and/or conference proceedings does not compromise novelty.)

The primary purpose of your review is to provide feedback on the soundness of the research reported. This will help authors to improve their manuscript and editors to reach a decision. We do not ask that you make a recommendation regarding publication, but you can set out the arguments for and against publication if you so wish.

Elements of a reviewer report
In your report, please comment on the following aspects of the manuscript.
- Key results: Your overview of the key messages of the study, in your own words, highlighting what you find significant or notable. Usually, this can be summarized in a short paragraph.
- Validity: Your evaluation of the validity and robustness of the data interpretation and conclusions. If you feel there are flaws that prohibit the manuscript's publication, please describe them in detail.
- Significance: Your view on the potential significance of the conclusions for the field and related fields. If you think that other findings in the published literature compromise the manuscript's significance, please provide relevant references.
- Data and methodology: Your assessment of the validity of the approach, the quality of the data, and the quality of presentation. We ask reviewers to assess all data, including those provided as supplementary information. If any aspect of the data is outside the scope of your expertise, please note this in your report or in the comments to the editor. We may, on a case-by-case basis, ask reviewers to check code provided by the authors (see this Nature editorial for more information).
- Analytical approach: Your assessment of the strength of the analytical approach, including the validity and comprehensiveness of any statistical tests. If any aspect of the analytical approach is outside the scope of your expertise, please note this in your report or in the comments to the editor.
- Suggested improvements: Your suggestions for additional experiments or data that could help strengthen the work and make it suitable for publication in the journal. Suggestions should be limited to the present scope of the manuscript; that is, they should only include what can be reasonably addressed in a revision and exclude what would significantly change the scope of the work. The editor will assess all the suggestions received and provide additional guidance to the authors.
- Clarity and context: Your view on the clarity and accessibility of the text, and whether the results have been provided with sufficient context and consideration of previous work. Note that we are not asking for you to comment on language issues such as spelling or grammatical mistakes.
- References: Your view on whether the manuscript references previous literature appropriately.
- Your expertise: Please indicate any particular part of the manuscript, data or analyses that you feel is outside the scope of your expertise, or that you were unable to assess fully.

Providing constructive feedback
We ask reviewers to approach peer review with a sincere intention to help the authors improve their manuscripts. Nearly all submissions have weaknesses to be addressed: the best and most constructive reports suggest specific improvements; such feedback can be used by authors to improve their manuscript to the point where it might be suitable for acceptance. Even in instances where manuscripts are rejected, your report will help authors interpret the editor's decision and improve their work prior to submission elsewhere.
You should be direct in your report, but you should also maintain a respectful tone. As a matter of policy, we do not censor the content of reviewer reports; any comments that were intended for the authors are transmitted, regardless of what we may think of the content. On rare occasions, we may edit a report to remove offensive language or comments that reveal confidential information about other matters."""

PROMPT_fewshots = [
    ("not useful", """x"""),
    ("useful", """x""")
]

def build_system_prompt() -> str:
    system = PROMPT_system
    #Fewshot
    if VAR_fewshot and PROMPT_fewshots:
        lines = [
            "\n\nBelow are examples of review snippets from other papers. "
            "Mimic the style of 'useful' ones and avoid the style of 'not useful' ones. "
            "Use them only as guidance; do not copy them verbatim or the judgement.",
        ]
        for label, ex in PROMPT_fewshots:
            lines.append(f"\n[{label.upper()}]\n{ex}")
        system = system + "\n" + "\n".join(lines)
        
    #Guidelines
    # Extra guidance text files (only if flags are True and files exist)
    if VAR_imagestandards:
        txt = Path(PATH_imagestandards).read_text(encoding="utf-8")
        system += "\n\n[Standards for images]\n" + txt

    if VAR_researchethics:
        txt = Path(PATH_researchethics).read_text(encoding="utf-8")
        system += "\n\n[General research ethics]\n" + txt

    if VAR_peerreview:
        txt = Path(PATH_peerreview).read_text(encoding="utf-8")
        system += "\n\n[General guidelines for writing a reviewer report]\n" + txt

    return system

PROMPT_2ndPass_global = (
    "You are editing a complete academic peer review (answers to multiple fixed questions).\n"
    "Your task is to rewrite ALL answers to be much more concise and non-repetitive, while preserving meaning and factual content.\n"
    "Rules:\n"
    "1) Preserve every concrete technical point, criticism, suggestion, question, and positive remark that adds relevant information for the authors of the paper or the area chair.\n"
    "2) Remove filler, hedging, verbosity, and any sentences that only repeat earlier content.\n"
    "3) Eliminate duplication across answers: if a point appears in multiple answers, keep the most appropriate occurrence (e.g. repetitions between Weaknesses and Questions) and remove or shorten it elsewhere.\n"
    "4) Do NOT change any numeric ratings or options. Lines like 'Rating: <number>' and ethics / conduct labels must stay exactly as they are if present.\n"
    "5) Respect the role of each section: Summary is neutral description; Soundness/Presentation/Contribution briefly justify their ratings; Strengths and Weaknesses are numbered lists with only the most specific, non-generic items; Questions focus on unclarities or actionable requests.\n"
    "6) Output the edited review in exactly the same structure as the input: for each question, use the same heading line starting with '### ' and the same question header text, followed by its edited answer. Do not add or remove questions, and do not add commentary outside the review.\n"
)
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Parse preprocessed PDF -------------------------------------------------------------------------------------------#####
def image_to_dataUrl(path, max_bytes = VAR_maxImageSize):
    """Load image, re-encode to stay under max_bytes, return data URL or None if impossible."""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            for quality in [95, 85, 75, 65, 50, 40, 30]:
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True)
                data = buf.getvalue()
                if len(data) <= max_bytes:
                    b64 = base64.b64encode(data).decode("ascii")
                    return f"data:image/jpeg;base64,{b64}"
    except Exception:
        print(f"Error for image {path}. Skipped.")
        pass
    return None

def strip_thinking(s: str) -> str:
    """Remove <think>...</think> blocks (and variants) from model output."""
    LT = r"(?:&lt;|&#0*60;|&#x0*3c;)"
    GT = r"(?:&gt;|&#0*62;|&#x0*3e;)"
    # Cut everything from start through first closing tag
    for pat in [
        re.compile(r"(?is)\A.*?</\s*(think|thinking)\s*>"),
        re.compile(rf"(?is)\A.*?{LT}\s*/\s*(think|thinking)\s*{GT}"),
    ]:
        s, n = pat.subn("", s)
        if n:
            break
    # Remove any remaining full blocks
    for pat in [
        re.compile(r"(?is)<\s*(think|thinking)\s*>.*?<\s*/\s*\1\s*>"),
        re.compile(rf"(?is){LT}\s*(think|thinking)\s*{GT}.*?{LT}\s*/\s*\1\s*{GT}"),
        re.compile(r"(?is)```[ \t]*(think|thinking)[^\n]*\n.*?\n```"),
    ]:
        s = pat.sub("", s)
    # Remove stray tags
    for pat in [
        re.compile(r"(?is)<\s*/?(?:think|thinking)\s*>"),
        re.compile(rf"(?is){LT}\s*/?(?:think|thinking)\s*{GT}"),
    ]:
        s = pat.sub("", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()

def parse_md(md_path: Path, images_path: Path, exclude_sections: List[str],
             max_images: Optional[int] = None) -> Dict[str, Any]:
    """
    Docling-generated Markdown -> {"sections": [{"title","text","images":[{"path":...}]}]}
    Splits on ## headers, filters excluded sections, and collects image paths from
    the artifacts folder in order of appearance.
    """
    if not md_path.exists():
        raise FileNotFoundError(f"Missing preprocessed MD: {md_path}")

    text = md_path.read_text(encoding="utf-8")

    def is_excluded_title(title: str) -> bool:
        excl_norm = [
            ' '.join(((s or '').lower().strip().rstrip(':.')).split())
            for s in (exclude_sections or []) if s
        ]
        t = ' '.join(((title or '').lower().strip().rstrip(':.')).split())
        return any(
            t.startswith(e) or e.startswith(t)   # handles singular/plural both ways
            for e in excl_norm
        )

    # Split into blocks by any markdown header level
    header_pattern = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
    splits = list(header_pattern.finditer(text))

    out_sections: List[Dict[str, Any]] = []

    # Content before the first header goes into a default section
    if splits:
        pre = text[:splits[0].start()].strip()
        if pre:
            out_sections.append({"title": "Document", "text": pre, "images": []})
    else:
        # No headers at all — whole doc is one section
        out_sections.append({"title": "Document", "text": text.strip(), "images": []})
        return {"sections": out_sections}

    for i, match in enumerate(splits):
        title = match.group(1).strip()
        if is_excluded_title(title):
            continue
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        body = text[start:end].strip()

        # Extract image references from the body: ![...](path)
        img_pattern = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
        images = []
        for img_match in img_pattern.finditer(body):
            raw_path = img_match.group(1).strip().replace("\\", "/")
            cand = (md_path.parent / raw_path).resolve()
            if cand.exists():
                images.append({"path": str(cand)})
            else:
                cand2 = images_path / Path(raw_path).name
                if cand2.exists():
                    images.append({"path": str(cand2)})

        # Remove image markdown syntax from body text so the LLM doesn't see raw paths
        clean_body = img_pattern.sub('', body).strip()

        out_sections.append({"title": title, "text": clean_body, "images": images})

    # Apply max_images limit across all sections
    if max_images is not None:
        total = 0
        for sec in out_sections:
            allowed = max(0, max_images - total)
            sec["images"] = sec["images"][:allowed]
            total += len(sec["images"])

    return {"sections": out_sections}
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- LLM Setup --------------------------------------------------------------------------------------------------------#####
def http_post(path: str, payload: dict) -> dict:
    url = VLLM_BASE_URL.rstrip("/") + path
    r = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=VLLM_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()

def load_model(modelID: str):
    """
    We are using an already-running vLLM OpenAI-compatible server (vllm serve ...),
    so 'load_model' just returns a lightweight handle/config.
    """
    return {"model": modelID}

def prompt_model(processor, llm, system: str, context_array, max_tokens: int, temperature: float):
    """
    Calls the vLLM OpenAI server at /v1/chat/completions.
    Builds OpenAI-style messages with image_url content (data URLs).
    """
    content = []
    loggable_content = []

    for x in context_array:
        if x.get("type") == "text":
            txt = x.get("text", "")
            content.append({"type": "text", "text": txt})
            loggable_content.append({"type": "text", "text": txt})

        elif x.get("type") == "image" and x.get("path"):
            path = x["path"]

            # vLLM OpenAI server accepts http(s) URLs or data: URLs.
            # Local file paths should be converted to data URLs.
            data_url = image_to_dataUrl(path, max_bytes=VAR_maxImageSize)
            if not data_url:
                print(f"[warn] Could not encode image under {VAR_maxImageSize} bytes: {path}. Skipped.")
                continue

            content.append({"type": "image_url", "image_url": {"url": data_url}})
            loggable_content.append({"type": "image", "path": path})

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    loggable_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": loggable_content},
    ]

    payload = {
        "model": (llm or {}).get("model", VAR_model),
        "messages": messages,
        "temperature": float(temperature) if temperature else 0.0,
        "max_tokens": int(max_tokens),
    }

    data = http_post("/v1/chat/completions", payload)
    out_text = (data["choices"][0]["message"]["content"] or "").strip()

    prompt_dump = {
        "messages": loggable_messages,
        "rendered_chat_template": None,  # server-side; we don't have it here
        "token_count": None,
        "max_new_tokens": int(max_tokens),
        "temperature": float(temperature) if temperature else 0.0,
        "raw_response": data,  # useful for debugging
    }

    return out_text, prompt_dump
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- HTML & JSON stuff ------------------------------------------------------------------------------------------------#####
def escape_html(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))

def write_html_report(path: Path, bundle: Dict[str, Any]) -> None:
    title = f"Review – {bundle.get('pdfname','')}"
    parts = [
        "<!doctype html>",
        "<html><head>",
        f"<meta charset='utf-8'><title>{escape_html(title)}</title>",
        "<style>",
        "body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.5;margin:40px;color:#111}",
        "h1{font-size:1.6rem;margin-bottom:0.5rem}",
        "h2{font-size:1.1rem;margin-top:1.6rem;margin-bottom:0.3rem;color:#333}",
        ".q{border:1px solid #e5e7eb;border-radius:10px;padding:14px;margin:14px 0;background:#fafafa}",
        ".qh{font-weight:600;color:#111;margin-bottom:0.3rem}",
        ".qt{color:#444;margin-bottom:0.6rem}",
        ".ans{white-space:pre-wrap}",
        "</style>",
        "</head><body>",
        f"<h1>{escape_html(title)}</h1>",
    ]
    for q in bundle.get("questions", []):
        header = q["question_header"]
        question = q["question"]
        answer = q["final_answer"]
        parts += [
            "<div class='q'>",
            f"<div class='qh'>{escape_html(header)}</div>",
            f"<div class='qt'>{escape_html(question)}</div>",
            f"<div class='ans'>{escape_html(answer)}</div>",
            "</div>"
        ]
    parts += ["</body></html>"]
    path.write_text("\n".join(parts), encoding="utf-8")
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Main -------------------------------------------------------------------------------------------------------------#####
def run_review(pdfname: str, processor=None, model=None):
    debug = True
    outdir = PATH_output / pdfname
    outdir.mkdir(parents=True, exist_ok=True)
    mdpath = PATH_input / pdfname / f"{pdfname}.md"
    ipath  = PATH_input / pdfname / f"{pdfname}_artifacts"

    parsed = parse_md(mdpath, ipath, VAR_excludedSections, max_images=VAR_maxImages if VAR_maxImages > 0 else None)
    if debug:
        print(f"\nParsed json:\n\n{parsed}")

    #Build context (sections + images)
    context_base = []
    sections = parsed["sections"]
    imgs_used = 0
    images_total = sum(len(s.get("images", [])) for s in sections)
    warned = False

    for s in sections:
        context_base.append({"type": "text", "text": f"# {s['title']}\n{s['text']}"})
        for im in s.get("images", []):
            context_base.append({"type": "image", "path": im["path"]})
            imgs_used += 1

    if VAR_maxImages > 0 and imgs_used == VAR_maxImages:
        print(f"[info] Image limit reached: including {imgs_used} of {images_total} total images.")

    if model is None:
        model = load_model(VAR_model)

    system_prompt = build_system_prompt() 
    
    context = list(context_base)
    context.append({"type": "text", "text": PROMPT_reviewInstruction})
    context.append({"type": "text", "text":
        "Now write the full reviewer report in natural flowing prose, as a human expert would. Do not use labeled section headers. Mentally consider all the elements above (validity, significance, methodology, suggested improvements, etc.) but weave them naturally into your writing rather than addressing each as a separate heading."
    })         

    answer_raw, prompt_dump = prompt_model(
        processor=processor,
        llm=model,
        system=system_prompt,
        context_array=context,
        max_tokens=VAR_tokensPerAnswer,
        temperature=VAR_temperature,
    )
    answer = strip_thinking(answer_raw)

    #Save thinking trace
    (outdir / "final_review_thinking.txt").write_text(answer_raw, encoding="utf-8")

    #Save artifacts
    rec = {
        "pdfname": pdfname,
        "model": VAR_model,
        "final_answer": answer,
        "sections_used": [s["title"] for s in sections],
        "images_used": imgs_used,
        "prompt": {
            "system": system_prompt,
            "messages": prompt_dump.get("messages"),
            "rendered_chat_template": prompt_dump.get("rendered_chat_template"),
            "token_count": prompt_dump.get("token_count"),
            "max_new_tokens": prompt_dump.get("max_new_tokens"),
            "temperature": prompt_dump.get("temperature"),
        },
        "context_for_llm": context,
    }

    (outdir / "final_review.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    html = {
        "pdfname": f"{pdfname}  |  {VAR_model}",
        "questions": [
            {
                "question_header": "Single-pass reviewer report (Nature-style).",
                "question": PROMPT_reviewInstruction,
                "final_answer": answer,
            }
        ],
    }
    write_html_report(outdir / "final_review.html", html)

    #Plain text for convenience
    (outdir / "final_review.txt").write_text(answer or "", encoding="utf-8")

    print(f"Wrote: {outdir/'final_review.json'}, {outdir/'final_review.html'}, {outdir/'final_review.txt'}")
#####------------------------------------------------------------------------------------------------------------------------#####



if __name__ == "__main__":
    processor = None
    model = load_model(VAR_model)
    pdfnames = ["s41467-025-61040-5", "s41467-025-64681-8", "s41467-025-67178-6", "s41467-025-61599-z", "s41467-025-58872-6", "s41467-025-61684-3", "s41467-025-64574-w", "s41467-025-66644-5", "s41467-024-55146-5", "s41467-024-53742-z", "s41467-025-63846-9", "s41467-025-64894-x", "s41467-025-57473-7", "s41467-024-53457-1", "s41467-024-49798-6", "s41467-025-66049-4", "s41467-024-51669-z", "s41467-024-53287-1", "s41467-025-64227-y", "s41467-025-66620-z", "s41467-025-66507-z", "s41467-025-56625-z", "s41467-025-57426-0", "s41467-025-67791-5", "s41467-025-61309-9", "s41467-025-55920-z", "s41467-024-53768-3", "s41467-025-63160-4", "s41467-025-60989-7", "s41467-025-64105-7", "s41467-025-61754-6"]
    
    for pdfname in pdfnames:
        try:
            print(f"\nRunning review for {pdfname}.")
            run_review(pdfname=pdfname, processor=processor, model=model)
        except torch.cuda.OutOfMemoryError as e:
            # log and recover VRAM, then continue with next paper
            outdir = PATH_output / pdfname
            outdir.mkdir(parents=True, exist_ok=True)
            msg = f"[OOM] {pdfname}: {e}\n{traceback.format_exc()}"
            print(msg)
            (outdir / "error.log").write_text(msg, encoding="utf-8")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            # log generic error and continue
            outdir = PATH_output / pdfname
            outdir.mkdir(parents=True, exist_ok=True)
            msg = f"[error] {pdfname}: {e}\n{traceback.format_exc()}"
            print(msg)
            (outdir / "error.log").write_text(msg, encoding="utf-8")
            continue
    print("\nDONE.")