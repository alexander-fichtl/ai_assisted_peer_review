"""
Input: JSON produced by the Docling preprocessing pipeline (document.save_as_json with ImageRefMode.REFERENCED)
- Local Transformers backend (AutoProcessor + Llama4ForConditionalGeneration)
- One pass per question over the whole paper context (selected sections + a limited amount of figures)
Outputs: per_question/*.json, final_review.json, final_review.html, audit.log
"""



#####----- Imports ----------------------------------------------------------------------------------------------------------#####
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

###--- Standard library
import base64
import copy
import gc
import json
import re
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

###--- Third-party
import numpy as np
from PIL import Image
import torch
import torch._dynamo
from transformers import AutoProcessor, Llama4ForConditionalGeneration
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
VAR_tokensPerAnswer = 2000
VAR_tokensPerAnswer_2ndPass = 2000
VAR_model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
VAR_setup = "transformers" #vLLM #quantized 
VAR_temperature = 0.6

###--- PATHS
PATH_output = Path("reviews_llama_natureCom")
PATH_input = Path("pdf_preprocessing/processed_pdfs_natureCom")   #Docling preprocessing outputs (folder with JSON + images)
PATH_imagestandards = Path("context/imagestandards.txt")
PATH_peerreview = Path("context/peerreview.txt")
PATH_researchethics = Path("context/researchethics.txt")

###--- Images
VAR_maxImages = 5 #Set to 0 for ignoring images
VAR_maxImageSize = 4_000_000   # ~4MB
IMAGE_TOKEN_ESTIMATE = 6000   # conservative per-image token cost at 1280px max side
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
                # Fallback: try just the filename in images_path
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

def log_context_tokens(processor, context_array, system):
    total = 0
    sys_tokens = len(processor.tokenizer.encode(system, add_special_tokens=False))
    total += sys_tokens
    for x in context_array:
        if x.get("type") == "text":
            n = len(processor.tokenizer.encode(x["text"], add_special_tokens=False))
            total += n
            print(f"  text '{x['text'][:60]}...': {n} tokens")
        elif x.get("type") == "image":
            print(f"  image {x['path']}: counted separately")
    print(f"  system: {sys_tokens} tokens")
    print(f"  text total (excl. images): {total} tokens")
    return total
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- LLM Setup --------------------------------------------------------------------------------------------------------######
def load_model(modelID: str) -> Tuple[AutoProcessor, Llama4ForConditionalGeneration]:
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(0)
    #attn = os.environ.get("LLAMA_ATTN_IMPL", "flex_attention")
    processor = AutoProcessor.from_pretrained(modelID, trust_remote_code=True, use_fast=True, token=HF_TOKEN, cache_dir=CACHE_DIR)
    model = Llama4ForConditionalGeneration.from_pretrained(
        modelID,
        attn_implementation="sdpa",          
        device_map="auto",
        torch_dtype=torch.bfloat16,          
        trust_remote_code=True,
        token=HF_TOKEN,
        cache_dir=CACHE_DIR,
    )
    return processor, model

def prompt_model(processor, model, system: str, context_array: List[dict],
                 max_tokens: int, temperature: float) -> Tuple[str, Dict[str, Any]]:

    # --- 1. Count system tokens and reserve output budget ---
    sys_tokens = len(processor.tokenizer.encode(system, add_special_tokens=False))
    budget = VAR_maxContext - sys_tokens - max_tokens  # reserve space for output too
    if budget <= 0:
        raise ValueError(f"System prompt ({sys_tokens} tokens) + max_tokens ({max_tokens}) already exceeds VAR_maxContext ({VAR_maxContext}).")

    # --- 2. Pass over context: measure text exactly, queue images separately ---
    text_items = []   # processed (possibly truncated) text dicts in original order
    image_items = []  # image dicts in original order

    for x in context_array:
        if x.get("type") == "text":
            txt = x.get("text", "")
            n = len(processor.tokenizer.encode(txt, add_special_tokens=False))
            if budget <= 0:
                print(f"[warn] Budget exhausted — dropping text block starting with: '{txt[:80]}'")
                text_items.append(None)  # placeholder to preserve ordering loop below
                continue
            if n > budget:
                # Truncate to fit remaining budget
                tokens = processor.tokenizer.encode(txt, add_special_tokens=False)[:budget]
                txt = processor.tokenizer.decode(tokens)
                print(f"[warn] Text truncated from {n} to {budget} tokens to fit context budget.")
                budget = 0
            else:
                budget -= n
            text_items.append({"type": "text", "text": txt})
        elif x.get("type") == "image" and x.get("path"):
            image_items.append(x)

    # --- 3. Include images only if budget allows ---
    included_image_paths = set()
    for im in image_items:
        if budget >= IMAGE_TOKEN_ESTIMATE:
            included_image_paths.add(im["path"])
            budget -= IMAGE_TOKEN_ESTIMATE
        else:
            print(f"[info] Skipping image {im['path']} — context budget exhausted ({budget} tokens remaining, need {IMAGE_TOKEN_ESTIMATE}).")

    print(f"[info] Context budget remaining after all content: {budget} tokens. Images included: {len(included_image_paths)}/{len(image_items)}.")

    # --- 4. Build final content lists, preserving original interleaving order ---
    content = []
    loggable_content = []
    text_iter = iter(text_items)

    for x in context_array:
        if x.get("type") == "text":
            item = next(text_iter)
            if item is None:
                continue  # was dropped due to budget
            content.append({"type": "text", "text": item["text"]})
            loggable_content.append({"type": "text", "text": item["text"]})
        elif x.get("type") == "image" and x.get("path"):
            if x["path"] not in included_image_paths:
                continue  # skipped due to budget
            try:
                with Image.open(x["path"]) as pil0:
                    pil = pil0.convert("RGB")
                    max_side = 1280
                    w, h = pil.size
                    if max(w, h) > max_side:
                        scale = max_side / float(max(w, h))
                        pil = pil.resize((int(w*scale), int(h*scale)))
                content.append({"type": "image", "image": pil})
            except Exception as e:
                print(f"[warn] failed to open image {x['path']}: {e}")
            loggable_content.append({"type": "image", "path": x["path"]})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user",   "content": content},
    ]
    loggable_messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user",   "content": loggable_content},
    ]

    rendered_prompt = processor.apply_chat_template(
        loggable_messages, add_generation_prompt=True, tokenize=False
    )

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    eos_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None) or getattr(model.config, "eos_token_id", None)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=bool(temperature and temperature > 0),
            temperature=float(temperature) if temperature else 0.0,
            pad_token_id=eos_id,
            eos_token_id=eos_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    out_text = processor.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)[0].strip()

    try:
        input_ids = inputs["input_ids"][0].to("cpu").tolist()
        token_count = len(input_ids)
    except Exception:
        token_count = None

    print(f"[info] Actual input token count (from model): {token_count}")

    prompt_dump = {
        "messages": loggable_messages,              # structured; images as paths
        "rendered_chat_template": rendered_prompt,  # exact string form
        "token_count": token_count,                 # actual tokens seen by model (incl. images)
        "images_included": len(included_image_paths),
        "images_skipped": len(image_items) - len(included_image_paths),
        "max_new_tokens": int(max_tokens),
        "temperature": float(temperature) if temperature else 0.0,
    }

    try:
        del inputs, outputs
        torch.cuda.empty_cache()
    except Exception:
        pass

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



#####----- 2nd Pass ---------------------------------------------------------------------------------------------------------#####
def split_headings_block(text: str) -> Dict[str, str]:
    """
    Given a markdown block:

        ### Q1 (Summary):
        answer...

        ### Q2 (Soundness):
        answer...

    return a dict: { "Q1 (Summary):": "answer...", "Q2 (Soundness):": "answer..." }.
    """
    lines = text.splitlines()
    chunks: Dict[str, List[str]] = {}
    current_header: Optional[str] = None
    buf: List[str] = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        m = re.match(r"^###\s+(Q[0-9]+\s*\([^)]*\):?)\s*$", line.strip())
        if m:
            # flush previous
            if current_header is not None:
                chunks[current_header] = "\n".join(buf).strip()
                buf = []
            current_header = m.group(1)
        else:
            if current_header is not None:
                buf.append(raw_line)

    if current_header is not None:
        chunks[current_header] = "\n".join(buf).strip()

    return {k: v for k, v in chunks.items() if v}

def tighten_full_review(
    processor,
    model,
    q_records: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Second pass over the complete review: aggressively compress,
    remove redundancy within and across answers,
    but keep ratings/flags and structure.
    """
    if not q_records:
        return q_records, {}

    # Build one markdown block with all answers
    parts: List[str] = []
    for rec in q_records:
        header = rec["question_header"]
        answer = rec["final_answer"] or ""
        parts.append(f"### {header}\n{answer.strip()}\n")

    full_review_text = "\n".join(parts).strip()

    context_text = (
        "Edit the following complete review.\n"
        "Make each answer much more concise and remove repetition both within answers "
        "and across different answers, following your rules.\n\n"
        "REVIEW START\n"
        f"{full_review_text}\n"
        "REVIEW END"
    )

    context = [{"type": "text", "text": context_text}]

    edited_text, prompt_dump = prompt_model(
        processor=processor,
        model=model,
        system=PROMPT_2ndPass_global,
        context_array=context,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    edited_text = (edited_text or "").strip()
    if not edited_text:
        # If something went wrong, keep originals.
        return q_records, prompt_dump

    # Parse edited answers back into per-question chunks
    edited_chunks = split_headings_block(edited_text)

    for rec in q_records:
        header = rec["question_header"]
        original = rec.get("final_answer", "")
        rec["raw_answer"] = original  # keep original
        new = edited_chunks.get(header)
        if new:
            rec["final_answer"] = new.strip()

    return q_records, prompt_dump

def snapshot_rating_values(q_records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    For each question_header, collect the list of numeric rating values (as strings)
    found in lines like 'Rating: <number>' in the first-pass answers.
    """
    pat = re.compile(r"Rating:\s*([0-9]+)", re.I)
    snapshot: Dict[str, List[str]] = {}
    for rec in q_records:
        header = rec.get("question_header", "")
        ans = rec.get("final_answer") or ""
        vals = pat.findall(ans)
        snapshot[header] = sorted(set(vals))
    return snapshot

def ratings_preserved(
    original_snapshot: Dict[str, List[str]],
    new_q_records: List[Dict[str, Any]],
) -> bool:
    """
    Check that for every question_header, the multiset of rating values (as numbers)
    is identical before and after tightening. If any rating disappears, changes,
    or new ratings appear where there were none, return False.
    """
    pat = re.compile(r"Rating:\s*([0-9]+)", re.I)
    for rec in new_q_records:
        header = rec.get("question_header", "")
        ans = rec.get("final_answer") or ""
        orig_vals = original_snapshot.get(header, [])
        new_vals = sorted(set(pat.findall(ans)))
        if orig_vals != new_vals:
            return False

    # Also ensure no header disappeared entirely
    new_headers = {rec.get("question_header", "") for rec in new_q_records}
    for header in original_snapshot.keys():
        if header not in new_headers:
            return False

    return True
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

    for s in sections:
        context_base.append({"type": "text", "text": f"# {s['title']}\n{s['text']}"})
        for im in s.get("images", []):
            context_base.append({"type": "image", "path": im["path"]})
            imgs_used += 1

    if VAR_maxImages > 0 and imgs_used == VAR_maxImages:
        print(f"[info] Image limit reached: including {imgs_used} of {images_total} total images.")


    if processor is None or model is None:
        processor, model = load_model(VAR_model)

    system_prompt = build_system_prompt()

    # Log text token counts before calling the model
    print(f"\n[info] Token breakdown for {pdfname}:")
    log_context_tokens(processor, context_base, system_prompt)

    context = list(context_base)
    context.append({"type": "text", "text": PROMPT_reviewInstruction})
    context.append({"type": "text", "text":
        "Now write the full reviewer report in natural flowing prose, as a human expert would. Do not use labeled section headers. Mentally consider all the elements above (validity, significance, methodology, suggested improvements, etc.) but weave them naturally into your writing rather than addressing each as a separate heading."
    })         

    answer, prompt_dump = prompt_model(
        processor=processor,
        model=model,
        system=system_prompt,
        context_array=context,
        max_tokens=VAR_tokensPerAnswer,
        temperature=VAR_temperature,
    )

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
            "token_count": prompt_dump.get("token_count"),         # actual tokens seen by model (text + images)
            "images_included": prompt_dump.get("images_included"), # images that fit in budget
            "images_skipped": prompt_dump.get("images_skipped"),   # images dropped due to budget
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

    # Cleanup intermediate objects to free memory before next paper
    del context_base, context, sections, parsed
    gc.collect()
    torch.cuda.empty_cache()
#####------------------------------------------------------------------------------------------------------------------------#####



processor, model = load_model(VAR_model)
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
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
print("\nDONE.")