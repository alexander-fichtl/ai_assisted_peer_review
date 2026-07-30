###--- Imports ----------------------------------------------------------------------------------------###
import os
from pathlib import Path

from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.document_converter import DocumentConverter, PdfFormatOption
###----------------------------------------------------------------------------------------------------###



def pdfToJson(pdfpath, paper_id, output_base="data/processed_pdfs/natureCom"):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options.lang = ["en"]
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.CPU
    )

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    conv_result = doc_converter.convert(pdfpath)

    output_dir = Path(output_base) / paper_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create an artifacts folder manually (under output_dir)
    artifacts_dir = output_dir / f"{paper_id}_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cwd = os.getcwd()
    try:
        os.chdir(output_dir)
        conv_result.document.save_as_json(
            Path(f"{paper_id}.json"),
            image_mode=ImageRefMode.REFERENCED,  #EMBEDDED or REFERENCED or PLACEHOLDER
        )
    finally:
        os.chdir(cwd)

    (output_dir / f"{paper_id}.txt").write_text(
        conv_result.document.export_to_text(), encoding="utf-8"
    )
    
    cwd = os.getcwd()
    try:
        os.chdir(output_dir)
        conv_result.document.save_as_markdown(
            Path(f"{paper_id}.md"),
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=Path(f"{paper_id}_artifacts"),
        )
    finally:
        os.chdir(cwd)

def clean_docling_id(doc_id: str, root: str = "data/processed_pdfs/natureCom") -> bool:
    """
    In-place clean for:
      {root}/{ID}/{ID}.json, {ID}.md, {ID}.txt

    Gate: Only clean if the TXT file contains ALL of "001", "002", "003".
    Removes:
      - lines that are ONLY digits (len >= 3), e.g., "000", "1234"
      - leading runs at line start:
          (a) 2+ groups of >=3-digit numbers: "3243 3244 3245  ..."
          (b) a single zero-padded group:      "000 ", "001 "
    Keeps single/double-digit lines and headings like "2021 Title".
    Returns True if cleaning was performed; otherwise False.
    """
    import re, json
    from pathlib import Path

    base = Path(root) / doc_id
    json_p = base / f"{doc_id}.json"
    md_p   = base / f"{doc_id}.md"
    txt_p  = base / f"{doc_id}.txt"

    gate_text = txt_p.read_text(encoding="utf-8", errors="ignore")
    if not all(k in gate_text for k in ("001", "002", "003")):
        return False

    WHOLE_LINE_NUM = re.compile(r'^\s*\d{3,}\s*$')
    LEADING_COUNTERS = re.compile(
        r'^(?P<prefix>\s*(?:#+\s*|>\s*|[-*+]\s*)?)'   # keep markdown/bullets
        r'(?:(?:\d{3,}\s+){2,}|0\d{2,}\s+)'           # runs or single zero-padded
    )

    def _clean_text(s: str) -> str:
        out = []
        for line in s.splitlines():
            if WHOLE_LINE_NUM.fullmatch(line):
                continue
            line = LEADING_COUNTERS.sub(r'\g<prefix>', line)
            out.append(line)
        return re.sub(r'\n{3,}', '\n\n', "\n".join(out)).strip() + "\n"

    def _clean_json(x):
        if isinstance(x, dict):
            return {k: _clean_json(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_clean_json(v) for v in x]
        if isinstance(x, str):
            return _clean_text(x)
        return x

    # Clean txt/md/json in place
    txt_p.write_text(_clean_text(gate_text), encoding="utf-8")
    md_p.write_text(_clean_text(md_p.read_text(encoding="utf-8", errors="ignore")),
                    encoding="utf-8")
    data = json.loads(json_p.read_text(encoding="utf-8"))
    json_p.write_text(json.dumps(_clean_json(data), ensure_ascii=False, indent=2),
                      encoding="utf-8")

    return True
