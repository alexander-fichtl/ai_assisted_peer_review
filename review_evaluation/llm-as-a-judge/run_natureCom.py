"""
LLM-as-a-judge: evaluate Nature Communications reviews (human + generated).
"""

#####----- Imports --------------------------------------------------------------------------------------#####
from pathlib import Path
from evaluator_llmjudge import load_reviews_for_naturecom, judge_reviews
#####----------------------------------------------------------------------------------------------------#####



#####----- Variables ------------------------------------------------------------------------------------#####
MODEL = "gpt-5"
FEWSHOT = False
SKIP_EXISTING = True

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parents[1] / "data"
#####----------------------------------------------------------------------------------------------------#####



#####----- Run ------------------------------------------------------------------------------------------#####
reviews = load_reviews_for_naturecom(
    human_dir=DATA_ROOT / "reviews_human" / "natureCom",
    generated_dirs={
        "llama": DATA_ROOT / "reviews_generated" / "natureCom" / "llama",
        "qwen":  DATA_ROOT / "reviews_generated" / "natureCom" / "qwen",
        "gpt":   DATA_ROOT / "reviews_generated" / "natureCom" / "gpt",
    },
)

judge_reviews(
    reviews=reviews,
    papers_dir=DATA_ROOT / "processed_pdfs" / "natureCom",
    output_dir=DATA_ROOT / "llmjudge_evals" / MODEL / "natureCom",
    model=MODEL,
    fewshot=FEWSHOT,
    skip_existing=SKIP_EXISTING,
)
#####----------------------------------------------------------------------------------------------------#####