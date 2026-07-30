"""
LLM-as-a-judge: evaluate ICLR reviews (human + generated).
"""

#####----- Imports --------------------------------------------------------------------------------------#####
from pathlib import Path
from evaluator_llmjudge import load_reviews_from_iclr_csv, judge_reviews
#####----------------------------------------------------------------------------------------------------#####



#####----- Variables ------------------------------------------------------------------------------------#####
MODEL = "gpt-5"
FEWSHOT = False
DATASET = "ICLR26_rndm50"   #TODO: Change to dataset you want
SKIP_EXISTING = True # False

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parents[1] / "data"

GENERATED_ROOT = DATA_ROOT / "reviews_generated" / "ICLR"
#####----------------------------------------------------------------------------------------------------#####



#####----- Run ------------------------------------------------------------------------------------------#####
reviews = load_reviews_from_iclr_csv(
    csv_path=DATA_ROOT / "final_datasets" / DATASET / "dataset.csv",
    generated_dirs={
        "llama": GENERATED_ROOT / "llama",
        "qwen":  GENERATED_ROOT / "qwen",
        "gpt":   GENERATED_ROOT / "gpt",
    },
)

judge_reviews(
    reviews=reviews,
    papers_dir=DATA_ROOT / "processed_pdfs" / "ICLR",
    output_dir=DATA_ROOT / "llmjudge_evals" / MODEL / DATASET,
    model=MODEL,
    fewshot=FEWSHOT,
    skip_existing=SKIP_EXISTING,
)
#####----------------------------------------------------------------------------------------------------#####