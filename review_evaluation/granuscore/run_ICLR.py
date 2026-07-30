"""
GranuScore comparison: ICLR (generated vs. human)
"""

#####----- Imports ----------------------------------------------------------------------------------------------------------#####
from pathlib import Path
from evaluator_granuscore import (
    load_openreview_generated,
    load_openreview_human_from_dataset,
    run_comparison,
)
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Variables --------------------------------------------------------------------------------------------------------#####
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parents[1] / "data"

DATASET_NAME = "ICLR26_rndm50"

HUMAN_DATASET_CSV = DATA_ROOT / "final_datasets" / DATASET_NAME / "dataset.csv"
GENERATED_ROOT = DATA_ROOT / "reviews_generated" / "ICLR"
PLOTS = BASE_DIR / "plots" / "ICLR"

#TODO: Change to the names of your generated-review folders
MODEL_FOLDERS = {
    "llama": "llama",
    "qwen": "qwen",
    "gpt": "gpt",
}
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Run --------------------------------------------------------------------------------------------------------------#####
print("HUMAN_DATASET_CSV:", HUMAN_DATASET_CSV)
print("GENERATED_ROOT:", GENERATED_ROOT)
print("PLOTS:", PLOTS)

human = load_openreview_human_from_dataset(HUMAN_DATASET_CSV)

gen_llama = load_openreview_generated(MODEL_FOLDERS["llama"], GENERATED_ROOT)
gen_qwen = load_openreview_generated(MODEL_FOLDERS["qwen"], GENERATED_ROOT)
gen_gpt = load_openreview_generated(MODEL_FOLDERS["gpt"], GENERATED_ROOT)

print(f"Counts: llama={len(gen_llama)}, qwen={len(gen_qwen)}, gpt={len(gen_gpt)}, human={len(human)}")

run_comparison(
    generated_textdicts={
        "llama": gen_llama,
        "qwen": gen_qwen,
        "gpt": gen_gpt,
    },
    human_textdict=human,
    output_dir=PLOTS,
    filter_common=False,
)
#####------------------------------------------------------------------------------------------------------------------------#####