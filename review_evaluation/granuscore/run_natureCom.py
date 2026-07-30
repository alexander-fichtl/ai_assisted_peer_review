"""
GranuScore comparison: Nature Communications (generated vs human).
"""


#####----- Imports ----------------------------------------------------------------------------------------------------------#####
from pathlib import Path
from evaluator_granuscore import (
    load_naturecom_generated,
    load_naturecom_human,
    run_comparison,
)
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Variables --------------------------------------------------------------------------------------------------------#####
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parents[1] / "data"

DATA_DIR = DATA_ROOT / "reviews_generated" / "natureCom"
HUMAN_DIR = DATA_ROOT / "reviews_human" / "natureCom"
PLOTS = BASE_DIR / "plots" / "natureCom"
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Run --------------------------------------------------------------------------------------------------------------#####
print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", DATA_DIR)
print("HUMAN_DIR:", HUMAN_DIR)
print("PLOTS:", PLOTS)

gen_qwen = load_naturecom_generated("qwen", DATA_DIR)
gen_gpt = load_naturecom_generated("gpt", DATA_DIR)
gen_llama = load_naturecom_generated("llama", DATA_DIR)
human = load_naturecom_human(HUMAN_DIR)

print(f"Counts: qwen={len(gen_qwen)}, gpt={len(gen_gpt)}, llama={len(gen_llama)}, human={len(human)}")

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