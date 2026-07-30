"""
"""


#####----- Imports --------------------------------------------------------------------------------------#####
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#####----------------------------------------------------------------------------------------------------#####



#####----- Variables ------------------------------------------------------------------------------------#####
MODEL = "gpt-5"
DATASETS = ["ICLR26_rndm50", "natureCom"]
SHOW_LABELS = False
ANNOTATED_IDS = {"2L7KQ4qbHi_human_2"}   # optional
ONLY_SOURCES = None                # e.g. {"llama", "qwen"} or None for all

METRICS = [
    ("overall score", (1, 7)),
    ("understanding score", (1, 5)),
    ("coverage score", (1, 5)),
    ("evidence_support score", (1, 5)),
    ("constructiveness score", (1, 5)),
    ("conciseness score", (1, 5)),
]

###--- Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parents[1] / "data"
#####----------------------------------------------------------------------------------------------------#####



#####----- Functions ------------------------------------------------------------------------------------#####
def load_judgements(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing judgements CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"paper_id", "review_id", "human review", "source"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df["paper_id"] = df["paper_id"].astype(str)
    df["review_id"] = df["review_id"].astype(str)
    df["source"] = df["source"].astype(str).str.strip()
    df["human review"] = (
        df["human review"].astype(str).str.strip().str.lower().map(lambda x: "yes" if x == "yes" else "no")
    )

    for metric, _ in METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

    return df


def add_jitter(sub: pd.DataFrame, metric_col: str, width: float = 0.08) -> pd.DataFrame:
    sub = sub.copy()
    sub["x_plot"] = sub["x"]

    grouped = sub.groupby(["paper_id", metric_col], dropna=False)
    for _, idx in grouped.groups.items():
        idx = list(idx)
        n = len(idx)
        if n <= 1:
            continue
        offsets = np.linspace(-width, width, n)
        sub.loc[idx, "x_plot"] = sub.loc[idx[0], "x"] + offsets

    return sub


def human_summary(df: pd.DataFrame, metric_col: str, pid_to_x: dict[str, int]):
    h = df[(df["human review"] == "yes") & df[metric_col].notna()]
    if h.empty:
        return None

    xs, med, q25, q75 = [], [], [], []
    for pid in sorted(pid_to_x, key=pid_to_x.get):
        vals = h.loc[h["paper_id"] == pid, metric_col].dropna().astype(float)
        if vals.empty:
            continue
        arr = vals.to_numpy()
        xs.append(pid_to_x[pid])
        med.append(float(np.median(arr)))
        q25.append(float(np.percentile(arr, 25)))
        q75.append(float(np.percentile(arr, 75)))

    if not xs:
        return None
    return xs, med, q25, q75


def plot_metric(df: pd.DataFrame, metric_col: str, ylims: tuple[float, float], out_path: Path):
    sub = df[df[metric_col].notna()].copy()
    if sub.empty:
        print(f"Skipping {metric_col}: no data")
        return

    paper_ids = sorted(sub["paper_id"].unique())
    pid_to_x = {pid: i for i, pid in enumerate(paper_ids)}
    sub["x"] = sub["paper_id"].map(pid_to_x)
    sub = add_jitter(sub, metric_col)

    fig, ax = plt.subplots(figsize=(max(12, 0.35 * len(paper_ids) + 8), 6))

    # human band + median
    stats = human_summary(df, metric_col, pid_to_x)
    if stats is not None:
        xs, med, q25, q75 = stats
        ax.fill_between(xs, q25, q75, color="grey", alpha=0.15, label="Human IQR", zorder=1)
        ax.plot(xs, med, "--", linewidth=1.5, color="grey", alpha=0.8, label="Human median", zorder=2)

    # human points
    humans = sub[sub["human review"] == "yes"]
    if not humans.empty:
        ax.scatter(
            humans["x_plot"],
            humans[metric_col],
            marker="o",
            s=35,
            color="black",
            alpha=0.35,
            label="Human reviews",
            zorder=4,
        )

    # generated points + lines per source
    llms = sub[sub["human review"] == "no"]
    sources = sorted(llms["source"].dropna().unique())
    if ONLY_SOURCES:
        sources = [s for s in sources if s in ONLY_SOURCES]

    for source in sources:
        pts = llms[llms["source"] == source].sort_values("x")
        if pts.empty:
            continue

        ax.scatter(
            pts["x_plot"],
            pts[metric_col],
            marker="x",
            s=45,
            alpha=0.9,
            label=f"Generated: {source}",
            zorder=5,
        )
        ax.plot(
            pts["x"],
            pts[metric_col],
            linewidth=1.5,
            alpha=0.8,
            zorder=3,
        )

    # labels
    if SHOW_LABELS or ANNOTATED_IDS:
        yrange = ylims[1] - ylims[0]
        for _, row in sub.iterrows():
            rid = row["review_id"]
            if SHOW_LABELS or rid in ANNOTATED_IDS:
                ax.text(
                    row["x_plot"],
                    float(row[metric_col]) + 0.02 * yrange,
                    rid,
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=6,
                    zorder=6,
                )

    ax.set_title(f"{out_path.parent.name} — {metric_col}")
    ax.set_xlabel("paper_id")
    ax.set_ylabel(metric_col)
    ax.set_xlim(-0.5, len(paper_ids) - 0.5)
    ax.set_ylim(ylims[0] - 0.1, ylims[1] + 0.1)
    ax.set_xticks(range(len(paper_ids)))
    ax.set_xticklabels(paper_ids, rotation=90, fontsize=6)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
#####----------------------------------------------------------------------------------------------------#####



#####----- Main -----------------------------------------------------------------------------------------#####
def main():
    for dataset in DATASETS:
        csv_path = DATA_ROOT / "llmjudge_evals" / MODEL / dataset / "judgements.csv"
        out_dir = BASE_DIR / "plots" / MODEL / dataset

        df = load_judgements(csv_path)
        print(f"Loaded: {csv_path}")
        print(f"Output dir: {out_dir}")

        first20_ids = set(sorted(df["paper_id"].unique())[:20])
        df_first20 = df[df["paper_id"].isin(first20_ids)]

        for metric_col, ylims in METRICS:
            safe = metric_col.replace(" ", "_").replace("/", "-")
            plot_metric(df, metric_col, ylims, out_dir / f"{safe}.png")
            plot_metric(df_first20, metric_col, ylims, out_dir / f"{safe}_first20.png")

if __name__ == "__main__":
    main()
#####----------------------------------------------------------------------------------------------------#####