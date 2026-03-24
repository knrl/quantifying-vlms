"""
taxonomy_chart.py
==================
Horizontal stacked bar chart showing the GPT-4o error-taxonomy label
distribution across VQAv2 and COCO Captions (n=200 sampled failures each).

Output: taxonomy_distribution.png  (300 DPI)

Usage:
    python3 taxonomy_chart.py            # uses real llm_judge_labels.json
    python3 taxonomy_chart.py --dummy    # uses hard-coded illustrative values
"""

import sys
import json
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── colour palette (colorblind-friendly — Wong 2011) ─────────────────────────
PALETTE = {
    "Object Blindness":       "#0072B2",   # blue
    "Semantic Drift":         "#E69F00",   # orange
    "Prior Bias":             "#009E73",   # green
    "Spatial Error":          "#CC79A7",   # pink/purple
    "Other / Unclassifiable": "#999999",   # grey
}

CAT_ORDER = [
    "Object Blindness",
    "Semantic Drift",
    "Prior Bias",
    "Spatial Error",
    "Other / Unclassifiable",
]

CAT_MAP = {
    "A": "Object Blindness",
    "B": "Semantic Drift",
    "C": "Prior Bias",
    "D": "Spatial Error",
    "E": "Other / Unclassifiable",
}

# ── data loading ──────────────────────────────────────────────────────────────
def load_real() -> pd.DataFrame:
    labels = json.load(open("results/llm_judge_labels.json"))
    rows = {}
    for ds_key, ds_label in [("vqav2", "VQA v2"), ("coco_captions", "MS-COCO Captions")]:
        subset = [e for e in labels if e["dataset"] == ds_key]
        total  = len(subset)
        c = Counter(CAT_MAP.get(e["category"], "Other / Unclassifiable") for e in subset)
        rows[ds_label] = {cat: round(100 * c.get(cat, 0) / total, 1) for cat in CAT_ORDER}
    df = pd.DataFrame(rows).T          # index = dataset label
    df = df[CAT_ORDER]
    return df


def load_dummy() -> pd.DataFrame:
    """Illustrative values for layout testing (each row sums to 100)."""
    data = {
        "Object Blindness":       [65.0, 55.0],
        "Semantic Drift":         [18.0, 25.0],
        "Prior Bias":             [10.0, 12.0],
        "Spatial Error":          [ 4.0,  5.0],
        "Other / Unclassifiable": [ 3.0,  3.0],
    }
    return pd.DataFrame(data, index=["MS-COCO Captions", "VQA v2"])


# ── plotting ──────────────────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, title_suffix: str = "") -> plt.Figure:
    colors = [PALETTE[c] for c in df.columns]

    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=300)

    # stacked horizontal bars
    lefts = np.zeros(len(df))
    bars_per_col = []
    for col, color in zip(df.columns, colors):
        vals = df[col].values.astype(float)
        b = ax.barh(
            df.index, vals, left=lefts,
            color=color, label=col,
            height=0.52, edgecolor="white", linewidth=0.6,
        )
        bars_per_col.append((b, vals, lefts.copy()))
        lefts += vals

    # percentage labels — only draw if segment ≥ 3 % (avoid clutter)
    for b, vals, left_offsets in bars_per_col:
        for rect, val, left in zip(b.patches, vals, left_offsets):
            if val >= 3.0:
                cx = left + val / 2
                cy = rect.get_y() + rect.get_height() / 2
                ax.text(
                    cx, cy, f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white",
                )

    # axes styling
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlabel("Share of labelled failures (%)", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)

    # remove top / right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # grid lines (subtle)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#cccccc")
    ax.set_axisbelow(True)

    # legend at the top
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=len(df.columns),
        fontsize=8,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    fig.suptitle(
        f"Error Taxonomy Distribution (GPT-4o labels, n=200 per dataset{title_suffix})",
        fontsize=9.5, y=1.22, va="top",
    )

    fig.tight_layout()
    return fig


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    force_dummy = "--dummy" in sys.argv
    if force_dummy:
        df = load_dummy()
        suffix = " — illustrative"
        print("Using dummy data.")
    else:
        df = load_real()
        suffix = ""
        print("Using real llm_judge_labels.json data.")

    print(df.to_string())

    fig = build_chart(df, title_suffix=suffix)
    out = "figures/taxonomy_distribution.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
