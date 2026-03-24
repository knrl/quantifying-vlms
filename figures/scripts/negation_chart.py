"""
negation_chart.py
==================
Grouped bar chart showing negation probe success rates per template,
per model, across VQAv2 and COCO Captions.

Data is read from negation_probes_summary.json.

Output: negation_chart.png  (300 DPI)

Usage:
    python3 negation_chart.py            # real data
    python3 negation_chart.py --dummy    # synthetic layout test
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── palette (Wong 2011) ───────────────────────────────────────────────────────
COL_QWEN = "#0072B2"    # blue
COL_SMOL = "#D55E00"    # vermillion

TEMPLATE_LABELS = {
    "is_not":  "is_not",
    "absent":  "absent",
    "false_yn": "false_yn",
    "counter": "counter",
}

DS_TITLES = {
    "vqav2":         "VQA v2",
    "coco_captions": "MS-COCO Captions",
}

TEMPLATE_ORDER = ["is_not", "absent", "false_yn", "counter"]

# ── data loading ──────────────────────────────────────────────────────────────
def load_real():
    with open("results/negation_probes_summary.json") as f:
        data = json.load(f)
    return data


def load_dummy():
    return {
        "vqav2": {
            "is_not":   {"qwen": {"success_rate": 0.75}, "smolvlm2": {"success_rate": 0.60}},
            "absent":   {"qwen": {"success_rate": 0.80}, "smolvlm2": {"success_rate": 0.70}},
            "false_yn": {"qwen": {"success_rate": 0.55}, "smolvlm2": {"success_rate": 0.20}},
            "counter":  {"qwen": {"success_rate": 0.90}, "smolvlm2": {"success_rate": 0.45}},
        },
        "coco_captions": {
            "is_not":   {"qwen": {"success_rate": 0.88}, "smolvlm2": {"success_rate": 0.92}},
            "absent":   {"qwen": {"success_rate": 0.91}, "smolvlm2": {"success_rate": 0.95}},
            "false_yn": {"qwen": {"success_rate": 0.70}, "smolvlm2": {"success_rate": 0.05}},
            "counter":  {"qwen": {"success_rate": 0.95}, "smolvlm2": {"success_rate": 0.85}},
        },
    }


# ── plot ──────────────────────────────────────────────────────────────────────
def build_figure(data: dict, dummy: bool = False) -> plt.Figure:
    datasets = ["vqav2", "coco_captions"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300, sharey=True)

    bar_width = 0.35
    x = np.arange(len(TEMPLATE_ORDER))

    for ax, ds in zip(axes, datasets):
        qwen_vals = [
            data[ds][t]["qwen"]["success_rate"] * 100
            for t in TEMPLATE_ORDER
        ]
        smol_vals = [
            data[ds][t]["smolvlm2"]["success_rate"] * 100
            for t in TEMPLATE_ORDER
        ]

        bars_q = ax.bar(
            x - bar_width / 2, qwen_vals,
            width=bar_width, color=COL_QWEN, label="Qwen2.5-VL-7B",
            edgecolor="white", linewidth=0.6, zorder=3,
        )
        bars_s = ax.bar(
            x + bar_width / 2, smol_vals,
            width=bar_width, color=COL_SMOL, label="SmolVLM2-500M",
            edgecolor="white", linewidth=0.6, zorder=3,
        )

        # value labels on top of each bar
        for bars, vals in [(bars_q, qwen_vals), (bars_s, smol_vals)]:
            for bar, val in zip(bars, vals):
                label = f"{val:.0f}%" if val >= 5 else ("0%" if val == 0 else f"{val:.0f}%")
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    label,
                    ha="center", va="bottom",
                    fontsize=7.5, color="#333333",
                )

        # highlight false_yn column (most discriminating template)
        ax.axvspan(
            x[2] - bar_width * 1.2, x[2] + bar_width * 1.2,
            color="#ffffcc", alpha=0.6, zorder=0,
        )
        ax.text(
            x[2], 103, "★ most\ndiscriminating",
            ha="center", va="bottom", fontsize=6.5, color="#888800",
            style="italic",
        )

        ax.set_ylim(0, 115)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [TEMPLATE_LABELS[t] for t in TEMPLATE_ORDER],
            fontsize=9,
        )
        ax.set_title(DS_TITLES[ds], fontsize=10, pad=8)
        ax.set_xlabel("Negation Template", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("Negation Success Rate (%)", fontsize=9)

    # shared legend
    handles = [
        mpatches.Patch(color=COL_QWEN, label="Qwen2.5-VL-7B"),
        mpatches.Patch(color=COL_SMOL, label="SmolVLM2-500M"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=2,
        fontsize=9,
        frameon=False,
    )

    suffix = " (dummy data)" if dummy else ""
    fig.suptitle(
        f"Negation Probe Success Rates per Template{suffix}\n"
        "(higher = better compositional negation handling)",
        fontsize=10, y=1.03,
    )
    fig.tight_layout()
    return fig


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dummy = "--dummy" in sys.argv
    if dummy:
        print("Using dummy data.")
        data = load_dummy()
    else:
        print("Using real negation_probes_summary.json.")
        data = load_real()

    fig = build_figure(data, dummy=dummy)
    out = "figures/negation_chart.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
