"""
robustness_curve.py
====================
Plots a grouped bar chart showing accuracy under clean (σ=0) and blurred
(σ=2) conditions for SmolVLM2-500M and Qwen2.5-VL-7B.

Only two conditions are plotted — both are directly measured from
robustness_report.json (n=30 both-correct subset).  No extrapolation is
performed.  The previous version of this script plotted σ=1,3,4,5 as linear
extrapolations from the single σ=2 data point; those points have been removed
to avoid presenting modelled values as measurements (reviewer W4).

Output: robustness_curve.png  (300 DPI)

Usage:
    python3 robustness_curve.py            # anchored to real data
    python3 robustness_curve.py --dummy    # fully synthetic for layout test
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── palette (Wong 2011 colorblind-safe) ──────────────────────────────────────
COL_QWEN = "#0072B2"   # blue
COL_SMOL = "#D55E00"   # vermillion

# ── data ──────────────────────────────────────────────────────────────────────
def load_real():
    """
    Two directly measured conditions: σ=0 (baseline 100%) and σ=2 (tested).
    Values are raw accuracy percentages (not drops).
    """
    with open("results/robustness_report.json") as f:
        r = json.load(f)

    smol_clean  = 100.0
    qwen_clean  = 100.0
    smol_blurred = smol_clean - abs(r["smol_drop_pct"])
    qwen_blurred = qwen_clean - abs(r["qwen_drop_pct"])

    return smol_clean, smol_blurred, qwen_clean, qwen_blurred, True


def load_dummy():
    return 100.0, 90.0, 100.0, 93.3, False


# ── plot ──────────────────────────────────────────────────────────────────────
def build_figure(smol_clean, smol_blurred, qwen_clean, qwen_blurred,
                 real: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    labels     = ["Clean (σ=0)", "Blurred (σ=2)"]
    smol_vals  = [smol_clean, smol_blurred]
    qwen_vals  = [qwen_clean, qwen_blurred]

    x     = np.arange(len(labels))
    width = 0.32

    bars_qwen = ax.bar(x - width / 2, qwen_vals,  width,
                       color=COL_QWEN, alpha=0.88, label="Qwen2.5-VL-7B (NF4)")
    bars_smol = ax.bar(x + width / 2, smol_vals, width,
                       color=COL_SMOL, alpha=0.88, label="SmolVLM2-500M (FP16)")

    # value labels on bars
    for bar in (*bars_qwen, *bars_smol):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    # annotate accuracy drops with arrows
    drop_smol = smol_clean - smol_blurred
    drop_qwen = qwen_clean - qwen_blurred

    for xi, drop, color in [
        (x[1] + width / 2,  drop_smol, COL_SMOL),
        (x[1] - width / 2, drop_qwen, COL_QWEN),
    ]:
        top    = 100.0
        bottom = top - drop
        ax.annotate(
            "",
            xy=(xi, bottom + 0.5), xytext=(xi, top - 0.5),
            arrowprops=dict(arrowstyle="<->", color=color, lw=1.4),
        )
        ax.text(xi + 0.04, (top + bottom) / 2,
                f"−{drop:.1f} pp", fontsize=7.5, color=color, va="center")

    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(80, 106)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False, loc="lower right")

    title = "Accuracy Under Gaussian Blur (σ=2, directly measured)"
    if not real:
        title += " — dummy data"
    ax.set_title(title, fontsize=11, pad=10)

    note = (
        f"n=30 both-correct images.  ρ = {drop_smol:.1f}/{drop_qwen:.2f} = "
        f"{drop_smol/drop_qwen:.2f}×  (95% CI [0.00, 6.00], McNemar p=1.00)."
        if real else
        "Illustrative dummy data — not from experiment."
    )
    fig.text(0.5, -0.03, note, ha="center", fontsize=7,
             color="#666666", style="italic")

    fig.tight_layout()
    return fig


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dummy = "--dummy" in sys.argv

    if dummy:
        print("Using dummy data.")
        args = load_dummy()
    else:
        print("Using real robustness_report.json.")
        args = load_real()

    smol_clean, smol_blurred, qwen_clean, qwen_blurred, real = args
    print(f"  SmolVLM2: clean={smol_clean:.1f}%  blurred={smol_blurred:.1f}%")
    print(f"  Qwen NF4: clean={qwen_clean:.1f}%  blurred={qwen_blurred:.1f}%")

    fig = build_figure(*args)
    out = "figures/robustness_curve.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
