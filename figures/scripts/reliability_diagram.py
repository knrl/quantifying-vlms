"""
reliability_diagram.py
=======================
Plots Reliability Diagrams (Calibration Curves) for SmolVLM2-500M and
Qwen2.5-VL-7B on VQAv2 and COCO Captions.

Real data is loaded from calibration_results.json (pre-computed bin stats).
Pass --dummy to bypass the JSON and use synthetic overconfident / well-
calibrated curves for layout testing.

Output: reliability_diagram.png  (300 DPI)

Usage:
    python3 reliability_diagram.py            # real data
    python3 reliability_diagram.py --dummy    # synthetic layout test
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── style ─────────────────────────────────────────────────────────────────────
COL_QWEN = "#0072B2"   # blue  (Wong palette)
COL_SMOL = "#D55E00"   # vermillion (Wong palette)
COL_DIAG = "#333333"

MARKERS = {
    "qwen":     ("o", "solid"),
    "smolvlm2": ("^", "solid"),
}

LABELS = {
    "qwen":     "Qwen2.5-VL-7B",
    "smolvlm2": "SmolVLM2-500M",
}

DS_LABELS = {
    "vqav2":          "VQA v2",
    "coco_captions":  "MS-COCO Captions",
}

# ── data loading ──────────────────────────────────────────────────────────────
def load_real():
    """
    Returns dict:  { (model, dataset): (conf_list, acc_list) }
    Only non-empty bins are included.
    """
    with open("results/calibration_results.json") as f:
        data = json.load(f)

    curves = {}
    for model, datasets in data.items():
        for ds, info in datasets.items():
            pts = [
                (b["avg_conf"], b["accuracy"])
                for b in info["bin_stats"] if b["count"] > 0
            ]
            if pts:
                confs, accs = zip(*pts)
                curves[(model, ds)] = (list(confs), list(accs))
    return curves


def load_dummy():
    """Synthetic curves: Qwen overconfident on VQAv2, SmolVLM2 spread out."""
    rng = np.random.default_rng(42)

    def _make_curve(conf_mean, conf_std, acc_fn, n=10):
        confs = np.clip(np.linspace(0.1, 0.95, n) + rng.normal(0, 0.02, n), 0.05, 0.99)
        accs  = np.clip(np.array([acc_fn(c) for c in confs]) + rng.normal(0, 0.04, n), 0, 1)
        return sorted(confs), [a for _, a in sorted(zip(confs, accs))]

    curves = {
        # Qwen VQAv2: all mass at 0.999 (one point, severely overconfident)
        ("qwen", "vqav2"):          ([0.9985], [0.5555]),
        # Qwen COCO: all mass at 0.998, well-calibrated
        ("qwen", "coco_captions"):  ([0.9977], [0.9105]),
        # SmolVLM2 VQAv2: spread, overconfident in middle range
        ("smolvlm2", "vqav2"):       _make_curve(0.65, 0.2, lambda c: c * 0.6 + 0.1),
        # SmolVLM2 COCO: mostly high confidence, reasonably accurate
        ("smolvlm2", "coco_captions"): _make_curve(0.45, 0.15, lambda c: 0.7 + c * 0.25),
    }
    return curves


# ── plot ──────────────────────────────────────────────────────────────────────
def build_figure(curves: dict, dummy: bool = False) -> plt.Figure:
    datasets = ["vqav2", "coco_captions"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300, sharey=True)

    for ax, ds in zip(axes, datasets):
        # perfect calibration diagonal
        ax.plot(
            [0, 1], [0, 1],
            linestyle="--", color=COL_DIAG, linewidth=1.2,
            label="Perfectly Calibrated", zorder=1,
        )

        for model, color in [("qwen", COL_QWEN), ("smolvlm2", COL_SMOL)]:
            key = (model, ds)
            if key not in curves:
                continue
            confs, accs = curves[key]
            marker, ls = MARKERS[model]
            ax.plot(
                confs, accs,
                color=color, linestyle=ls, linewidth=1.8,
                marker=marker, markersize=7, markeredgewidth=0.8,
                markeredgecolor="white", zorder=3,
                label=LABELS[model],
            )
            # annotate single-point models (both Qwen curves land on one bin)
            if len(confs) == 1:
                ax.annotate(
                    f"({confs[0]:.3f}, {accs[0]:.3f})\nall 2 000 samples\nin this bin",
                    xy=(confs[0], accs[0]),
                    xytext=(-68, 12),
                    textcoords="offset points",
                    fontsize=6.5, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
                )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlabel("Mean Predicted Confidence", fontsize=9)
        ax.set_title(DS_LABELS[ds], fontsize=10, pad=8)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("Empirical Accuracy", fontsize=9)

    # shared legend below both panels
    handles, labels_ = axes[0].get_legend_handles_labels()
    # deduplicate (both panels have the same entries)
    seen = {}
    for h, l in zip(handles, labels_):
        seen.setdefault(l, h)
    fig.legend(
        seen.values(), seen.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=9,
        frameon=False,
        handlelength=2.0,
    )

    suffix = " (illustrative dummy data)" if dummy else ""
    fig.suptitle(
        f"Reliability Diagrams — Confidence Calibration{suffix}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    return fig


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dummy = "--dummy" in sys.argv
    if dummy:
        print("Using dummy data.")
        curves = load_dummy()
    else:
        print("Using real calibration_results.json.")
        curves = load_real()

    for k, (c, a) in curves.items():
        print(f"  {k}: {len(c)} bin(s) — confs={[round(x,4) for x in c]}")

    fig = build_figure(curves, dummy=dummy)
    out = "figures/reliability_diagram.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
