"""
qualitative_grid.py
====================
Generates failure_grid.png — a 3×4 publication-quality grid showing
three representative VLM failure examples side-by-side.

Columns: Image | Ground Truth | Qwen-7B output | SmolVLM2-0.5B output
Rows:    Object Blindness (VQAv2) | Object Blindness (COCO) | Spatial Error (VQAv2)

Images are loaded from pre-saved local PNGs (img_vqav2_*.png, img_coco_*.png).
If a local file is missing the script falls back to streaming from HuggingFace,
and finally to a grey placeholder so the layout can always be inspected.

Usage:
    python3 qualitative_grid.py            # uses real images
    python3 qualitative_grid.py --dummy    # forces dummy images (layout test)
"""

import sys
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── palette ──────────────────────────────────────────────────────────────────
COL_GT   = "#1f77b4"   # blue   – ground truth
COL_QWEN = "#2ca02c"   # green  – Qwen-7B
COL_SMOL = "#d62728"   # red    – SmolVLM2-0.5B
COL_BG   = "#f7f7f7"   # light grey cell background

# ── data ─────────────────────────────────────────────────────────────────────
ROWS = [
    {
        "label":    "Object Blindness\n(VQAv2 Q-ID 262509000)",
        "dataset":  "vqav2",
        "image_id": "262509000",
        "question": "Are the boats seaworthy?",
        "gt":       "yes",
        "qwen":     "It is not possible to definitively determine from the image "
                    "alone whether the boat in the foreground is seaworthy. However, "
                    "based on its appearance and the context of the photo, it seems "
                    "to be a small vessel.",
        "smol":     "The boats are not seaworthy, as they are not designed for "
                    "such conditions.",
    },
    {
        "label":    "Object Blindness\n(COCO val 517069)",
        "dataset":  "coco_captions",
        "image_id": "COCO_val2014_000000517069.jpg",
        "question": "Generate a short caption for this image.",
        "gt":       "Two women waiting at a bench next to a street.",
        "qwen":     "Two individuals wait at a bus stop in an urban setting with "
                    "modern buildings and palm trees in the background.",
        "smol":     "A woman is sitting on a bench and talking to another woman.",
    },
    {
        "label":    "Spatial Error\n(VQAv2 Q-ID 262161012)",
        "dataset":  "vqav2",
        "image_id": "262161012",
        "question": "How many wheels do you see?",
        "gt":       "6",
        "qwen":     "In the image, there are two wheels visible on the motorcycle "
                    "in the foreground.",
        "smol":     "There are two wheels visible in the image.",
    },
]

# ── image loading ─────────────────────────────────────────────────────────────
def _dummy_image(seed: int = 0) -> np.ndarray:
    """180×240 grey placeholder with a diagonal stripe."""
    rng = np.random.default_rng(seed)
    img = np.full((180, 240, 3), 200, dtype=np.uint8)
    for i in range(min(img.shape[:2])):
        img[i, min(i, img.shape[1]-1)] = [120, 120, 120]
    return img


# ── local file lookup map ─────────────────────────────────────────────────────
# These files are saved by the fetch step (python3 qualitative_grid.py --fetch)
# or produced as a side-effect of earlier pipeline runs.
LOCAL_IMAGE_FILES = {
    ("vqav2",         "262509000"):                      "figures/img_vqav2_262509000.png",
    ("vqav2",         "262161012"):                      "figures/img_vqav2_262161012.png",
    ("coco_captions", "COCO_val2014_000000517069.jpg"):  "figures/img_coco_517069.png",
}


def _load_local(dataset: str, image_id: str) -> "np.ndarray | None":
    key  = (dataset, image_id)
    path = LOCAL_IMAGE_FILES.get(key)
    if path and Path(path).exists():
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return np.array(img)
    return None


def _fetch_vqav2(image_id: str) -> "np.ndarray | None":
    try:
        from datasets import load_dataset
        target = int(image_id)
        ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
        for sample in ds:
            if sample.get("question_id") == target:
                return np.array(sample["image"].convert("RGB"))
    except Exception as exc:
        print(f"[warn] vqav2 HF fetch failed: {exc}")
    return None


def _fetch_coco(image_id: str) -> "np.ndarray | None":
    """image_id is the filename e.g. COCO_val2014_000000517069.jpg"""
    try:
        from datasets import load_dataset
        numeric_id = int(image_id.replace("COCO_val2014_", "").replace(".jpg", ""))
        ds = load_dataset("lmms-lab/COCO-Caption", split="val", streaming=True)
        for sample in ds:
            sid = sample.get("id") or sample.get("image_id") or 0
            if int(sid) == numeric_id:
                return np.array(sample["image"].convert("RGB"))
    except Exception as exc:
        print(f"[warn] coco HF fetch failed: {exc}")
    return None


def load_image(row: dict, force_dummy: bool = False) -> np.ndarray:
    idx = ROWS.index(row)
    if force_dummy:
        return _dummy_image(idx)
    ds  = row["dataset"]
    iid = row["image_id"]

    # 1. try local file first (fast, no network)
    img = _load_local(ds, iid)
    if img is not None:
        print(f"[info] loaded local image for {ds}/{iid}")
        return img

    # 2. fallback: stream from HuggingFace
    print(f"[info] local file not found — fetching {ds}/{iid} from HF…")
    if ds == "vqav2":
        img = _fetch_vqav2(iid)
    elif ds == "coco_captions":
        img = _fetch_coco(iid)

    # 3. last resort: grey placeholder
    if img is None:
        print(f"[warn] using dummy image for {ds}/{iid}")
        img = _dummy_image(idx)
    return img


# ── text rendering helper ─────────────────────────────────────────────────────
def render_text_cell(ax, header: str, body: str, color: str, wrap_width: int = 32):
    """Fill an axes with a coloured header line and wrapped body text."""
    ax.set_facecolor(COL_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # header (bold, coloured)
    ax.text(
        0.05, 0.97, header,
        transform=ax.transAxes,
        fontsize=7, fontweight="bold", color=color,
        va="top", ha="left",
    )
    # body (wrapped)
    wrapped = "\n".join(textwrap.wrap(body, width=wrap_width))
    ax.text(
        0.05, 0.82, wrapped,
        transform=ax.transAxes,
        fontsize=6.5, color="#222222",
        va="top", ha="left",
        linespacing=1.4,
    )


# ── main figure ───────────────────────────────────────────────────────────────
def build_figure(force_dummy: bool = False) -> plt.Figure:
    n_rows = len(ROWS)
    # column widths: image gets 1.4× the text columns
    fig = plt.figure(figsize=(14, 3.4 * n_rows), dpi=300)

    # outer left strip for row labels + 4 content columns
    gs = GridSpec(
        n_rows, 5,
        figure=fig,
        width_ratios=[0.55, 1.4, 1.35, 1.35, 1.35],
        hspace=0.08,
        wspace=0.04,
        left=0.01, right=0.99, top=0.93, bottom=0.03,
    )

    # ── column header row ────────────────────────────────────────────────────
    col_headers = ["", "Image", "Ground Truth", "Qwen2.5-VL-7B", "SmolVLM2-500M"]
    col_colors  = ["white", "#333333", COL_GT, COL_QWEN, COL_SMOL]
    header_ax = fig.add_axes([0.01, 0.945, 0.98, 0.048])
    header_ax.axis("off")
    xs = [0.04, 0.155, 0.385, 0.595, 0.800]
    for x, label, color in zip(xs, col_headers, col_colors):
        header_ax.text(
            x, 0.5, label,
            transform=header_ax.transAxes,
            fontsize=9, fontweight="bold", color=color,
            va="center", ha="left",
        )

    # ── rows ─────────────────────────────────────────────────────────────────
    for r_idx, row in enumerate(ROWS):
        # col 0 — row label
        ax_label = fig.add_subplot(gs[r_idx, 0])
        ax_label.axis("off")
        ax_label.text(
            0.5, 0.5, row["label"],
            transform=ax_label.transAxes,
            fontsize=7, fontweight="bold", color="#555555",
            va="center", ha="center",
            rotation=90,
            multialignment="center",
        )

        # col 1 — image
        ax_img = fig.add_subplot(gs[r_idx, 1])
        img_arr = load_image(row, force_dummy=force_dummy)
        ax_img.imshow(img_arr)
        ax_img.axis("off")
        # question as caption below image
        ax_img.set_title(
            textwrap.fill(f"Q: {row['question']}", width=34),
            fontsize=6, color="#444444", pad=3,
        )

        # col 2 — ground truth
        ax_gt = fig.add_subplot(gs[r_idx, 2])
        render_text_cell(ax_gt, "Ground Truth", row["gt"], COL_GT, wrap_width=30)

        # col 3 — qwen
        ax_qwen = fig.add_subplot(gs[r_idx, 3])
        render_text_cell(ax_qwen, "Qwen2.5-VL-7B", row["qwen"], COL_QWEN, wrap_width=30)

        # col 4 — smolvlm2
        ax_smol = fig.add_subplot(gs[r_idx, 4])
        render_text_cell(ax_smol, "SmolVLM2-500M", row["smol"], COL_SMOL, wrap_width=30)

    # ── figure title ──────────────────────────────────────────────────────────
    fig.suptitle(
        "Figure 1. Qualitative Failure Grid — Representative VLM Error Examples\n"
        "(Green = correct/near-correct  |  Red = failure  |  Blue = ground truth)",
        fontsize=9, y=0.998, va="top",
    )

    return fig


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    force_dummy = "--dummy" in sys.argv
    print(f"Building figure (dummy={force_dummy}) …")
    fig = build_figure(force_dummy=force_dummy)
    out = "figures/failure_grid.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
