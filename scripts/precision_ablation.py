"""
Edge Reliability Gap in Vision–Language Models
===============================================
Precision Ablation: Qwen2.5-VL-7B FP16 vs NF4 on Blur Robustness

This script directly addresses the three-variable confound in the main study
(scale, precision, architecture). It re-runs the Gaussian-blur robustness
experiment from robustness_blur.py with Qwen2.5-VL-7B loaded in native FP16
(no bitsandbytes quantisation) rather than 4-bit NF4.

Motivation
----------
The main study compares Qwen2.5-VL-7B (NF4, 4-bit) vs SmolVLM2-500M (FP16).
A reviewer correctly noted that the observed 1.5× blur-sensitivity ratio (ρ)
could be explained by three confounded variables:
  1. Model scale       (7B vs 0.5B)
  2. Numerical precision (NF4 vs FP16)
  3. Architecture      (Qwen2.5-VL vs SmolVLM2)

Hypothesis: if NF4 quantisation acts as an implicit regulariser (by adding
weight noise), then the NF4 model may be *artificially* more robust to blur
than FP16 Qwen would be. This ablation tests that hypothesis directly.

Interpretation key
------------------
  - If FP16 Qwen drop ≈ NF4 Qwen drop  → quantisation has negligible effect;
    the robustness advantage is real and attributable to scale + architecture.
  - If FP16 Qwen drop > NF4 Qwen drop  → NF4 regularisation was inflating the
    robustness gap; the reported ρ overestimates the true architecture effect.
  - If FP16 Qwen drop < NF4 Qwen drop  → NF4 was *hurting* robustness (less
    likely); FP16 is the more appropriate point of comparison.

VRAM budget (RTX 5090, 32 GiB)
-------------------------------
  Qwen2.5-VL-7B FP16 weights : ~14.0 GiB  (7B params × 2 bytes)
  Activations / KV cache      :  ~1.5 GiB  (single-sample forward passes)
  Total headroom required     : ~16 GiB  ✔  well within 32 GiB

The NF4 version used in the main study required only ~4.5 GiB. Running FP16
is therefore feasible on this hardware without any architectural changes.

Execution
---------
    # From workspace root:
    python scripts/precision_ablation.py

    # To run on a smaller pilot subset (faster sanity check):
    python scripts/precision_ablation.py --n 30

Outputs
-------
    results/precision_ablation_results.json
        JSON comparing NF4 vs FP16 Qwen blur robustness side-by-side.

    results/precision_ablation_rows.csv
        Per-row FP16 inference results for the same n=100 blur sample.

Requirements
------------
    pip install torch torchvision transformers accelerate \
                qwen-vl-utils pillow requests datasets tqdm
    # Note: bitsandbytes is NOT required for this script.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import csv
import gc
import hashlib
import io
import json
import logging
import os
import random
import re
import time
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import torch
import torchvision.transforms.functional as TF
from PIL import Image, UnidentifiedImageError
from torch.cuda.amp import autocast
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

try:
    from qwen_vl_utils import process_vision_info
    _QWEN_UTILS = True
except ImportError:
    _QWEN_UTILS = False

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit("Install `datasets`: pip install datasets") from exc


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42  # Must match robustness_blur.py to reproduce identical sample subset
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 128

# Source artefacts from the main blur experiment (never modified)
BLURRED_RESULTS_CSV  = Path("results/robustness_blurred_results.csv")
ROBUSTNESS_REPORT    = Path("results/robustness_report.json")

# Gaussian-blur parameters — must match robustness_blur.py exactly
BLUR_KERNEL = 5
BLUR_SIGMA  = 2.0

# Image cache shared with robustness_blur.py (avoids re-downloading)
IMAGE_CACHE_DIR = Path("image_cache")

# Ablation output artefacts
ABLATION_JSON = Path("results/precision_ablation_results.json")
ABLATION_CSV  = Path("results/precision_ablation_rows.csv")

ABLATION_CSV_FIELDNAMES = [
    "dataset_name", "image_id", "task_prompt", "ground_truth",
    "qwen_fp16_output_original", "qwen_fp16_output_blurred",
    "qwen_fp16_correct_original", "qwen_fp16_correct_blurred",
    # NF4 results copied from blurred_results.csv for side-by-side review
    "qwen_nf4_output_original",  "qwen_nf4_output_blurred",
    "qwen_nf4_correct_original", "qwen_nf4_correct_blurred",
]


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("precision_ablation.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# VRAM helpers
# ══════════════════════════════════════════════════════════════════════════════

def vram_allocated_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(DEVICE) / 1024 ** 2


def vram_reserved_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved(DEVICE) / 1024 ** 2


def log_vram(label: str):
    log.info(
        "%s | VRAM allocated: %.0f MiB  reserved: %.0f MiB",
        label, vram_allocated_mb(), vram_reserved_mb(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Model loader — FP16, NO quantisation
# ══════════════════════════════════════════════════════════════════════════════

def load_qwen_fp16(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    """
    Load Qwen2.5-VL-7B in native FP16 (no bitsandbytes, no quantisation).

    VRAM estimate: ~14.0 GiB for weights + ~1.5 GiB activations = ~16 GiB.
    Feasible on RTX 5090 (32 GiB); NOT feasible on 16 GiB or smaller GPUs.

    This is the ablation counterpart to the NF4 load in robustness_blur.py,
    which uses BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4').
    """
    log.info("Loading %s in native FP16 (no quantisation) …", model_id)
    log_vram("Before FP16 load")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,          # native FP16 — no BnB
        device_map={"": DEVICE},
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()

    log_vram("After FP16 load")
    log.info(
        "FP16 Qwen weight footprint estimate: %.1f GiB",
        sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 3,
    )
    return model, processor


# ══════════════════════════════════════════════════════════════════════════════
# Inference helper
# ══════════════════════════════════════════════════════════════════════════════

def qwen_infer(model, processor, image: Image.Image | None, prompt: str) -> str:
    """Run a single forward pass.  Identical logic to robustness_blur.py."""
    messages = [
        {
            "role": "user",
            "content": [
                *(
                    [{"type": "image", "image": image}]
                    if image is not None else []
                ),
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if image is not None and _QWEN_UTILS:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text_prompt],
            images=[image] if image is not None else None,
            padding=True,
            return_tensors="pt",
        )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad(), autocast(dtype=torch.float16):
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def safe_infer(model, processor, image, prompt, label: str) -> str:
    try:
        return qwen_infer(model, processor, image, prompt)
    except torch.cuda.OutOfMemoryError:
        log.warning("[OOM] %s – clearing cache", label)
        torch.cuda.empty_cache()
        gc.collect()
        return "[OOM]"
    except Exception as exc:
        log.warning("[ERR] %s – %s: %s", label, type(exc).__name__, exc)
        return f"[ERROR: {type(exc).__name__}]"


# ══════════════════════════════════════════════════════════════════════════════
# Correctness metric  (identical to robustness_blur.py)
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_correct(dataset_name: str, prediction: str, ground_truth: str) -> bool:
    pred = _normalize(prediction)
    gt   = _normalize(ground_truth)
    if not pred or not gt:
        return False
    if dataset_name == "vqav2":
        return gt in pred or pred in gt
    else:
        gt_words = [w for w in gt.split() if len(w) >= 3]
        return any(w in pred for w in gt_words)


# ══════════════════════════════════════════════════════════════════════════════
# Gaussian blur  (identical parameters as robustness_blur.py)
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_blur(image: Image.Image, kernel_size: int = BLUR_KERNEL, sigma: float = BLUR_SIGMA) -> Image.Image:
    ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    tensor = TF.to_tensor(image)
    blurred = TF.gaussian_blur(tensor, kernel_size=[ks, ks], sigma=[sigma, sigma])
    return TF.to_pil_image(blurred)


# ══════════════════════════════════════════════════════════════════════════════
# Image retrieval  (shared cache with robustness_blur.py)
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_CACHE_DIR.mkdir(exist_ok=True)

DS_MAP = {
    "vqav2":         ("lmms-lab/VQAv2",        "validation", "image", "question_id"),
    "coco_captions": ("lmms-lab/COCO-Caption",  "val",        "image", "id"),
}


def _cache_key(dataset_name: str, image_id: str) -> str:
    return hashlib.md5(f"{dataset_name}:{image_id}".encode()).hexdigest()


def _decode_image(raw) -> Image.Image | None:
    try:
        if isinstance(raw, Image.Image):
            return raw.convert("RGB")
        if isinstance(raw, (bytes, bytearray)):
            return Image.open(io.BytesIO(raw)).convert("RGB")
        if isinstance(raw, dict) and "bytes" in raw:
            return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        if hasattr(raw, "read"):
            return Image.open(raw).convert("RGB")
    except Exception:
        pass
    return None


def fetch_image(dataset_name: str, image_id: str) -> Image.Image | None:
    """
    Fetch image from disk cache first; fall back to HuggingFace streaming.
    Cache is shared with robustness_blur.py so previously cached images
    are retrieved instantly without network I/O.
    """
    cache_path = IMAGE_CACHE_DIR / f"{_cache_key(dataset_name, image_id)}.png"

    if cache_path.exists():
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            cache_path.unlink(missing_ok=True)

    if dataset_name not in DS_MAP:
        log.warning("Unknown dataset_name '%s'", dataset_name)
        return None

    hf_path, split, img_field, id_field = DS_MAP[dataset_name]
    try:
        ds = load_dataset(hf_path, split=split, streaming=True, trust_remote_code=True)
        for sample in ds:
            sid = str(sample.get(id_field, sample.get("id", sample.get("image_id", ""))))
            if sid == image_id:
                img = _decode_image(sample.get(img_field))
                if img is not None:
                    img.save(cache_path, format="PNG")
                return img
    except Exception as exc:
        log.warning("Stream error for %s id=%s: %s", dataset_name, image_id, exc)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Load original blur-experiment rows
# ══════════════════════════════════════════════════════════════════════════════

def load_blur_rows(n_limit: int | None = None) -> list[dict]:
    """
    Read the rows from robustness_blurred_results.csv — these are the exact
    both-correct samples used in the main blur experiment.  We re-run the FP16
    model on the same rows for a paired comparison.

    If n_limit is set, use only the first n_limit rows (for quick pilot runs).
    """
    if not BLURRED_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find {BLURRED_RESULTS_CSV}.\n"
            "Run robustness_blur.py first to generate the baseline CSV."
        )
    with open(BLURRED_RESULTS_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if n_limit:
        rows = rows[:n_limit]
    log.info("Loaded %d rows from %s", len(rows), BLURRED_RESULTS_CSV)
    return rows


def load_nf4_report() -> dict:
    """Load the robustness_report.json produced by robustness_blur.py."""
    if not ROBUSTNESS_REPORT.exists():
        log.warning("robustness_report.json not found – NF4 baselines will be missing.")
        return {}
    with open(ROBUSTNESS_REPORT) as fh:
        return json.load(fh)


# ══════════════════════════════════════════════════════════════════════════════
# Main ablation loop
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(rows: list[dict], model, processor) -> list[dict]:
    """
    For each row from the original blur experiment:
      1. Fetch the clean image.
      2. Run FP16 Qwen on clean image → fp16_original output.
      3. Apply Gaussian blur.
      4. Run FP16 Qwen on blurred image → fp16_blurred output.
      5. Evaluate correctness for both.

    Returns a list of augmented row dicts ready for CSV/JSON output.
    """
    result_rows = []
    n = len(rows)

    for i, row in enumerate(rows):
        dataset_name = row["dataset_name"]
        image_id     = row["image_id"]
        prompt       = row["task_prompt"]
        ground_truth = row["ground_truth"]

        log.info("[%d/%d] %s id=%s", i + 1, n, dataset_name, image_id)

        # ── Fetch clean image ─────────────────────────────────────────────────
        image = fetch_image(dataset_name, image_id)
        if image is None:
            log.warning("  Image not found – skipping row.")
            continue

        # ── FP16 inference on original image ─────────────────────────────────
        fp16_orig_out = safe_infer(
            model, processor, image, prompt,
            label=f"FP16-orig/{dataset_name}/{image_id}",
        )
        fp16_orig_correct = is_correct(dataset_name, fp16_orig_out, ground_truth)

        # ── Apply Gaussian blur ───────────────────────────────────────────────
        blurred_image = gaussian_blur(image)

        # ── FP16 inference on blurred image ──────────────────────────────────
        fp16_blur_out = safe_infer(
            model, processor, blurred_image, prompt,
            label=f"FP16-blur/{dataset_name}/{image_id}",
        )
        fp16_blur_correct = is_correct(dataset_name, fp16_blur_out, ground_truth)

        log.info(
            "  FP16 orig=%s blur=%s  |  NF4 orig=%s blur=%s",
            "✓" if fp16_orig_correct else "✗",
            "✓" if fp16_blur_correct else "✗",
            "✓" if row.get("qwen_correct_original", "True") == "True" else "✗",
            "✓" if row.get("qwen_correct_blurred",  "False") == "True" else "✗",
        )

        result_rows.append({
            "dataset_name":            dataset_name,
            "image_id":                image_id,
            "task_prompt":             prompt,
            "ground_truth":            ground_truth,
            # FP16 results
            "qwen_fp16_output_original":  fp16_orig_out,
            "qwen_fp16_output_blurred":   fp16_blur_out,
            "qwen_fp16_correct_original": fp16_orig_correct,
            "qwen_fp16_correct_blurred":  fp16_blur_correct,
            # NF4 results copied from original CSV for side-by-side comparison
            "qwen_nf4_output_original":   row.get("qwen_output_original", ""),
            "qwen_nf4_output_blurred":    row.get("qwen_output_blurred",  ""),
            "qwen_nf4_correct_original":  row.get("qwen_correct_original", ""),
            "qwen_nf4_correct_blurred":   row.get("qwen_correct_blurred",  ""),
        })

    return result_rows


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_ablation_metrics(
    result_rows: list[dict],
    nf4_report: dict,
) -> dict:
    """
    Compare FP16 Qwen blur robustness against:
      (a) NF4 Qwen from the original experiment (from robustness_report.json)
      (b) FP16 Qwen as computed here

    Returns a structured dict that can be directly serialised to JSON.
    """
    n = len(result_rows)
    if n == 0:
        return {"error": "No result rows — all images failed to load."}

    def _bool(v) -> bool:
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() == "true"

    fp16_orig_acc  = 100.0 * sum(_bool(r["qwen_fp16_correct_original"]) for r in result_rows) / n
    fp16_blur_acc  = 100.0 * sum(_bool(r["qwen_fp16_correct_blurred"])  for r in result_rows) / n
    fp16_drop      = fp16_orig_acc - fp16_blur_acc

    # ── NF4 figures from the original robustness report ───────────────────────
    # robustness_report.json uses key structure:
    #   {"overall": {"qwen_original_acc": ..., "qwen_blurred_acc": ...,
    #                "qwen_drop_pct": ..., "smol_drop_pct": ...,
    #                "relative_robustness_ratio": ...}}
    nf4_overall = nf4_report.get("overall", nf4_report)
    nf4_orig      = nf4_overall.get("qwen_original_acc",  None)
    nf4_blur      = nf4_overall.get("qwen_blurred_acc",   None)
    nf4_drop      = nf4_overall.get("qwen_drop_pct",      None)
    smol_drop     = nf4_overall.get("smol_drop_pct",      None)
    nf4_rho       = nf4_overall.get("relative_robustness_ratio", None)  # smol/nf4

    # ── Recalculate ρ using FP16 Qwen as the denominator ─────────────────────
    if fp16_drop == 0:
        fp16_rho = float("inf") if smol_drop and smol_drop > 0 else 1.0
    else:
        fp16_rho = (smol_drop / fp16_drop) if smol_drop is not None else None

    # ── Delta: how much does precision change the robustness gap? ────────────
    precision_delta = None
    if nf4_drop is not None:
        precision_delta = fp16_drop - nf4_drop   # positive → FP16 is MORE sensitive than NF4

    return {
        "ablation_type": "precision_fp16_vs_nf4",
        "model":         "Qwen/Qwen2.5-VL-7B-Instruct",
        "hardware":      "NVIDIA RTX 5090 (32 GiB VRAM)",
        "blur_params":   {"kernel_size": BLUR_KERNEL, "sigma": BLUR_SIGMA},
        "n_samples":     n,

        "qwen_fp16": {
            "precision":    "FP16",
            "original_acc": round(fp16_orig_acc, 1),
            "blurred_acc":  round(fp16_blur_acc,  1),
            "drop_pp":      round(fp16_drop,       1),
            "rho_vs_smolvlm2": round(fp16_rho, 3) if fp16_rho is not None and fp16_rho != float("inf") else str(fp16_rho),
        },

        "qwen_nf4": {
            "precision":    "NF4 (4-bit, bitsandbytes)",
            "original_acc": nf4_orig,
            "blurred_acc":  nf4_blur,
            "drop_pp":      nf4_drop,
            "rho_vs_smolvlm2": nf4_rho,
            "source":       str(ROBUSTNESS_REPORT),
        },

        "smolvlm2_fp16": {
            "drop_pp":  smol_drop,
            "source":   str(ROBUSTNESS_REPORT),
        },

        "precision_effect": {
            "fp16_drop_minus_nf4_drop_pp": round(precision_delta, 1) if precision_delta is not None else None,
            "interpretation": (
                "FP16 Qwen is MORE sensitive to blur than NF4 Qwen "
                "(quantisation appears to act as a robustness regulariser)"
                if precision_delta is not None and precision_delta > 0
                else "FP16 Qwen is LESS sensitive to blur than NF4 Qwen "
                     "(quantisation was HURTING robustness; FP16 is more robust)"
                if precision_delta is not None and precision_delta < 0
                else "Precision has negligible effect on blur sensitivity "
                     "(quantisation neither helps nor hurts robustness)"
                if precision_delta is not None
                else "Cannot determine (NF4 baseline missing)"
            ),
            "confound_severity": (
                "HIGH — reported ρ is primarily a quantisation artefact"
                if precision_delta is not None and abs(precision_delta) > 3.0
                else "MODERATE — quantisation accounts for some of the ρ signal"
                if precision_delta is not None and abs(precision_delta) > 1.0
                else "LOW — quantisation has minimal effect; ρ reflects true architecture/scale difference"
                if precision_delta is not None
                else "UNKNOWN"
            ),
        },
    }


def print_summary(metrics: dict):
    print("\n" + "=" * 70)
    print("PRECISION ABLATION SUMMARY")
    print("=" * 70)
    smol_drop = metrics.get("smolvlm2_fp16", {}).get("drop_pp")
    fp16  = metrics.get("qwen_fp16", {})
    nf4   = metrics.get("qwen_nf4",  {})
    eff   = metrics.get("precision_effect", {})

    print(f"\n  SmolVLM2-500M (FP16)  drop: {smol_drop} pp  [from main experiment]")
    print(f"\n  Qwen2.5-VL-7B (NF4)   drop: {nf4.get('drop_pp')} pp")
    print(f"    ρ vs SmolVLM2:             {nf4.get('rho_vs_smolvlm2')}")
    print(f"\n  Qwen2.5-VL-7B (FP16)  drop: {fp16.get('drop_pp')} pp   ← NEW")
    print(f"    ρ vs SmolVLM2:             {fp16.get('rho_vs_smolvlm2')}")
    print(f"\n  FP16 vs NF4 drop delta:      {eff.get('fp16_drop_minus_nf4_drop_pp')} pp")
    print(f"  Interpretation:              {eff.get('interpretation')}")
    print(f"  Confound severity:           {eff.get('confound_severity')}")
    print("=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CSV writer
# ══════════════════════════════════════════════════════════════════════════════

def write_ablation_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ABLATION_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in ABLATION_CSV_FIELDNAMES})
    log.info("Saved ablation CSV → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Precision ablation: Qwen2.5-VL-7B FP16 vs NF4 on Gaussian-blur robustness."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit to first N rows (default: use all rows from robustness_blurred_results.csv).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model ID for the 7B model (default: Qwen/Qwen2.5-VL-7B-Instruct).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("PRECISION ABLATION: Qwen2.5-VL-7B FP16 vs NF4")
    log.info("Device: %s", DEVICE)
    log_vram("Startup")
    log.info("=" * 60)

    # ── Load original blur rows ───────────────────────────────────────────────
    blur_rows = load_blur_rows(n_limit=args.n)
    nf4_report = load_nf4_report()

    # ── Load FP16 model ───────────────────────────────────────────────────────
    model, processor = load_qwen_fp16(args.model)

    # ── Run ablation ──────────────────────────────────────────────────────────
    t0 = time.time()
    result_rows = run_ablation(blur_rows, model, processor)
    elapsed = time.time() - t0
    log.info("Ablation completed in %.1f s for %d rows (%.1f s/row)",
             elapsed, len(result_rows), elapsed / max(len(result_rows), 1))

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_ablation_metrics(result_rows, nf4_report)
    print_summary(metrics)

    # ── Save outputs ──────────────────────────────────────────────────────────
    ABLATION_JSON.parent.mkdir(parents=True, exist_ok=True)
    ABLATION_JSON.write_text(json.dumps(metrics, indent=2))
    log.info("Saved ablation report → %s", ABLATION_JSON)

    write_ablation_csv(result_rows, ABLATION_CSV)

    # ── Free VRAM ─────────────────────────────────────────────────────────────
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_vram("After cleanup")

    log.info("Done.")


if __name__ == "__main__":
    main()
