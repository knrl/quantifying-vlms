"""
Edge Reliability Gap in Vision–Language Models
===============================================
Scale Ablation: Qwen2.5-VL-3B FP16 vs Qwen2.5-VL-7B FP16 on Blur Robustness

This script is the second controlled ablation targeting W1 (Fundamental Design
Confound).  The main study compares two models that differ simultaneously in:
  1. Model scale       (7B vs 0.5B parameters)
  2. Numerical precision (NF4 vs FP16)
  3. Architecture      (Qwen2.5-VL vs SmolVLM2)

``precision_ablation.py`` addresses axis 2 by running Qwen2.5-VL-7B in FP16.
This script addresses axis 1 by adding Qwen2.5-VL-3B in FP16 — a model that
shares the identical Qwen2.5-VL architecture and FP16 precision as the 7B
ablation point, differing only in parameter count (3B vs 7B).

The resulting four-point design is:

  Model                    Params   Precision   Architecture
  ──────────────────────── ──────   ─────────   ─────────────────────
  SmolVLM2-500M (original)  0.5B    FP16        SmolVLM2 (SigLIP)
  Qwen2.5-VL-3B  (NEW)      3.0B    FP16        Qwen2.5-VL
  Qwen2.5-VL-7B  (ablation) 7.0B    FP16        Qwen2.5-VL
  Qwen2.5-VL-7B  (original) 7.0B    NF4         Qwen2.5-VL

Controlled comparisons:
  Qwen-7B NF4  → Qwen-7B FP16  : isolates PRECISION effect (same arch, same scale)
  Qwen-7B FP16 → Qwen-3B FP16  : isolates SCALE effect (same arch, same precision)
  Qwen-3B FP16 → SmolVLM2 FP16 : residual ARCHITECTURE + SCALE difference

This partial factorial design allows us to attribute portions of the observed
blur-sensitivity ratio ρ = 1.50 to each source of variation.

VRAM budget (RTX 5090, 32 GiB)
──────────────────────────────
  Qwen2.5-VL-3B FP16 weights : ~6.0 GiB  (3B params × 2 bytes)
  Activations / KV cache      : ~0.8 GiB  (single-sample forward passes)
  Total                       : ~7 GiB  ✔  well within 32 GiB

Execution
─────────
    # From workspace root:
    python scripts/scale_ablation.py

    # Pilot run (faster):
    python scripts/scale_ablation.py --n 10

    # Custom model ID:
    python scripts/scale_ablation.py --model Qwen/Qwen2.5-VL-3B-Instruct

Outputs
───────
    results/scale_ablation_results.json
        Structured comparison: Qwen-3B FP16 vs Qwen-7B FP16 vs SmolVLM2.

    results/scale_ablation_rows.csv
        Per-row inference results plus copied NF4/FP16 baselines.

Requirements
────────────
    pip install torch torchvision transformers accelerate \
                qwen-vl-utils pillow requests datasets tqdm
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

SEED = 42   # Must match robustness_blur.py and precision_ablation.py
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 128

DEFAULT_MODEL_3B = "Qwen/Qwen2.5-VL-3B-Instruct"

# Source artefacts
BLURRED_RESULTS_CSV   = Path("results/robustness_blurred_results.csv")
ROBUSTNESS_REPORT     = Path("results/robustness_report.json")
PRECISION_ABLATION_JSON = Path("results/precision_ablation_results.json")

# Gaussian-blur parameters — must match robustness_blur.py exactly
BLUR_KERNEL = 5
BLUR_SIGMA  = 2.0

IMAGE_CACHE_DIR = Path("image_cache")

# Output artefacts
SCALE_ABLATION_JSON = Path("results/scale_ablation_results.json")
SCALE_ABLATION_CSV  = Path("results/scale_ablation_rows.csv")

SCALE_CSV_FIELDNAMES = [
    "dataset_name", "image_id", "task_prompt", "ground_truth",
    # Qwen-3B FP16 (this run)
    "qwen3b_fp16_output_original", "qwen3b_fp16_output_blurred",
    "qwen3b_fp16_correct_original", "qwen3b_fp16_correct_blurred",
    # Qwen-7B FP16 from precision_ablation_rows.csv (if available)
    "qwen7b_fp16_output_original", "qwen7b_fp16_output_blurred",
    "qwen7b_fp16_correct_original", "qwen7b_fp16_correct_blurred",
    # Qwen-7B NF4 from robustness_blurred_results.csv
    "qwen7b_nf4_output_original",  "qwen7b_nf4_output_blurred",
    "qwen7b_nf4_correct_original", "qwen7b_nf4_correct_blurred",
    # SmolVLM2 FP16 from robustness_blurred_results.csv
    "smol_output_original", "smol_output_blurred",
    "smol_correct_original", "smol_correct_blurred",
]


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scale_ablation.log", mode="a"),
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
# Model loader
# ══════════════════════════════════════════════════════════════════════════════

def load_qwen_fp16(model_id: str):
    """
    Load any Qwen2.5-VL model in native FP16 (no quantisation).

    VRAM for 3B: ~6.0 GiB weights + ~0.8 GiB activations ≈ 7 GiB.
    VRAM for 7B: ~14.0 GiB weights + ~1.5 GiB activations ≈ 16 GiB.
    Both are feasible on RTX 5090 (32 GiB).
    """
    log.info("Loading %s in native FP16 (no quantisation) …", model_id)
    log_vram("Before FP16 load")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()

    log_vram("After FP16 load")
    param_gib = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 3
    log.info("FP16 %s weight footprint: %.2f GiB", model_id, param_gib)
    return model, processor


# ══════════════════════════════════════════════════════════════════════════════
# Inference helper (identical to precision_ablation.py)
# ══════════════════════════════════════════════════════════════════════════════

def qwen_infer(model, processor, image: Image.Image | None, prompt: str) -> str:
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
# Correctness metric (identical to precision_ablation.py)
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
# Gaussian blur (identical parameters as robustness_blur.py)
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_blur(image: Image.Image, kernel_size: int = BLUR_KERNEL, sigma: float = BLUR_SIGMA) -> Image.Image:
    ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    tensor = TF.to_tensor(image)
    blurred = TF.gaussian_blur(tensor, kernel_size=[ks, ks], sigma=[sigma, sigma])
    return TF.to_pil_image(blurred)


# ══════════════════════════════════════════════════════════════════════════════
# Image retrieval (shared cache)
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
    Shared cache with robustness_blur.py and precision_ablation.py.
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
# Load source data
# ══════════════════════════════════════════════════════════════════════════════

def load_blur_rows(n_limit: int | None = None) -> list[dict]:
    """Load the same rows used in the main blur experiment and precision ablation."""
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


def load_robustness_report() -> dict:
    if not ROBUSTNESS_REPORT.exists():
        log.warning("robustness_report.json not found.")
        return {}
    with open(ROBUSTNESS_REPORT) as fh:
        return json.load(fh)


def load_precision_ablation() -> dict:
    """Load Qwen-7B FP16 results from the precision ablation if available."""
    if not PRECISION_ABLATION_JSON.exists():
        log.warning(
            "precision_ablation_results.json not found. "
            "Run precision_ablation.py first for the full factorial comparison."
        )
        return {}
    with open(PRECISION_ABLATION_JSON) as fh:
        return json.load(fh)


def load_precision_ablation_rows() -> dict[str, dict]:
    """
    Load per-row FP16 results from precision_ablation_rows.csv into a dict
    keyed by (dataset_name, image_id) for fast lookup.
    """
    prec_csv = Path("results/precision_ablation_rows.csv")
    if not prec_csv.exists():
        log.warning("precision_ablation_rows.csv not found — FP16 7B per-row data unavailable.")
        return {}
    row_map = {}
    with open(prec_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["dataset_name"], row["image_id"])
            row_map[key] = row
    log.info("Loaded %d FP16-7B rows from precision_ablation_rows.csv", len(row_map))
    return row_map


# ══════════════════════════════════════════════════════════════════════════════
# Main ablation loop
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(rows: list[dict], model, processor, fp16_7b_rows: dict[str, dict]) -> list[dict]:
    """
    For each row from the original blur experiment:
      1. Fetch clean image.
      2. Run Qwen-3B FP16 on clean → qwen3b_fp16_original.
      3. Apply Gaussian blur.
      4. Run Qwen-3B FP16 on blurred → qwen3b_fp16_blurred.
      5. Copy Qwen-7B FP16, Qwen-7B NF4, and SmolVLM2 baselines for comparison.

    Returns augmented row dicts.
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

        # ── Qwen-3B FP16 on clean image ───────────────────────────────────────
        orig_out = safe_infer(
            model, processor, image, prompt,
            label=f"Qwen3B-FP16-orig/{dataset_name}/{image_id}",
        )
        orig_correct = is_correct(dataset_name, orig_out, ground_truth)

        # ── Gaussian blur ─────────────────────────────────────────────────────
        blurred = gaussian_blur(image)

        # ── Qwen-3B FP16 on blurred image ─────────────────────────────────────
        blur_out = safe_infer(
            model, processor, blurred, prompt,
            label=f"Qwen3B-FP16-blur/{dataset_name}/{image_id}",
        )
        blur_correct = is_correct(dataset_name, blur_out, ground_truth)

        log.info(
            "  Qwen3B-FP16 orig=%s blur=%s  |  NF4 orig=%s blur=%s  |  Smol orig=%s blur=%s",
            "✓" if orig_correct else "✗",
            "✓" if blur_correct else "✗",
            "✓" if row.get("qwen_correct_original", "True") == "True" else "✗",
            "✓" if row.get("qwen_correct_blurred",  "False") == "True" else "✗",
            "✓" if row.get("smol_correct_original", "True") == "True" else "✗",
            "✓" if row.get("smol_correct_blurred",  "False") == "True" else "✗",
        )

        # ── Fetch Qwen-7B FP16 per-row data if available ─────────────────────
        fp16_7b = fp16_7b_rows.get((dataset_name, image_id), {})

        result_rows.append({
            "dataset_name":  dataset_name,
            "image_id":      image_id,
            "task_prompt":   prompt,
            "ground_truth":  ground_truth,
            # Qwen-3B FP16 (this run)
            "qwen3b_fp16_output_original":  orig_out,
            "qwen3b_fp16_output_blurred":   blur_out,
            "qwen3b_fp16_correct_original": orig_correct,
            "qwen3b_fp16_correct_blurred":  blur_correct,
            # Qwen-7B FP16 from precision ablation
            "qwen7b_fp16_output_original":  fp16_7b.get("qwen_fp16_output_original", ""),
            "qwen7b_fp16_output_blurred":   fp16_7b.get("qwen_fp16_output_blurred",  ""),
            "qwen7b_fp16_correct_original": fp16_7b.get("qwen_fp16_correct_original", ""),
            "qwen7b_fp16_correct_blurred":  fp16_7b.get("qwen_fp16_correct_blurred",  ""),
            # Qwen-7B NF4 from original blurred results
            "qwen7b_nf4_output_original":   row.get("qwen_output_original", ""),
            "qwen7b_nf4_output_blurred":    row.get("qwen_output_blurred",  ""),
            "qwen7b_nf4_correct_original":  row.get("qwen_correct_original", ""),
            "qwen7b_nf4_correct_blurred":   row.get("qwen_correct_blurred",  ""),
            # SmolVLM2 FP16
            "smol_output_original": row.get("smol_output_original", ""),
            "smol_output_blurred":  row.get("smol_output_blurred",  ""),
            "smol_correct_original": row.get("smol_correct_original", ""),
            "smol_correct_blurred":  row.get("smol_correct_blurred",  ""),
        })

    return result_rows


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_scale_metrics(
    result_rows: list[dict],
    robustness_report: dict,
    precision_ablation: dict,
) -> dict:
    """
    Build a four-model comparison table:
      SmolVLM2 FP16 | Qwen-3B FP16 | Qwen-7B FP16 | Qwen-7B NF4

    Derives:
      ρ_precision = SmolVLM2_drop / Qwen7B_FP16_drop   (precision isolated)
      ρ_scale     = Qwen3B_FP16_drop / Qwen7B_FP16_drop (scale within arch)
      ρ_total     = SmolVLM2_drop / Qwen7B_NF4_drop     (original confounded ratio)
    """
    n = len(result_rows)
    if n == 0:
        return {"error": "No result rows — all images failed to load."}

    def _bool(v) -> bool:
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() == "true"

    # ── Qwen-3B FP16 (this run) ───────────────────────────────────────────────
    q3b_orig_acc = 100.0 * sum(_bool(r["qwen3b_fp16_correct_original"]) for r in result_rows) / n
    q3b_blur_acc = 100.0 * sum(_bool(r["qwen3b_fp16_correct_blurred"])  for r in result_rows) / n
    q3b_drop     = q3b_orig_acc - q3b_blur_acc

    # ── Qwen-7B FP16 (from precision_ablation.py) ────────────────────────────
    prec = precision_ablation.get("qwen_fp16", {})
    q7b_fp16_orig = prec.get("original_acc")
    q7b_fp16_blur = prec.get("blurred_acc")
    q7b_fp16_drop = prec.get("drop_pp")

    # ── Qwen-7B NF4 (from robustness_report.json) ────────────────────────────
    nf4_rep = robustness_report.get("overall", robustness_report)
    q7b_nf4_orig = nf4_rep.get("qwen_original_acc")
    q7b_nf4_blur = nf4_rep.get("qwen_blurred_acc")
    q7b_nf4_drop = nf4_rep.get("qwen_drop_pct")

    # ── SmolVLM2 FP16 (from robustness_report.json) ──────────────────────────
    smol_orig = nf4_rep.get("smol_original_acc")
    smol_blur = nf4_rep.get("smol_blurred_acc")
    smol_drop = nf4_rep.get("smol_drop_pct")

    # ── Ratios ────────────────────────────────────────────────────────────────
    def safe_ratio(numerator, denominator):
        if denominator is None or denominator == 0:
            return None
        if numerator is None:
            return None
        return round(numerator / denominator, 3)

    rho_total     = safe_ratio(smol_drop, q7b_nf4_drop)    # original confounded
    rho_precision = safe_ratio(smol_drop, q7b_fp16_drop)   # SmolVLM2 vs Qwen-7B FP16
    rho_scale     = safe_ratio(q3b_drop,  q7b_fp16_drop)   # Qwen-3B vs Qwen-7B (pure scale)
    rho_arch      = safe_ratio(smol_drop, q3b_drop)        # SmolVLM2 vs Qwen-3B (arch+scale residual)

    # ── Precision effect (from precision ablation) ────────────────────────────
    prec_effect = precision_ablation.get("precision_effect", {})
    fp16_minus_nf4 = prec_effect.get("fp16_drop_minus_nf4_drop_pp")
    prec_interp    = prec_effect.get("interpretation", "N/A")
    prec_severity  = prec_effect.get("confound_severity", "UNKNOWN")

    # ── Scale effect within Qwen architecture ────────────────────────────────
    scale_delta = (round(q3b_drop - q7b_fp16_drop, 1)
                   if q7b_fp16_drop is not None else None)
    if scale_delta is not None:
        if scale_delta > 2.0:
            scale_interp = (
                "Qwen-3B is MORE sensitive to blur than Qwen-7B "
                "(scale DOES contribute to robustness within the Qwen architecture)"
            )
            scale_severity = "MODERATE — scale accounts for some of the ρ signal"
        elif scale_delta < -2.0:
            scale_interp = (
                "Qwen-3B is LESS sensitive to blur than Qwen-7B "
                "(larger scale counter-intuitively more fragile; artefact possible)"
            )
            scale_severity = "UNEXPECTED — verify results before drawing conclusions"
        else:
            scale_interp = (
                "Qwen-3B and Qwen-7B have similar blur sensitivity within FP16 "
                "(scale has minimal effect on robustness within the Qwen architecture)"
            )
            scale_severity = "LOW — scale has minimal effect; ρ likely reflects architecture difference"
    else:
        scale_interp  = "Cannot determine (Qwen-7B FP16 data unavailable — run precision_ablation.py first)"
        scale_severity = "UNKNOWN"

    return {
        "ablation_type": "scale_within_qwen_architecture",
        "hardware":  "NVIDIA RTX 5090 (32 GiB VRAM)",
        "blur_params": {"kernel_size": BLUR_KERNEL, "sigma": BLUR_SIGMA},
        "n_samples": n,

        "four_model_comparison": {
            "smolvlm2_500m_fp16": {
                "params": "0.5B", "precision": "FP16", "architecture": "SmolVLM2 (SigLIP)",
                "original_acc": smol_orig, "blurred_acc": smol_blur, "drop_pp": smol_drop,
                "source": str(ROBUSTNESS_REPORT),
            },
            "qwen25vl_3b_fp16": {
                "params": "3B", "precision": "FP16", "architecture": "Qwen2.5-VL",
                "original_acc": round(q3b_orig_acc, 1),
                "blurred_acc":  round(q3b_blur_acc, 1),
                "drop_pp":      round(q3b_drop, 1),
                "source": "this run (scale_ablation.py)",
            },
            "qwen25vl_7b_fp16": {
                "params": "7B", "precision": "FP16", "architecture": "Qwen2.5-VL",
                "original_acc": q7b_fp16_orig, "blurred_acc": q7b_fp16_blur, "drop_pp": q7b_fp16_drop,
                "source": str(PRECISION_ABLATION_JSON),
            },
            "qwen25vl_7b_nf4": {
                "params": "7B", "precision": "NF4 (4-bit)", "architecture": "Qwen2.5-VL",
                "original_acc": q7b_nf4_orig, "blurred_acc": q7b_nf4_blur, "drop_pp": q7b_nf4_drop,
                "source": str(ROBUSTNESS_REPORT),
            },
        },

        "robustness_ratios": {
            "rho_total_confounded":  {"value": rho_total,
                                      "description": "SmolVLM2 / Qwen-7B NF4 (original, all confounds)"},
            "rho_precision_controlled": {"value": rho_precision,
                                         "description": "SmolVLM2 / Qwen-7B FP16 (controls precision)"},
            "rho_scale_within_arch": {"value": rho_scale,
                                      "description": "Qwen-3B FP16 / Qwen-7B FP16 (pure scale, same arch+precision)"},
            "rho_arch_residual":     {"value": rho_arch,
                                      "description": "SmolVLM2 / Qwen-3B FP16 (architecture+scale difference)"},
        },

        "precision_effect": {
            "fp16_drop_minus_nf4_drop_pp": fp16_minus_nf4,
            "interpretation": prec_interp,
            "confound_severity": prec_severity,
        },

        "scale_effect_within_qwen": {
            "qwen3b_drop_minus_qwen7b_fp16_drop_pp": scale_delta,
            "interpretation": scale_interp,
            "confound_severity": scale_severity,
        },

        "conclusion": {
            "rho_total_confounded": rho_total,
            "rho_after_precision_control": rho_precision,
            "rho_after_scale_control": rho_scale,
            "summary": (
                f"Original ρ = {rho_total} (all three confounds mixed). "
                f"Controlling precision: ρ = {rho_precision} (SmolVLM2 vs Qwen-7B FP16). "
                f"Pure scale effect within Qwen arch: ρ_scale = {rho_scale} (Qwen-3B vs 7B FP16). "
                "Remaining architecture-driven gap between SmolVLM2 and Qwen-3B: "
                f"ρ_arch = {rho_arch}."
            ),
        },
    }


def print_summary(metrics: dict):
    print("\n" + "=" * 72)
    print("SCALE ABLATION SUMMARY  (Qwen-3B FP16 vs 7B FP16)")
    print("=" * 72)
    fc = metrics.get("four_model_comparison", {})
    smol = fc.get("smolvlm2_500m_fp16", {})
    q3 = fc.get("qwen25vl_3b_fp16", {})
    q7f = fc.get("qwen25vl_7b_fp16", {})
    q7n = fc.get("qwen25vl_7b_nf4", {})
    ratios = metrics.get("robustness_ratios", {})
    scale_eff = metrics.get("scale_effect_within_qwen", {})

    print(f"\n  SmolVLM2-500M  FP16 : drop = {smol.get('drop_pp')} pp  [original baseline]")
    print(f"  Qwen2.5-VL-3B  FP16 : drop = {q3.get('drop_pp')} pp  ← NEW (this run)")
    print(f"  Qwen2.5-VL-7B  FP16 : drop = {q7f.get('drop_pp')} pp  [precision ablation]")
    print(f"  Qwen2.5-VL-7B  NF4  : drop = {q7n.get('drop_pp')} pp  [original baseline]")
    print()
    print(f"  ρ_total (confounded)        = {ratios.get('rho_total_confounded',{}).get('value')}")
    print(f"  ρ_precision_controlled      = {ratios.get('rho_precision_controlled',{}).get('value')}")
    print(f"  ρ_scale (Qwen-3B/7B FP16)   = {ratios.get('rho_scale_within_arch',{}).get('value')}")
    print(f"  ρ_arch  (SmolVLM2/Qwen-3B)  = {ratios.get('rho_arch_residual',{}).get('value')}")
    print()
    print(f"  Scale effect delta:  {scale_eff.get('qwen3b_drop_minus_qwen7b_fp16_drop_pp')} pp")
    print(f"  Interpretation:      {scale_eff.get('interpretation')}")
    print(f"  Confound severity:   {scale_eff.get('confound_severity')}")
    print("=" * 72 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CSV writer
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCALE_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SCALE_CSV_FIELDNAMES})
    log.info("Saved scale ablation CSV → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scale ablation: Qwen2.5-VL-3B FP16 vs 7B FP16 on Gaussian-blur robustness."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit to first N rows (default: use all rows).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_3B,
        help=f"HuggingFace model ID for the 3B model (default: {DEFAULT_MODEL_3B}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("SCALE ABLATION: Qwen2.5-VL-3B FP16 (scale within Qwen arch)")
    log.info("Compares against Qwen-7B FP16 (precision_ablation.py) and")
    log.info("Qwen-7B NF4 + SmolVLM2 (robustness_blur.py) baselines.")
    log.info("Device: %s", DEVICE)
    log_vram("Startup")
    log.info("=" * 60)

    # ── Load source data ──────────────────────────────────────────────────────
    blur_rows    = load_blur_rows(n_limit=args.n)
    rob_report   = load_robustness_report()
    prec_ablat   = load_precision_ablation()
    fp16_7b_rows = load_precision_ablation_rows()

    # ── Load Qwen-3B FP16 model ───────────────────────────────────────────────
    model, processor = load_qwen_fp16(args.model)

    # ── Run ablation ──────────────────────────────────────────────────────────
    t0 = time.time()
    result_rows = run_ablation(blur_rows, model, processor, fp16_7b_rows)
    elapsed = time.time() - t0
    log.info(
        "Ablation completed in %.1f s for %d rows (%.1f s/row)",
        elapsed, len(result_rows), elapsed / max(len(result_rows), 1),
    )

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_scale_metrics(result_rows, rob_report, prec_ablat)
    print_summary(metrics)

    # ── Save outputs ──────────────────────────────────────────────────────────
    SCALE_ABLATION_JSON.parent.mkdir(parents=True, exist_ok=True)
    SCALE_ABLATION_JSON.write_text(json.dumps(metrics, indent=2))
    log.info("Saved scale ablation report → %s", SCALE_ABLATION_JSON)

    write_csv(result_rows, SCALE_ABLATION_CSV)

    # ── Free VRAM ─────────────────────────────────────────────────────────────
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_vram("After cleanup")

    log.info("Done.")


if __name__ == "__main__":
    main()
