"""
Edge Reliability Gap in Vision–Language Models
===============================================
Robustness experiment: Gaussian-blur perturbation.

Reads vlm_inference_results.csv, filters rows where both models were
correct, randomly samples 100, re-runs inference on blurred images,
then reports accuracy drop and relative robustness ratio.

Model weights are loaded ONCE and reused for both the (already-scored)
original pass and the blurred inference pass.

Requirements
------------
    pip install torch torchvision transformers accelerate \
                bitsandbytes qwen-vl-utils pillow requests datasets
"""

# ── Standard library ──────────────────────────────────────────────────────────
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
from typing import Any

# ── Third-party ───────────────────────────────────────────────────────────────
import torch
import torchvision.transforms.functional as TF
from PIL import Image, UnidentifiedImageError
from torch.cuda.amp import autocast
from transformers import (
    SmolVLMForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
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

SEED = 42
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 128

# Input file produced by batch_inference.py  (never modified)
RESULTS_CSV = Path("results/vlm_inference_results.csv")

# How many "both-correct" rows to sample
N_SAMPLE = 100

# Gaussian-blur parameters
BLUR_KERNEL = 5        # must be odd
BLUR_SIGMA = 2.0

# Where to cache downloaded / decoded images
IMAGE_CACHE_DIR = Path("image_cache")

# Output artefacts (new files – originals are untouched)
BLURRED_RESULTS_CSV = Path("results/robustness_blurred_results.csv")
ROBUSTNESS_REPORT_JSON = Path("results/robustness_report.json")

BLURRED_CSV_FIELDNAMES = [
    "dataset_name", "image_id", "task_prompt", "ground_truth",
    "smol_output_original", "qwen_output_original",
    "smol_output_blurred",  "qwen_output_blurred",
    "smol_correct_original", "qwen_correct_original",
    "smol_correct_blurred",  "qwen_correct_blurred",
]

# ── Correctness helpers ────────────────────────────────────────────────────────
# VQA: predicted answer must contain the ground-truth token (case-insensitive)
# COCO: at least one ground-truth token (len≥3) appears in the prediction

def _normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_correct(dataset_name: str, prediction: str, ground_truth: str) -> bool:
    pred = _normalize(prediction)
    gt   = _normalize(ground_truth)
    if not pred or not gt:
        return False
    if dataset_name == "vqav2":
        # Exact-match or substring inclusion (VQA soft metric approximation)
        return gt in pred or pred in gt
    else:
        # COCO: at least one meaningful word from ground truth appears in prediction
        gt_words = [w for w in gt.split() if len(w) >= 3]
        return any(w in pred for w in gt_words)


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("robustness_blur.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Seed
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


# ══════════════════════════════════════════════════════════════════════════════
# Model loaders  (identical config to previous scripts)
# ══════════════════════════════════════════════════════════════════════════════

def load_qwen(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    log.info("Loading %s (4-bit NF4) …", model_id)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map={"": DEVICE},
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    log.info("Qwen loaded | VRAM alloc: %.0f MiB", vram_allocated_mb())
    return model, processor


def load_smolvlm(model_id: str = "HuggingFaceTB/SmolVLM2-500M-Instruct"):
    log.info("Loading %s (FP16) …", model_id)
    model = SmolVLMForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    log.info("SmolVLM2 loaded | VRAM alloc: %.0f MiB", vram_allocated_mb())
    return model, processor


# ══════════════════════════════════════════════════════════════════════════════
# Inference helpers  (identical to batch_inference.py)
# ══════════════════════════════════════════════════════════════════════════════

def _qwen_infer(model, processor, image: Image.Image | None, prompt: str) -> str:
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


def _smol_infer(model, processor, image: Image.Image | None, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                *(
                    [{"type": "image"}]
                    if image is not None else []
                ),
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=text_prompt,
        images=[image] if image is not None else None,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad(), autocast(dtype=torch.float16):
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    decoded = processor.batch_decode(out_ids, skip_special_tokens=True)
    raw = decoded[0] if decoded else ""
    if prompt in raw:
        raw = raw.split(prompt, 1)[-1].strip()
    return raw.strip()


def safe_infer(fn, model, processor, image, prompt, label: str) -> str:
    """Wrap a single inference call; never raises."""
    try:
        return fn(model, processor, image, prompt)
    except torch.cuda.OutOfMemoryError:
        log.warning("[OOM] %s – image_id context lost", label)
        torch.cuda.empty_cache()
        return "[OOM]"
    except Exception as exc:
        log.warning("[ERR] %s – %s: %s", label, type(exc).__name__, exc)
        return f"[ERROR: {type(exc).__name__}]"


# ══════════════════════════════════════════════════════════════════════════════
# Gaussian blur
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_blur(image: Image.Image, kernel_size: int = BLUR_KERNEL, sigma: float = BLUR_SIGMA) -> Image.Image:
    """Apply Gaussian blur via torchvision (CPU, no GPU memory consumed)."""
    # torchvision expects kernel_size to be odd
    ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    tensor = TF.to_tensor(image)                        # [C, H, W] float32 in [0,1]
    blurred = TF.gaussian_blur(tensor, kernel_size=[ks, ks], sigma=[sigma, sigma])
    return TF.to_pil_image(blurred)


# ══════════════════════════════════════════════════════════════════════════════
# Image retrieval with local cache
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_CACHE_DIR.mkdir(exist_ok=True)


def _cache_key(dataset_name: str, image_id: str) -> str:
    raw = f"{dataset_name}:{image_id}"
    return hashlib.md5(raw.encode()).hexdigest()


def fetch_image_for_row(row: dict) -> Image.Image | None:
    """
    Retrieve the PIL image for a result-CSV row.

    Strategy:
    1. Check local disk cache (IMAGE_CACHE_DIR/<hash>.png).
    2. Re-stream the HuggingFace dataset to find the matching image_id.
    3. Save to cache for future runs.

    Returns None if the image cannot be retrieved.
    """
    dataset_name = row["dataset_name"]
    image_id     = row["image_id"]
    cache_path   = IMAGE_CACHE_DIR / f"{_cache_key(dataset_name, image_id)}.png"

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if cache_path.exists():
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            cache_path.unlink(missing_ok=True)

    # ── Dataset config lookup ──────────────────────────────────────────────────
    DS_MAP = {
        "vqav2":          ("lmms-lab/VQAv2",         "validation", "image", "question_id"),
        "coco_captions":  ("lmms-lab/COCO-Caption",   "val",        "image", "file_name"),
    }
    if dataset_name not in DS_MAP:
        log.warning("Unknown dataset_name '%s' – cannot fetch image.", dataset_name)
        return None

    hf_path, split, img_field, id_field = DS_MAP[dataset_name]

    if img_field is None:
        # Text-only dataset – no image to fetch
        return None

    # ── Stream until we find the matching row ─────────────────────────────────
    try:
        ds = load_dataset(hf_path, split=split, streaming=True)
        for sample in ds:
            sid = str(sample.get(id_field, sample.get("id", sample.get("image_id", ""))))
            if sid == image_id:
                raw_img = sample.get(img_field)
                img = _decode_image(raw_img)
                if img is not None:
                    img.save(cache_path, format="PNG")
                return img
    except Exception as exc:
        log.warning("Failed to stream image for %s id=%s: %s", dataset_name, image_id, exc)

    return None


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


# ══════════════════════════════════════════════════════════════════════════════
# CSV helpers
# ══════════════════════════════════════════════════════════════════════════════

def read_results_csv(path: Path) -> list[dict]:
    """Read the original results CSV without modifying it."""
    if not path.exists():
        raise FileNotFoundError(
            f"Results CSV not found: {path}\n"
            "Run batch_inference.py first to generate it."
        )
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_blurred_csv(rows: list[dict], path: Path):
    """Write / append blurred results; header only if file is new."""
    is_new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=BLURRED_CSV_FIELDNAMES)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(result_rows: list[dict]) -> dict:
    """
    result_rows: list of dicts with keys *_correct_original / *_correct_blurred.
    Returns a metrics dict.
    """
    n = len(result_rows)
    if n == 0:
        return {}

    def pct(key):
        return 100.0 * sum(r[key] for r in result_rows) / n

    smol_orig  = pct("smol_correct_original")
    smol_blur  = pct("smol_correct_blurred")
    qwen_orig  = pct("qwen_correct_original")
    qwen_blur  = pct("qwen_correct_blurred")

    smol_drop  = smol_orig - smol_blur
    qwen_drop  = qwen_orig - qwen_blur

    # Relative robustness ratio: lower drop ⟹ more robust
    # If qwen_drop == 0, smol is strictly worse (or equal)
    if qwen_drop == 0:
        rel_ratio = float("inf") if smol_drop > 0 else 1.0
    else:
        rel_ratio = smol_drop / qwen_drop

    return {
        "n_samples": n,
        "smol_original_acc":  smol_orig,
        "smol_blurred_acc":   smol_blur,
        "smol_drop_pct":      smol_drop,
        "qwen_original_acc":  qwen_orig,
        "qwen_blurred_acc":   qwen_blur,
        "qwen_drop_pct":      qwen_drop,
        "relative_robustness_ratio": rel_ratio,   # smol_drop / qwen_drop
    }


def print_report(metrics: dict):
    if not metrics:
        print("No metrics to display.")
        return

    sep  = "=" * 56
    sep2 = "-" * 56

    print(f"\n{sep}")
    print("  ROBUSTNESS REPORT – Gaussian Blur Perturbation")
    print(f"  kernel={BLUR_KERNEL}×{BLUR_KERNEL}  σ={BLUR_SIGMA}  n={metrics['n_samples']}")
    print(sep)
    print(f"{'Model':<22} {'Original':>10} {'Blurred':>10} {'Drop %':>10}")
    print(sep2)
    print(
        f"{'SmolVLM2-500M (FP16)':<22} "
        f"{metrics['smol_original_acc']:>9.1f}% "
        f"{metrics['smol_blurred_acc']:>9.1f}% "
        f"{metrics['smol_drop_pct']:>+9.1f}%"
    )
    print(
        f"{'Qwen2.5-VL-7B (NF4)':<22} "
        f"{metrics['qwen_original_acc']:>9.1f}% "
        f"{metrics['qwen_blurred_acc']:>9.1f}% "
        f"{metrics['qwen_drop_pct']:>+9.1f}%"
    )
    print(sep)

    ratio = metrics["relative_robustness_ratio"]
    if ratio == float("inf"):
        ratio_str = "∞ (Qwen dropped 0 %)"
    else:
        ratio_str = f"{ratio:.3f}"

    print(f"\nRelative robustness ratio (SmolDrop / QwenDrop): {ratio_str}")
    print()

    # ── Textual interpretation ─────────────────────────────────────────────
    smol_drop = metrics["smol_drop_pct"]
    qwen_drop = metrics["qwen_drop_pct"]

    if smol_drop < qwen_drop:
        more_robust = "SmolVLM2-500M"
        less_robust = "Qwen2.5-VL-7B"
        margin = qwen_drop - smol_drop
    elif qwen_drop < smol_drop:
        more_robust = "Qwen2.5-VL-7B"
        less_robust = "SmolVLM2-500M"
        margin = smol_drop - qwen_drop
    else:
        more_robust = less_robust = None
        margin = 0.0

    print("── Interpretation ──────────────────────────────────────")
    if more_robust is None:
        print("Both models showed identical accuracy drop under blur.")
        print("Neither model has a clear robustness advantage.")
    else:
        print(
            f"{more_robust} is MORE robust to Gaussian blur, "
            f"suffering only {min(smol_drop, qwen_drop):.1f}% accuracy drop "
            f"versus {max(smol_drop, qwen_drop):.1f}% for {less_robust}."
        )
        print(
            f"The robustness margin is {margin:.1f} percentage points "
            f"(ratio = {ratio_str})."
        )
        if ratio > 1.0:
            print(
                "A ratio > 1 means SmolVLM2 is proportionally MORE sensitive "
                "to blur than Qwen2.5-VL."
            )
        elif ratio < 1.0:
            print(
                "A ratio < 1 means SmolVLM2 is proportionally LESS sensitive "
                "to blur than Qwen2.5-VL."
            )

    if abs(smol_drop) < 5 and abs(qwen_drop) < 5:
        print(
            "\nBoth models show minimal degradation (<5 pp), suggesting "
            "reasonable blur invariance at σ=2.0."
        )
    elif max(abs(smol_drop), abs(qwen_drop)) > 20:
        print(
            "\nAt least one model degrades sharply (>20 pp), indicating "
            "high sensitivity to low-frequency visual corruption."
        )
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("  Robustness Experiment – Gaussian Blur")
    log.info("=" * 60)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(DEVICE)
        log.info("GPU: %s  (%.1f GiB total)", props.name, props.total_memory / 1024 ** 3)

    # ── Step 1: Read original CSV (never modified) ────────────────────────────
    log.info("Reading original results from: %s", RESULTS_CSV)
    all_rows = read_results_csv(RESULTS_CSV)
    log.info("Total rows in CSV: %d", len(all_rows))

    # ── Step 2: Recompute correctness flags on original outputs ───────────────
    for row in all_rows:
        row["smol_correct"] = is_correct(
            row["dataset_name"], row["smol_output"], row["ground_truth"]
        )
        row["qwen_correct"] = is_correct(
            row["dataset_name"], row["qwen_output"], row["ground_truth"]
        )

    # ── Step 3: Filter rows where BOTH models were correct ────────────────────
    both_correct = [r for r in all_rows if r["smol_correct"] and r["qwen_correct"]]
    log.info("Rows where both models correct: %d", len(both_correct))

    if len(both_correct) == 0:
        log.error(
            "No rows found where both models were correct. "
            "Check that vlm_inference_results.csv contains valid outputs."
        )
        return

    # ── Step 4: Sample N_SAMPLE rows ─────────────────────────────────────────
    sample_size = min(N_SAMPLE, len(both_correct))
    sampled = random.sample(both_correct, sample_size)
    log.info("Sampled %d rows for perturbation experiment.", sample_size)

    # ── Step 5: Load models ONCE ──────────────────────────────────────────────
    try:
        qwen_model, qwen_proc = load_qwen()
    except torch.cuda.OutOfMemoryError:
        log.error("[OOM] Cannot load Qwen2.5-VL-7B – aborting.")
        return

    try:
        smol_model, smol_proc = load_smolvlm()
    except torch.cuda.OutOfMemoryError:
        log.error("[OOM] Cannot load SmolVLM2-500M – aborting.")
        return

    log.info("Both models ready | VRAM alloc: %.0f MiB", vram_allocated_mb())

    # ── Step 6: For each sampled row: fetch image, blur, re-infer ─────────────
    result_rows: list[dict] = []
    t_start = time.perf_counter()

    for idx, row in enumerate(sampled):
        dataset_name = row["dataset_name"]
        image_id     = row["image_id"]
        prompt       = row["task_prompt"]
        ground_truth = row["ground_truth"]

        log.info("[%d/%d] %s  id=%s", idx + 1, sample_size, dataset_name, image_id)

        # Fetch original image (cached locally after first fetch)
        original_image = fetch_image_for_row(row)

        if original_image is None:
            log.warning("  Could not retrieve image – skipping row.")
            continue

        # Apply Gaussian blur
        try:
            blurred_image = gaussian_blur(original_image)
        except Exception as exc:
            log.warning("  Blur failed: %s – skipping.", exc)
            continue

        # Blurred inference
        smol_out_blurred = safe_infer(_smol_infer, smol_model, smol_proc,
                                      blurred_image, prompt, f"SmolVLM2[blur idx={idx}]")
        qwen_out_blurred = safe_infer(_qwen_infer, qwen_model, qwen_proc,
                                      blurred_image, prompt, f"Qwen[blur idx={idx}]")

        result_rows.append({
            "dataset_name":          dataset_name,
            "image_id":              image_id,
            "task_prompt":           prompt,
            "ground_truth":          ground_truth,
            # Original outputs from the CSV (unchanged)
            "smol_output_original":  row["smol_output"],
            "qwen_output_original":  row["qwen_output"],
            # New blurred outputs
            "smol_output_blurred":   smol_out_blurred,
            "qwen_output_blurred":   qwen_out_blurred,
            # Correctness flags
            "smol_correct_original": row["smol_correct"],
            "qwen_correct_original": row["qwen_correct"],
            "smol_correct_blurred":  is_correct(dataset_name, smol_out_blurred, ground_truth),
            "qwen_correct_blurred":  is_correct(dataset_name, qwen_out_blurred, ground_truth),
        })

        # Flush after every sample for safety
        write_blurred_csv([result_rows[-1]], BLURRED_RESULTS_CSV)

        elapsed = time.perf_counter() - t_start
        log.info(
            "  SmolBlur=%s  QwenBlur=%s | elapsed=%.1fs | VRAM=%.0f MiB",
            result_rows[-1]["smol_correct_blurred"],
            result_rows[-1]["qwen_correct_blurred"],
            elapsed,
            vram_allocated_mb(),
        )

    # ── Step 7: Compute and print metrics ─────────────────────────────────────
    metrics = compute_metrics(result_rows)
    print_report(metrics)

    # ── Step 8: Save JSON report ──────────────────────────────────────────────
    ROBUSTNESS_REPORT_JSON.write_text(
        json.dumps(metrics, indent=2, default=str)
    )
    log.info("JSON report saved to: %s", ROBUSTNESS_REPORT_JSON)
    log.info("Blurred results CSV : %s", BLURRED_RESULTS_CSV)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del qwen_model, smol_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Done. Models unloaded.")


if __name__ == "__main__":
    main()
