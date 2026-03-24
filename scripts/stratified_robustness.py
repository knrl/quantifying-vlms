"""
stratified_robustness.py
========================
Companion to robustness_blur.py — addresses the "both-correct" bias raised
in reviewer feedback.

robustness_blur.py measures blur sensitivity only on the "both-correct"
subset (rows where *both* models were originally correct).  This filters for
the easiest images in the dataset, so the reported ρ=1.50 may reflect image
difficulty rather than architectural blur sensitivity.

This script runs the same Gaussian-blur perturbation experiment across ALL
FOUR correctness strata defined by the 2×2 cross of per-model correctness:

    Stratum A  both_correct  : Qwen=✓  Smol=✓  (original robustness_blur.py)
    Stratum B  qwen_only     : Qwen=✓  Smol=✗  (easy for Qwen, hard for Smol)
    Stratum C  smol_only     : Qwen=✗  Smol=✓  (hard for Qwen, easy for Smol)
    Stratum D  both_wrong    : Qwen=✗  Smol=✗  (hard for both)

Key derived metrics
-------------------
    drop_pp[model, stratum] = baseline_acc[model] − blurred_acc[model]

Cross-stratum ratio (reviewer's key question):
    rho_A  = smol_drop_A / qwen_drop_A   (original, should reproduce ≈1.50)
    rho_BC = smol_drop_C / qwen_drop_B   (on "average-difficulty" images)

    If rho_BC ≈ rho_A  → image selection is NOT the main driver of the gap.
    If rho_BC < rho_A  → gap is partially inflated by image-difficulty bias.
    If rho_BC > rho_A  → gap is WORSE on harder images (conservative claim).

Stratum D is degenerate (baseline ≈0 % for both models) and reported for
completeness only; no ρ is computed.

Outputs
-------
    results/stratified_robustness_results.csv
    results/stratified_robustness_report.json
    stratified_robustness.log

Requirements
------------
    pip install torch torchvision transformers accelerate \\
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

# ── Third-party ───────────────────────────────────────────────────────────────
import torch
import torchvision.transforms.functional as TF
from PIL import Image
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

RESULTS_CSV      = Path("results/vlm_inference_results.csv")
OUTPUT_CSV       = Path("results/stratified_robustness_results.csv")
OUTPUT_JSON      = Path("results/stratified_robustness_report.json")
LOG_FILE         = Path("stratified_robustness.log")
IMAGE_CACHE_DIR  = Path("image_cache")

# Samples per stratum.  50 gives a reasonable estimate; use --n to override.
N_SAMPLES_PER_STRATUM = 50

# Gaussian blur parameters (identical to robustness_blur.py)
BLUR_KERNEL = 5
BLUR_SIGMA  = 2.0

# Stratum definitions: (key, human label, filter: qwen_ok, smol_ok)
STRATA = [
    ("both_correct", "A – both correct",   True,  True),
    ("qwen_only",    "B – Qwen only",       True,  False),
    ("smol_only",    "C – Smol only",       False, True),
    ("both_wrong",   "D – both wrong",      False, False),
]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Seed ──────────────────────────────────────────────────────────────────────
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# Correctness helpers  (identical to robustness_blur.py)
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
# VRAM helpers
# ══════════════════════════════════════════════════════════════════════════════

def vram_allocated_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(DEVICE) / 1024 ** 2


# ══════════════════════════════════════════════════════════════════════════════
# Model loaders  (identical to robustness_blur.py)
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
# Inference helpers  (identical to robustness_blur.py)
# ══════════════════════════════════════════════════════════════════════════════

def _qwen_infer(model, processor, image, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                *(([{"type": "image", "image": image}]) if image is not None else []),
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
            text=[text_prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text_prompt],
            images=[image] if image is not None else None,
            padding=True, return_tensors="pt",
        )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad(), autocast(dtype=torch.float16):
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def _smol_infer(model, processor, image, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                *(([{"type": "image"}]) if image is not None else []),
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
    try:
        return fn(model, processor, image, prompt)
    except torch.cuda.OutOfMemoryError:
        log.warning("[OOM] %s", label)
        torch.cuda.empty_cache()
        return "[OOM]"
    except Exception as exc:
        log.warning("[ERR] %s – %s: %s", label, type(exc).__name__, exc)
        return f"[ERROR: {type(exc).__name__}]"


# ══════════════════════════════════════════════════════════════════════════════
# Gaussian blur  (identical to robustness_blur.py)
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_blur(image: Image.Image) -> Image.Image:
    ks = BLUR_KERNEL if BLUR_KERNEL % 2 == 1 else BLUR_KERNEL + 1
    tensor  = TF.to_tensor(image)
    blurred = TF.gaussian_blur(tensor, kernel_size=[ks, ks], sigma=[BLUR_SIGMA, BLUR_SIGMA])
    return TF.to_pil_image(blurred)


# ══════════════════════════════════════════════════════════════════════════════
# Image cache / retrieval  (identical to robustness_blur.py)
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
    cache_path = IMAGE_CACHE_DIR / f"{_cache_key(dataset_name, image_id)}.png"
    if cache_path.exists():
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            cache_path.unlink(missing_ok=True)

    if dataset_name not in DS_MAP:
        log.warning("Unknown dataset '%s'", dataset_name)
        return None

    hf_path, split, img_field, id_field = DS_MAP[dataset_name]
    try:
        ds = load_dataset(hf_path, split=split, streaming=True, trust_remote_code=True)
        for sample in ds:
            sid = str(sample.get(id_field, sample.get("id", "")))
            if sid == image_id:
                img = _decode_image(sample.get(img_field))
                if img is not None:
                    img.save(cache_path, format="PNG")
                return img
    except Exception as exc:
        log.warning("Stream error for %s id=%s: %s", dataset_name, image_id, exc)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Per-stratum blur experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_stratum(stratum_key: str, stratum_label: str,
                sample: list[dict],
                qwen_model, qwen_proc,
                smol_model, smol_proc) -> tuple[list[dict], dict]:
    """
    Run blur inference on ``sample`` rows for a single stratum.

    Returns
    -------
    result_rows : list of per-row dicts (written to the output CSV)
    summary     : aggregated metrics for this stratum
    """
    log.info("── Stratum %s  (n=%d) ──────────────────────────────────", stratum_label, len(sample))

    rows_out: list[dict] = []
    skipped = 0

    for idx, row in enumerate(sample):
        ds_name   = row["dataset_name"]
        image_id  = row["image_id"]
        prompt    = row["task_prompt"]
        gt        = row["ground_truth"]

        log.info("  [%d/%d] %s id=%s", idx + 1, len(sample), ds_name, image_id)

        image = fetch_image(ds_name, image_id)
        if image is None:
            log.warning("    Could not retrieve image – skipping.")
            skipped += 1
            continue

        try:
            blurred = gaussian_blur(image)
        except Exception as exc:
            log.warning("    Blur failed: %s – skipping.", exc)
            skipped += 1
            continue

        qwen_blurred = safe_infer(_qwen_infer, qwen_model, qwen_proc,
                                  blurred, prompt, f"qwen[{stratum_key}|{idx}]")
        smol_blurred = safe_infer(_smol_infer, smol_model, smol_proc,
                                  blurred, prompt, f"smol[{stratum_key}|{idx}]")

        qwen_orig_correct = is_correct(ds_name, row["qwen_output"], gt)
        smol_orig_correct = is_correct(ds_name, row["smol_output"], gt)
        qwen_blur_correct = is_correct(ds_name, qwen_blurred, gt)
        smol_blur_correct = is_correct(ds_name, smol_blurred, gt)

        rows_out.append({
            "stratum":              stratum_key,
            "dataset_name":         ds_name,
            "image_id":             image_id,
            "task_prompt":          prompt,
            "ground_truth":         gt,
            "qwen_output_original": row["qwen_output"],
            "smol_output_original": row["smol_output"],
            "qwen_output_blurred":  qwen_blurred,
            "smol_output_blurred":  smol_blurred,
            "qwen_correct_original": qwen_orig_correct,
            "smol_correct_original": smol_orig_correct,
            "qwen_correct_blurred":  qwen_blur_correct,
            "smol_correct_blurred":  smol_blur_correct,
        })

    n = len(rows_out)
    if n == 0:
        log.warning("Stratum %s: no valid rows produced.", stratum_label)
        return rows_out, {"stratum": stratum_key, "n": 0, "skipped": skipped, "error": "no_data"}

    def pct(field: str) -> float:
        return 100.0 * sum(1 for r in rows_out if r[field]) / n

    qwen_base  = pct("qwen_correct_original")
    smol_base  = pct("smol_correct_original")
    qwen_blur_ = pct("qwen_correct_blurred")
    smol_blur_ = pct("smol_correct_blurred")

    qwen_drop = qwen_base - qwen_blur_
    smol_drop = smol_base - smol_blur_

    # ρ is meaningful only when BOTH models have a non-zero baseline in stratum
    if qwen_drop > 0 and smol_drop > 0:
        rho = smol_drop / qwen_drop
    elif qwen_drop == 0 and smol_drop > 0:
        rho = float("inf")
    elif smol_drop == 0 and qwen_drop > 0:
        rho = 0.0
    else:
        rho = None  # both drops are zero or undefined

    summary = {
        "stratum":          stratum_key,
        "stratum_label":    stratum_label,
        "n":                n,
        "skipped":          skipped,
        "qwen_baseline_acc":  round(qwen_base,  2),
        "smol_baseline_acc":  round(smol_base,  2),
        "qwen_blurred_acc":   round(qwen_blur_, 2),
        "smol_blurred_acc":   round(smol_blur_, 2),
        "qwen_drop_pp":       round(qwen_drop,  2),
        "smol_drop_pp":       round(smol_drop,  2),
        "rho":                round(rho, 3) if isinstance(rho, float) else rho,
    }

    log.info("    Qwen  baseline=%.1f%%  blurred=%.1f%%  drop=%.1f pp",
             qwen_base, qwen_blur_, qwen_drop)
    log.info("    Smol  baseline=%.1f%%  blurred=%.1f%%  drop=%.1f pp",
             smol_base, smol_blur_, smol_drop)
    if rho is not None:
        log.info("    ρ = %.3f (%s)", rho,
                 "Smol more sensitive" if rho > 1 else
                 "Qwen more sensitive" if rho < 1 else "equal")
    else:
        log.info("    ρ undefined (both drops are zero)")

    return rows_out, summary


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(n_per_stratum: int = N_SAMPLES_PER_STRATUM):
    log.info("=" * 60)
    log.info("  Stratified Robustness Experiment – Gaussian Blur")
    log.info("  N per stratum = %d", n_per_stratum)
    log.info("=" * 60)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(DEVICE)
        log.info("GPU: %s  (%.1f GiB)", props.name, props.total_memory / 1024 ** 3)

    # ── Load CSV and compute correctness flags ────────────────────────────────
    log.info("Reading %s …", RESULTS_CSV)
    all_rows = list(csv.DictReader(open(RESULTS_CSV, encoding="utf-8")))
    log.info("Total rows: %d", len(all_rows))

    for row in all_rows:
        row["_qwen_ok"] = is_correct(row["dataset_name"], row["qwen_output"], row["ground_truth"])
        row["_smol_ok"] = is_correct(row["dataset_name"], row["smol_output"], row["ground_truth"])

    # ── Build strata pools ────────────────────────────────────────────────────
    pools: dict[str, list[dict]] = {}
    for key, label, qwen_want, smol_want in STRATA:
        pools[key] = [
            r for r in all_rows
            if r["_qwen_ok"] == qwen_want and r["_smol_ok"] == smol_want
        ]
        log.info("Stratum %s (%s): %d rows available", key, label, len(pools[key]))

    # ── Sample from each stratum ──────────────────────────────────────────────
    samples: dict[str, list[dict]] = {}
    for key, label, _, _ in STRATA:
        pool = pools[key]
        size = min(n_per_stratum, len(pool))
        if size == 0:
            log.warning("Stratum %s is empty – skipping.", key)
            samples[key] = []
        else:
            samples[key] = random.sample(pool, size)
            log.info("Stratum %s: sampled %d rows", key, size)

    # ── Load models once ──────────────────────────────────────────────────────
    try:
        qwen_model, qwen_proc = load_qwen()
        smol_model, smol_proc = load_smolvlm()
    except torch.cuda.OutOfMemoryError:
        log.error("[OOM] Cannot load models.")
        return

    log.info("Both models loaded | VRAM alloc: %.0f MiB", vram_allocated_mb())

    # ── Run each stratum ──────────────────────────────────────────────────────
    all_result_rows: list[dict] = []
    strata_summaries: list[dict] = []
    t_total = time.perf_counter()

    for key, label, _, _ in STRATA:
        if not samples[key]:
            log.info("Stratum %s: no sample – skipping inference.", key)
            continue
        rows, summary = run_stratum(
            key, label, samples[key],
            qwen_model, qwen_proc,
            smol_model, smol_proc,
        )
        all_result_rows.extend(rows)
        strata_summaries.append(summary)

    elapsed = time.perf_counter() - t_total
    log.info("Total elapsed: %.1f s", elapsed)

    # ── Compute cross-stratum analysis ────────────────────────────────────────
    by_key = {s["stratum"]: s for s in strata_summaries}

    cross_stratum: dict = {}

    # rho_BC: Smol drop in stratum C vs Qwen drop in stratum B
    # This is the fairest cross-stratum comparison:
    # - Smol's drop in C = how much Smol degrades on images that were already hard for Qwen
    # - Qwen's drop in B = how much Qwen degrades on images that were already hard for Smol
    if "smol_only" in by_key and "qwen_only" in by_key:
        smol_drop_C = by_key["smol_only"]["smol_drop_pp"]
        qwen_drop_B = by_key["qwen_only"]["qwen_drop_pp"]
        if qwen_drop_B > 0 and smol_drop_C > 0:
            rho_BC = smol_drop_C / qwen_drop_B
        elif qwen_drop_B == 0 and smol_drop_C > 0:
            rho_BC = float("inf")
        elif smol_drop_C == 0 and qwen_drop_B > 0:
            rho_BC = 0.0
        else:
            rho_BC = None
        cross_stratum["rho_BC_smol_drop_C_vs_qwen_drop_B"] = (
            round(rho_BC, 3) if isinstance(rho_BC, float) else rho_BC
        )
        cross_stratum["smol_drop_C_pp"]  = smol_drop_C
        cross_stratum["qwen_drop_B_pp"]  = qwen_drop_B

    if "both_correct" in by_key:
        cross_stratum["rho_A_original_subset"] = by_key["both_correct"]["rho"]

    # Interpretation helper
    rho_A  = cross_stratum.get("rho_A_original_subset")
    rho_BC = cross_stratum.get("rho_BC_smol_drop_C_vs_qwen_drop_B")
    if rho_A is not None and rho_BC is not None and isinstance(rho_BC, float):
        if abs(rho_BC - rho_A) < 0.15:
            interpretation = "image_selection_NOT_main_driver"
        elif rho_BC > rho_A:
            interpretation = "gap_WORSE_on_harder_images_conservative_claim"
        else:
            interpretation = "gap_PARTIALLY_inflated_by_image_difficulty_bias"
        cross_stratum["interpretation"] = interpretation

    # ── Build report ──────────────────────────────────────────────────────────
    report = {
        "experiment":         "stratified_robustness",
        "blur_kernel":         BLUR_KERNEL,
        "blur_sigma":          BLUR_SIGMA,
        "n_per_stratum":       n_per_stratum,
        "total_elapsed_s":     round(elapsed, 1),
        "strata":              strata_summaries,
        "cross_stratum":       cross_stratum,
    }

    # ── Write outputs ─────────────────────────────────────────────────────────
    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("Report written → %s", OUTPUT_JSON)

    if all_result_rows:
        fieldnames = [
            "stratum", "dataset_name", "image_id", "task_prompt", "ground_truth",
            "qwen_output_original", "smol_output_original",
            "qwen_output_blurred",  "smol_output_blurred",
            "qwen_correct_original", "smol_correct_original",
            "qwen_correct_blurred",  "smol_correct_blurred",
        ]
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_result_rows)
        log.info("Per-row results written → %s  (%d rows)", OUTPUT_CSV, len(all_result_rows))

    # ── Pretty-print summary ──────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  STRATIFIED ROBUSTNESS SUMMARY")
    print("=" * 64)
    hdr = f"{'Stratum':<22} {'n':>4}  {'Qwen base':>9}  {'Qwen blur':>9}  {'Smol base':>9}  {'Smol blur':>9}  {'ρ':>6}"
    print(hdr)
    print("-" * 64)
    for s in strata_summaries:
        rho_str = f"{s['rho']:.3f}" if isinstance(s.get('rho'), float) else str(s.get('rho', '—'))
        print(
            f"  {s['stratum_label']:<20} {s['n']:>4}  "
            f"{s['qwen_baseline_acc']:>8.1f}%  {s['qwen_blurred_acc']:>8.1f}%  "
            f"{s['smol_baseline_acc']:>8.1f}%  {s['smol_blurred_acc']:>8.1f}%  "
            f"{rho_str:>6}"
        )
    print("=" * 64)
    if cross_stratum:
        print("\n  Cross-stratum analysis:")
        print(f"    ρ_A  (both-correct subset, original) = {cross_stratum.get('rho_A_original_subset', '—')}")
        print(f"    ρ_BC (Smol drop C / Qwen drop B)     = {cross_stratum.get('rho_BC_smol_drop_C_vs_qwen_drop_B', '—')}")
        if "interpretation" in cross_stratum:
            print(f"    Interpretation: {cross_stratum['interpretation']}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stratified blur robustness experiment")
    parser.add_argument("--n", type=int, default=N_SAMPLES_PER_STRATUM,
                        help=f"Samples per stratum (default: {N_SAMPLES_PER_STRATUM})")
    args = parser.parse_args()
    main(n_per_stratum=args.n)
