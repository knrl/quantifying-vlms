"""
calibration_ece.py
==================
Phase 4: Confidence Calibration + Expected Calibration Error (ECE)

Reads vlm_inference_results.csv, re-runs generation with output_scores=True
to extract per-token log-probabilities, computes sequence-level confidence
as the geometric mean token probability, bins predictions, computes ECE and
plots reliability diagrams for both models on both datasets.

Outputs
-------
  calibration_results.json
  calibration_reliability_diagram_{model}_{dataset}.png
  calibration_ece.log
"""

import csv
import gc
import json
import logging
import math
import os
import random
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
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
    raise SystemExit("pip install datasets") from exc

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 64          # shorter for speed — confidence doesn't need full answer
N_BINS = 10
RESULTS_CSV = Path("results/vlm_inference_results.csv")
OUTPUT_JSON = Path("results/calibration_results.json")
LOG_FILE = Path("calibration_ece.log")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="a")],
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


# ── Correctness (same as robustness_blur.py) ──────────────────────────────────
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


# ── VRAM helpers ──────────────────────────────────────────────────────────────
def vram_mb():
    return torch.cuda.memory_allocated(DEVICE) / 1024**2 if torch.cuda.is_available() else 0.0


# ── Model loaders (identical config) ─────────────────────────────────────────
def load_qwen(model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    log.info("Loading %s (4-bit NF4)...", model_id)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb, device_map={"": DEVICE},
        torch_dtype=torch.float16, trust_remote_code=True,
    )
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model, proc


def load_smolvlm(model_id="HuggingFaceTB/SmolVLM2-500M-Instruct"):
    log.info("Loading %s (FP16)...", model_id)
    model = SmolVLMForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16,
        device_map={"": DEVICE}, trust_remote_code=True,
    )
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model, proc


# ── Image cache ───────────────────────────────────────────────────────────────
import hashlib, io
from PIL import UnidentifiedImageError

IMAGE_CACHE_DIR = Path("image_cache")
IMAGE_CACHE_DIR.mkdir(exist_ok=True)

DS_MAP = {
    "vqav2":         ("lmms-lab/VQAv2",        "validation", "image", "question_id"),
    "coco_captions": ("lmms-lab/COCO-Caption",  "val",        "image", "question_id"),
    # NOTE: COCO-Caption HF field "id" is a sequential row counter (useless).
    #       "question_id" stores the full filename e.g. "COCO_val2014_000000203564.jpg"
    #       which matches the image_id column in vlm_inference_results.csv directly.
}


def _cache_path(dataset_name, image_id):
    # Always key the cache on the original CSV image_id string
    key = hashlib.md5(f"{dataset_name}:{image_id}".encode()).hexdigest()
    return IMAGE_CACHE_DIR / f"{key}.png"


def _normalize_id(dataset_name: str, csv_image_id: str) -> str:
    """
    Convert the CSV image_id to the id string used in the HF dataset.
    For both VQAv2 and COCO-Caption we now use the 'question_id' field,
    which stores the same value as the CSV image_id — no conversion needed.
    """
    return str(csv_image_id)


def _decode(raw):
    try:
        if isinstance(raw, Image.Image):        return raw.convert("RGB")
        if isinstance(raw, (bytes, bytearray)): return Image.open(io.BytesIO(raw)).convert("RGB")
        if isinstance(raw, dict) and "bytes" in raw:
            return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        if hasattr(raw, "read"):                return Image.open(raw).convert("RGB")
    except Exception:
        pass
    return None


def fetch_image(dataset_name, image_id):
    # Cache is keyed on original CSV image_id
    cp = _cache_path(dataset_name, image_id)
    if cp.exists():
        try: return Image.open(cp).convert("RGB")
        except Exception: cp.unlink(missing_ok=True)
    if dataset_name not in DS_MAP:
        return None
    hf_path, split, img_field, id_field = DS_MAP[dataset_name]
    if img_field is None:
        return None
    hf_id = _normalize_id(dataset_name, image_id)
    try:
        ds = load_dataset(hf_path, split=split, streaming=True)
        for sample in ds:
            sid = str(sample.get(id_field, ""))
            if sid == hf_id:
                img = _decode(sample.get(img_field))
                if img:
                    img.save(cp, format="PNG")
                return img
    except Exception as e:
        log.warning("fetch_image failed %s id=%s: %s", dataset_name, image_id, e)
    return None


def warm_image_cache(dataset_name: str, needed_ids: set):
    """
    Single-pass streaming download: iterate the HF dataset once and save every
    image whose id is in needed_ids.  Far fewer HTTP requests than calling
    fetch_image() one-at-a-time (which re-opens the stream per image).

    needed_ids are original CSV image_id strings; we build a mapping
    hf_id → csv_image_id so the cache file is keyed on the CSV id.
    """
    # hf_id → original csv image_id
    hf_to_csv = {_normalize_id(dataset_name, iid): str(iid) for iid in needed_ids}

    already = {hf_id for hf_id, csv_id in hf_to_csv.items()
               if _cache_path(dataset_name, csv_id).exists()}
    missing = set(hf_to_csv.keys()) - already

    if not missing:
        log.info("[cache] %s – all %d images already cached", dataset_name, len(needed_ids))
        return

    log.info("[cache] %s – downloading %d/%d images in one pass …",
             dataset_name, len(missing), len(needed_ids))
    if dataset_name not in DS_MAP:
        return
    hf_path, split, img_field, id_field = DS_MAP[dataset_name]
    if img_field is None:
        return

    saved = 0
    try:
        ds = load_dataset(hf_path, split=split, streaming=True)
        for sample in ds:
            sid = str(sample.get(id_field, ""))
            if sid in missing:
                csv_id = hf_to_csv[sid]
                img = _decode(sample.get(img_field))
                if img:
                    img.save(_cache_path(dataset_name, csv_id), format="PNG")
                    saved += 1
                missing.discard(sid)
                if (saved % 200) == 0 and saved:
                    log.info("[cache] %s – %d/%d saved …", dataset_name, saved, len(hf_to_csv))
                if not missing:
                    break   # got everything – stop streaming early
    except Exception as e:
        log.warning("[cache] warm_image_cache failed for %s: %s", dataset_name, e)

    log.info("[cache] %s – cached %d images (%d still missing)",
             dataset_name, saved, len(missing))


def prefetch_images(dataset_name: str, needed_ids: list) -> dict:
    """
    Return {csv_image_id: PIL.Image} for all needed_ids in one streaming pass.
    Hits disk cache first; only streams for truly missing ones.
    """
    result = {}
    missing_ids = []
    for iid in needed_ids:
        cp = _cache_path(dataset_name, iid)
        if cp.exists():
            try:
                result[iid] = Image.open(cp).convert("RGB")
            except Exception:
                cp.unlink(missing_ok=True)
                missing_ids.append(iid)
        else:
            missing_ids.append(iid)

    if not missing_ids:
        log.info("[prefetch] %s – all %d from disk cache", dataset_name, len(needed_ids))
        return result

    log.info("[prefetch] %s – streaming %d missing images …", dataset_name, len(missing_ids))
    hf_to_csv = {_normalize_id(dataset_name, iid): iid for iid in missing_ids}
    remaining = set(hf_to_csv.keys())

    if dataset_name in DS_MAP:
        hf_path, split, img_field, id_field = DS_MAP[dataset_name]
        try:
            ds = load_dataset(hf_path, split=split, streaming=True)
            fetched = 0
            for sample in ds:
                sid = str(sample.get(id_field, ""))
                if sid in remaining:
                    csv_id = hf_to_csv[sid]
                    img = _decode(sample.get(img_field))
                    if img:
                        img.save(_cache_path(dataset_name, csv_id), format="PNG")
                        result[csv_id] = img
                        fetched += 1
                    remaining.discard(sid)
                    if fetched % 200 == 0 and fetched:
                        log.info("[prefetch] %s – %d/%d fetched …",
                                 dataset_name, fetched, len(missing_ids))
                    if not remaining:
                        break
        except Exception as e:
            log.warning("[prefetch] stream failed for %s: %s", dataset_name, e)

    log.info("[prefetch] %s – done: %d/%d images ready (%d missing)",
             dataset_name, len(result), len(needed_ids), len(remaining))
    return result


# ── Inference with scores ──────────────────────────────────────────────────────

def _seq_confidence(scores: tuple, output_ids: torch.Tensor) -> float:
    """
    Geometric mean token probability over generated tokens.
    scores: tuple of (vocab_size,) tensors (one per generated step, raw logits).
    output_ids: (seq_len,) tensor of generated token ids.
    """
    if not scores:
        return 0.0
    log_probs = []
    for step_idx, logit in enumerate(scores):
        # logit shape: (1, vocab_size)
        probs = torch.softmax(logit[0], dim=-1)
        tok_id = output_ids[step_idx].item()
        p = probs[tok_id].item()
        log_probs.append(math.log(max(p, 1e-10)))
    return math.exp(sum(log_probs) / len(log_probs))


def qwen_infer_with_conf(model, processor, image, prompt):
    messages = [{"role": "user", "content": [
        *(([{"type": "image", "image": image}]) if image else []),
        {"type": "text", "text": prompt},
    ]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if image and _QWEN_UTILS:
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = processor(text=[text_prompt], images=img_inputs, videos=vid_inputs,
                           padding=True, return_tensors="pt")
    else:
        inputs = processor(text=[text_prompt], images=[image] if image else None,
                           padding=True, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad(), autocast(dtype=torch.float16):
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             return_dict_in_generate=True, output_scores=True)
    trimmed = out.sequences[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(trimmed, skip_special_tokens=True).strip()
    conf = _seq_confidence(out.scores, trimmed)
    return text, conf


def smol_infer_with_conf(model, processor, image, prompt):
    messages = [{"role": "user", "content": [
        *(([{"type": "image"}]) if image else []),
        {"type": "text", "text": prompt},
    ]}]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text_prompt, images=[image] if image else None, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad(), autocast(dtype=torch.float16):
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             return_dict_in_generate=True, output_scores=True)
    gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(gen_ids, skip_special_tokens=True).strip()
    if prompt in text:
        text = text.split(prompt, 1)[-1].strip()
    conf = _seq_confidence(out.scores, gen_ids)
    return text, conf


def safe_infer_conf(fn, model, processor, image, prompt, label):
    try:
        return fn(model, processor, image, prompt)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        log.warning("[OOM] %s", label)
        return "[OOM]", 0.0
    except Exception as exc:
        log.warning("[ERR] %s: %s", label, exc)
        return f"[ERROR]", 0.0


# ── ECE computation ───────────────────────────────────────────────────────────

def compute_ece(confidences, corrects, n_bins=N_BINS):
    bins = [[] for _ in range(n_bins)]
    for conf, correct in zip(confidences, corrects):
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, correct))
    ece = 0.0
    bin_stats = []
    N = len(confidences)
    for b, items in enumerate(bins):
        if not items:
            bin_stats.append({"bin": b, "count": 0, "avg_conf": 0.0, "accuracy": 0.0})
            continue
        avg_conf = sum(c for c, _ in items) / len(items)
        accuracy = sum(1 for _, ok in items if ok) / len(items)
        ece += (len(items) / N) * abs(accuracy - avg_conf)
        lo = b / n_bins
        hi = (b + 1) / n_bins
        bin_stats.append({"bin": b, "range": f"[{lo:.1f},{hi:.1f})", "count": len(items),
                          "avg_conf": round(avg_conf, 4), "accuracy": round(accuracy, 4)})
    return round(ece, 6), bin_stats


# ── Reliability diagram ────────────────────────────────────────────────────────

def plot_reliability_diagram(bin_stats, ece, model_label, dataset_label, out_path):
    centers = [(b["bin"] + 0.5) / N_BINS for b in bin_stats if b["count"] > 0]
    accs    = [b["accuracy"] for b in bin_stats if b["count"] > 0]
    confs   = [b["avg_conf"] for b in bin_stats if b["count"] > 0]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)
    ax.bar([c - 0.04 for c in centers], accs, width=0.08, alpha=0.7,
           color="steelblue", label="Accuracy")
    ax.step([0] + [b / N_BINS for b in range(1, N_BINS + 1)],
            [0] + [b["avg_conf"] for b in bin_stats],
            where="post", color="orange", linewidth=1.5, label="Avg confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram\n{model_label} | {dataset_label}\nECE={ece:.4f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Saved reliability diagram: %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  Phase 4: Confidence Calibration + ECE")
    log.info("=" * 60)

    if not RESULTS_CSV.exists():
        log.error("Missing %s -- run batch_inference.py first.", RESULTS_CSV)
        return

    # Load CSV rows
    rows = list(csv.DictReader(open(RESULTS_CSV, encoding="utf-8")))
    log.info("Loaded %d rows from %s", len(rows), RESULTS_CSV)

    # Pre-warm image cache: one streaming pass per dataset instead of
    # one HF stream open per image (avoids rate-limiting on unauthenticated requests)
    for dataset_name in ["vqav2", "coco_captions"]:
        needed = {r["image_id"] for r in rows if r["dataset_name"] == dataset_name}
        warm_image_cache(dataset_name, needed)

    # Load models once
    try:
        qwen_model, qwen_proc = load_qwen()
        smol_model, smol_proc = load_smolvlm()
    except torch.cuda.OutOfMemoryError:
        log.error("[OOM] Cannot load models.")
        return

    results = {}  # {model: {dataset: {ece, bin_stats}}}

    for model_key, infer_fn, model, proc in [
        ("smolvlm2", smol_infer_with_conf, smol_model, smol_proc),
        ("qwen",     qwen_infer_with_conf,  qwen_model, qwen_proc),
    ]:
        results[model_key] = {}
        for dataset_name in ["vqav2", "coco_captions"]:
            dataset_rows = [r for r in rows if r["dataset_name"] == dataset_name]
            log.info("[%s | %s] %d rows", model_key, dataset_name, len(dataset_rows))

            # Prefetch all images in one streaming pass (avoids re-opening stream per image)
            needed_ids = [r["image_id"] for r in dataset_rows]
            image_map = prefetch_images(dataset_name, needed_ids)

            confidences, corrects = [], []
            t0 = time.perf_counter()

            for i, row in enumerate(dataset_rows):
                image = image_map.get(row["image_id"])
                prompt = row["task_prompt"]
                gt = row["ground_truth"]

                text, conf = safe_infer_conf(infer_fn, model, proc, image, prompt,
                                             f"{model_key}[{i}]")
                correct = is_correct(dataset_name, text, gt)
                confidences.append(conf)
                corrects.append(correct)

                if (i + 1) % 100 == 0:
                    elapsed = time.perf_counter() - t0
                    log.info("  [%s|%s] %d/%d done | %.1fs | VRAM %.0f MiB",
                             model_key, dataset_name, i + 1, len(dataset_rows),
                             elapsed, vram_mb())

            ece, bin_stats = compute_ece(confidences, corrects)
            acc = sum(corrects) / len(corrects) * 100 if corrects else 0
            log.info("[%s | %s] ECE=%.4f  Accuracy=%.1f%%", model_key, dataset_name, ece, acc)

            results[model_key][dataset_name] = {
                "n": len(dataset_rows),
                "accuracy_pct": round(acc, 2),
                "ece": ece,
                "bin_stats": bin_stats,
            }

            # Reliability diagram
            model_label = "SmolVLM2-500M" if model_key == "smolvlm2" else "Qwen2.5-VL-7B"
            ds_label = dataset_name.replace("_", " ").title()
            out_png = Path(f"figures/calibration_reliability_diagram_{model_key}_{dataset_name}.png")
            plot_reliability_diagram(bin_stats, ece, model_label, ds_label, out_png)

    # Print summary table
    print("\n" + "=" * 62)
    print("  CALIBRATION SUMMARY")
    print("=" * 62)
    print(f"{'Model':<22} {'Dataset':<16} {'Accuracy':>10} {'ECE':>10}")
    print("-" * 62)
    for model_key in ["smolvlm2", "qwen"]:
        label = "SmolVLM2-500M" if model_key == "smolvlm2" else "Qwen2.5-VL-7B"
        for ds in ["vqav2", "coco_captions"]:
            r = results[model_key].get(ds, {})
            print(f"{label:<22} {ds:<16} {r.get('accuracy_pct', 0):>9.1f}% {r.get('ece', 0):>10.4f}")
    print("=" * 62)

    OUTPUT_JSON.write_text(json.dumps(results, indent=2))
    log.info("Saved calibration results to %s", OUTPUT_JSON)

    del qwen_model, smol_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
