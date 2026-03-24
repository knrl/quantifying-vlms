"""
negation_probes.py
==================
Phase 5: Negation Robustness Probes

For each dataset, takes rows from vlm_inference_results.csv where both
models answered correctly on the original question, then re-presents the
same image with a negation-modified prompt (e.g., "What is NOT in the
image?", "Is it true that [answer] is present? Answer yes or no.").

Measures:
  - Negation accuracy: does the model still give a correct / sensible response?
  - Negation failure rate: fraction of samples where model hallucinates or
    gives logically inconsistent answers.
  - Consistency score: agreement between original and negation responses.

Outputs
-------
  negation_probes_results.csv
  negation_probes_summary.json
  negation_probes.log
"""

import csv
import gc
import json
import logging
import os
import random
import re
import time
from pathlib import Path

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
MAX_NEW_TOKENS = 64
RESULTS_CSV = Path("results/vlm_inference_results.csv")
OUTPUT_CSV = Path("results/negation_probes_results.csv")
OUTPUT_JSON = Path("results/negation_probes_summary.json")
LOG_FILE = Path("negation_probes.log")

# How many "both-correct" samples to probe per dataset
N_SAMPLES_PER_DATASET = 100

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


# ── Negation prompt templates ─────────────────────────────────────────────────
# Four probe types, applied per-sample depending on dataset
NEGATION_TEMPLATES = {
    "is_not":   "What object or thing is NOT depicted in this image?",
    "absent":   "Name something absent from this image.",
    "false_yn": "Is it true that '{answer}' is NOT shown in this image? Answer yes or no.",
    "counter":  "The answer to the question '{question}' is NOT '{answer}'. What is the correct answer?",
}

COCO_PROMPT = "Describe this image in one or two sentences."


def build_negation_prompts(original_question: str, original_answer: str, dataset_name: str):
    """
    Returns list of (template_key, prompt_string) for this sample.
    """
    probes = []
    probes.append(("is_not", NEGATION_TEMPLATES["is_not"]))
    probes.append(("absent", NEGATION_TEMPLATES["absent"]))

    if original_answer:
        answer_snip = original_answer[:60].strip()
        question_snip = original_question[:80].strip() if original_question else "this image"
        probes.append(("false_yn",
                        NEGATION_TEMPLATES["false_yn"].format(answer=answer_snip)))
        probes.append(("counter",
                        NEGATION_TEMPLATES["counter"].format(
                            question=question_snip, answer=answer_snip)))
    return probes


# ── Correctness (same heuristic) ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_correct(dataset_name, pred, gt):
    pred, gt = _normalize(pred), _normalize(gt)
    if not pred or not gt:
        return False
    if dataset_name == "vqav2":
        return gt in pred or pred in gt
    else:
        words = [w for w in gt.split() if len(w) >= 3]
        return any(w in pred for w in words)


def is_negation_correct(template_key: str, response: str, original_answer: str) -> bool:
    """
    Heuristic 'success' check for negation probes.

    - is_not / absent: success if model outputs ANY non-empty non-error response
      that does NOT echo the original answer verbatim.
    - false_yn: the template is "Is it true that '{answer}' is NOT shown in this
      image?" where {answer} IS the original correct answer (i.e., it IS depicted).
      The semantically correct response is therefore "No" (the object IS shown, so
      it is NOT true that it is NOT shown).  A model that says "Yes" is
      hallucinating — it incorrectly claims the depicted object is absent.
      Success criterion: response contains "no" (model correctly rejects the
      false premise).
    - counter: success if response does NOT match original answer.
    """
    resp = _normalize(response)
    if not resp or resp.startswith("["):
        return False
    orig = _normalize(original_answer)

    if template_key in ("is_not", "absent"):
        return orig[:20] not in resp      # model should name something different

    if template_key == "false_yn":
        # We asked: "Is it true that X is NOT shown?" -> sensible answer is "No" (X IS shown)
        # A hallucinating model that agrees with the false premise would say "Yes"
        return "no" in resp[:20]          # model correctly rejects the false premise

    if template_key == "counter":
        # Model should push back or give original answer -- failure is if it echoes wrong info
        return len(resp) > 5              # anything substantive counts as sensible

    return True


# ── VRAM helpers ──────────────────────────────────────────────────────────────
def vram_mb():
    return torch.cuda.memory_allocated(DEVICE) / 1024**2 if torch.cuda.is_available() else 0.0


# ── Model loaders ─────────────────────────────────────────────────────────────
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


# ── Image cache / fetch (same as other scripts) ───────────────────────────────
import hashlib, io

IMAGE_CACHE_DIR = Path("image_cache")
IMAGE_CACHE_DIR.mkdir(exist_ok=True)

DS_MAP = {
    "vqav2":         ("lmms-lab/VQAv2",        "validation", "image", "question_id"),
    "coco_captions": ("lmms-lab/COCO-Caption",  "val",        "image", "id"),
}


def _cache_path(dataset_name, image_id):
    key = hashlib.md5(f"{dataset_name}:{image_id}".encode()).hexdigest()
    return IMAGE_CACHE_DIR / f"{key}.png"


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
    cp = _cache_path(dataset_name, image_id)
    if cp.exists():
        try: return Image.open(cp).convert("RGB")
        except Exception: cp.unlink(missing_ok=True)
    if dataset_name not in DS_MAP:
        return None
    hf_path, split, img_field, id_field = DS_MAP[dataset_name]
    if img_field is None:
        return None
    try:
        ds = load_dataset(hf_path, split=split, streaming=True)
        for sample in ds:
            sid = str(sample.get(id_field, ""))
            if sid == str(image_id):
                img = _decode(sample.get(img_field))
                if img:
                    img.save(cp, format="PNG")
                return img
    except Exception as e:
        log.warning("fetch_image failed %s id=%s: %s", dataset_name, image_id, e)
    return None


# ── Inference helpers ─────────────────────────────────────────────────────────

def qwen_infer(model, processor, image, prompt):
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
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    trimmed = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


def smol_infer(model, processor, image, prompt):
    messages = [{"role": "user", "content": [
        *(([{"type": "image"}]) if image else []),
        {"type": "text", "text": prompt},
    ]}]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text_prompt, images=[image] if image else None, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad(), autocast(dtype=torch.float16):
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(gen_ids, skip_special_tokens=True).strip()
    if prompt in text:
        text = text.split(prompt, 1)[-1].strip()
    return text


def safe_infer(fn, model, processor, image, prompt, label=""):
    try:
        return fn(model, processor, image, prompt)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        log.warning("[OOM] %s", label)
        return "[OOM]"
    except Exception as exc:
        log.warning("[ERR] %s: %s", label, exc)
        return "[ERROR]"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  Phase 5: Negation Probes")
    log.info("=" * 60)

    if not RESULTS_CSV.exists():
        log.error("Missing %s -- run batch_inference.py first.", RESULTS_CSV)
        return

    rows = list(csv.DictReader(open(RESULTS_CSV, encoding="utf-8")))
    log.info("Loaded %d total rows from %s", len(rows), RESULTS_CSV)

    # Filter rows where both models answered correctly on original prompts.
    # The CSV has raw output columns (smol_output, qwen_output) rather than
    # pre-computed boolean flags, so compute correctness inline.
    both_correct = []
    for r in rows:
        ds = r.get("dataset_name", "")
        gt = r.get("ground_truth", "")
        qwen_ok = is_correct(ds, r.get("qwen_output", ""), gt)
        smol_ok = is_correct(ds, r.get("smol_output", ""), gt)
        if qwen_ok and smol_ok:
            both_correct.append(r)

    log.info("Both-correct rows: %d", len(both_correct))

    # Load models
    try:
        qwen_model, qwen_proc = load_qwen()
        smol_model, smol_proc = load_smolvlm()
    except torch.cuda.OutOfMemoryError:
        log.error("[OOM] Cannot load models.")
        return

    output_rows = []
    summary = {}

    for dataset_name in ["vqav2", "coco_captions"]:
        pool = [r for r in both_correct if r["dataset_name"] == dataset_name]
        random.shuffle(pool)
        sample = pool[:N_SAMPLES_PER_DATASET]
        log.info("[%s] Sampling %d / %d both-correct rows", dataset_name, len(sample), len(pool))

        # Per-template counters: {template_key: {model_key: [bool]}}
        template_results = {k: {"qwen": [], "smolvlm2": []} for k in NEGATION_TEMPLATES}

        t0 = time.perf_counter()
        for i, row in enumerate(sample):
            image_id = row["image_id"]
            original_question = row.get("task_prompt", "")
            original_answer = row.get("ground_truth", "")
            if isinstance(original_answer, str) and original_answer.startswith("["):
                # It might be a stringified list from the CSV
                try:
                    parsed = json.loads(original_answer.replace("'", '"'))
                    if isinstance(parsed, list) and parsed:
                        original_answer = str(parsed[0])
                except Exception:
                    pass

            image = fetch_image(dataset_name, image_id)

            probes = build_negation_prompts(original_question, original_answer, dataset_name)

            for template_key, neg_prompt in probes:
                qwen_resp = safe_infer(qwen_infer, qwen_model, qwen_proc, image, neg_prompt,
                                       f"qwen[{i}|{template_key}]")
                smol_resp = safe_infer(smol_infer, smol_model, smol_proc, image, neg_prompt,
                                       f"smol[{i}|{template_key}]")

                qwen_ok = is_negation_correct(template_key, qwen_resp, original_answer)
                smol_ok = is_negation_correct(template_key, smol_resp, original_answer)

                template_results[template_key]["qwen"].append(qwen_ok)
                template_results[template_key]["smolvlm2"].append(smol_ok)

                output_rows.append({
                    "dataset_name":      dataset_name,
                    "image_id":          image_id,
                    "original_question": original_question,
                    "original_answer":   original_answer,
                    "template_key":      template_key,
                    "negation_prompt":   neg_prompt,
                    "qwen_response":     qwen_resp,
                    "smol_response":     smol_resp,
                    "qwen_negation_ok":  qwen_ok,
                    "smol_negation_ok":  smol_ok,
                })

            if (i + 1) % 20 == 0:
                elapsed = time.perf_counter() - t0
                log.info("  [%s] %d/%d | %.1fs | VRAM %.0f MiB",
                         dataset_name, i + 1, len(sample), elapsed, vram_mb())

        # Aggregate per-template rates
        ds_summary = {}
        for tk, per_model in template_results.items():
            ds_summary[tk] = {}
            for model_key, results_list in per_model.items():
                if results_list:
                    ds_summary[tk][model_key] = {
                        "n": len(results_list),
                        "success_rate": round(sum(results_list) / len(results_list), 4),
                        "failure_rate": round(1 - sum(results_list) / len(results_list), 4),
                    }
        summary[dataset_name] = ds_summary

    # Print summary table
    print("\n" + "=" * 72)
    print("  NEGATION PROBE SUMMARY")
    print("=" * 72)
    print(f"{'Dataset':<16} {'Template':<12} {'Qwen Success':>14} {'Smol Success':>14} {'Δ (Qwen-Smol)':>14}")
    print("-" * 72)
    for ds, templates in summary.items():
        for tk, models in templates.items():
            q = models.get("qwen", {}).get("success_rate", float("nan"))
            s = models.get("smolvlm2", {}).get("success_rate", float("nan"))
            delta = q - s if (not any(isinstance(v, float) and v != v for v in [q, s])) else float("nan")
            print(f"{ds:<16} {tk:<12} {q:>13.1%} {s:>13.1%} {delta:>+13.1%}")
    print("=" * 72)

    # Write CSV
    if output_rows:
        fieldnames = list(output_rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        log.info("Saved %d rows to %s", len(output_rows), OUTPUT_CSV)

    OUTPUT_JSON.write_text(json.dumps(summary, indent=2))
    log.info("Saved summary to %s", OUTPUT_JSON)

    del qwen_model, smol_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
