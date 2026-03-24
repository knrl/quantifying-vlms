"""
llm_judge.py
============
Phase 6: LLM-as-Judge Error Taxonomy + Cohen's Kappa

Reads vlm_inference_results.csv, filters rows where at least one model was
incorrect, samples up to N_JUDGE_SAMPLES per model×dataset pair, sends each
failure to GPT-4o with a structured taxonomy prompt, collects labels, then
computes inter-rater Cohen's κ between the two model error distributions and
(optionally) against any human annotation CSV if provided.

Taxonomy categories
-------------------
  A  Object Blindness  – model fails to perceive the primary object
  B  Semantic Drift    – model recognises objects but misinterprets context
  C  Prior Bias        – model relies on training priors, ignores image cues
  D  Spatial Error     – model misjudges spatial relationships / counts
  E  Other             – does not fit A-D

Outputs
-------
  llm_judge_labels.json
  cohen_kappa_report.json
  llm_judge.log
"""

import csv
import json
import logging
import os
import random
import re
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit("pip install openai>=1.0") from exc

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError as exc:
    raise SystemExit("pip install scikit-learn") from exc

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
RESULTS_CSV = Path("results/vlm_inference_results.csv")
OUTPUT_LABELS = Path("results/llm_judge_labels.json")
OUTPUT_KAPPA  = Path("results/cohen_kappa_report.json")
LOG_FILE      = Path("llm_judge.log")

# GPT model for judging
JUDGE_MODEL = "gpt-4o"               # full power; swap to "gpt-4o-mini" for cheaper

# Number of failures to judge per (model, dataset) combination
N_JUDGE_SAMPLES = 100

# Optional: path to CSV with human-annotated labels (columns: row_id, human_label)
HUMAN_LABELS_CSV = Path("human_labels.csv")

# Retry config for OpenAI API
MAX_RETRIES = 3
RETRY_DELAY = 5.0   # seconds between retries

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

# ── Taxonomy ─────────────────────────────────────────────────────────────────
CATEGORIES = ["A", "B", "C", "D", "E"]
TAXONOMY_DESCRIPTION = """\
A  Object Blindness  – The model fails to detect or acknowledge the primary object.
B  Semantic Drift    – The model detects objects but misinterprets meaning / context.
C  Prior Bias        – The model ignores visual evidence, relying on statistical priors.
D  Spatial Error     – The model makes errors about position, size, count, or orientation.
E  Other             – The failure mode does not clearly fit categories A-D."""

JUDGE_SYSTEM_PROMPT = f"""\
You are an expert evaluator for vision-language model (VLM) failures.
Given a task description, ground-truth answer, and model prediction, classify
the type of error into exactly one of the following categories:

{TAXONOMY_DESCRIPTION}

Respond with a JSON object containing exactly two keys:
  "category": one of ["A","B","C","D","E"]
  "reason":   a one-sentence explanation of why you chose that category.

Do NOT add any text outside the JSON object."""

JUDGE_USER_TEMPLATE = """\
Dataset: {dataset}
Task / Question: {question}
Ground-truth answer: {ground_truth}
Model prediction: {prediction}

Classify the error."""


# ── Correctness helpers (mirroring batch_inference.py / robustness_blur.py) ──
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
    else:  # coco_captions
        gt_words = [w for w in gt.split() if len(w) >= 3]
        return any(w in pred for w in gt_words)


# ── OpenAI client ─────────────────────────────────────────────────────────────
def get_client():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log.error("OPENAI_API_KEY environment variable not set.")
        raise EnvironmentError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


# ── Classify one failure ──────────────────────────────────────────────────────
def classify_failure(client, dataset, question, ground_truth, prediction):
    user_msg = JUDGE_USER_TEMPLATE.format(
        dataset=dataset, question=question,
        ground_truth=ground_truth, prediction=prediction,
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
                max_tokens=120,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            category = data.get("category", "E")
            if category not in CATEGORIES:
                category = "E"
            reason = data.get("reason", "")
            return category, reason
        except Exception as exc:
            log.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    return "E", "API call failed after retries"


# ── Cohen's κ computation ────────────────────────────────────────────────────
def compute_kappa(labels_a, labels_b, label_name_a="qwen", label_name_b="smolvlm2"):
    """
    Compute Cohen's κ for two lists of categorical labels of equal length.
    Returns a dict with kappa, interpretation, and counts.
    """
    if len(labels_a) != len(labels_b):
        min_len = min(len(labels_a), len(labels_b))
        log.warning("Label lists differ in length (%d vs %d); truncating to %d",
                    len(labels_a), len(labels_b), min_len)
        labels_a = labels_a[:min_len]
        labels_b = labels_b[:min_len]

    kappa = cohen_kappa_score(labels_a, labels_b, labels=CATEGORIES)

    def interpret(k):
        if k < 0:        return "less than chance"
        if k < 0.2:      return "slight"
        if k < 0.4:      return "fair"
        if k < 0.6:      return "moderate"
        if k < 0.8:      return "substantial"
        return "almost perfect"

    # frequency tables
    def freq(labels):
        return {c: labels.count(c) for c in CATEGORIES}

    return {
        "n": len(labels_a),
        "kappa": round(kappa, 4),
        "interpretation": interpret(kappa),
        f"freq_{label_name_a}": freq(labels_a),
        f"freq_{label_name_b}": freq(labels_b),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  Phase 6: LLM-as-Judge + Cohen's Kappa")
    log.info("=" * 60)

    if not RESULTS_CSV.exists():
        log.error("Missing %s -- run batch_inference.py first.", RESULTS_CSV)
        return

    rows = list(csv.DictReader(open(RESULTS_CSV, encoding="utf-8")))
    log.info("Loaded %d rows from %s", len(rows), RESULTS_CSV)

    # Check API key early
    try:
        client = get_client()
    except EnvironmentError:
        return

    all_labels = []     # list of dicts to dump to JSON
    kappa_data = {}

    # ── Per dataset, per model: sample failures and judge them ────────────────
    for dataset_name in ["vqav2", "coco_captions"]:
        dataset_rows = [r for r in rows if r["dataset_name"] == dataset_name]
        log.info("[%s] total rows: %d", dataset_name, len(dataset_rows))

        kappa_data[dataset_name] = {}
        aligned_labels = {}   # {model_key: [label]} for matched image_ids

        for model_key, pred_col in [
            ("qwen",     "qwen_output"),
            ("smolvlm2", "smol_output"),
        ]:
            failures = [r for r in dataset_rows
                        if not is_correct(dataset_name,
                                          r.get(pred_col, ""),
                                          r.get("ground_truth", ""))]
            random.shuffle(failures)
            sample = failures[:N_JUDGE_SAMPLES]
            log.info("[%s | %s] %d failures -> judging %d",
                     dataset_name, model_key, len(failures), len(sample))

            model_labels = []
            t0 = time.perf_counter()
            for i, row in enumerate(sample):
                question    = row.get("task_prompt", "")
                ground_truth = row.get("ground_truth", "")
                raw_pred    = row.get(pred_col, "")
                image_id    = row.get("image_id", "")
                # Strip SmolVLM2 chat-template 'Assistant: ' prefix
                prediction  = raw_pred.strip()
                if prediction.lower().startswith("assistant:"):
                    prediction = prediction[len("assistant:"):].strip()

                category, reason = classify_failure(
                    client, dataset_name, question, ground_truth, prediction)

                model_labels.append(category)
                all_labels.append({
                    "dataset":        dataset_name,
                    "model":          model_key,
                    "image_id":       image_id,
                    "question":       question,
                    "ground_truth":   ground_truth,
                    "prediction":     prediction,
                    # Diagnostic fields for taxonomy stratification
                    "is_empty_output": not bool(prediction.strip()),
                    "output_length":   len(prediction.split()),
                    "category":       category,
                    "reason":         reason,
                })

                if (i + 1) % 25 == 0:
                    elapsed = time.perf_counter() - t0
                    log.info("  [%s|%s] %d/%d | %.1fs", dataset_name, model_key,
                             i + 1, len(sample), elapsed)

            aligned_labels[model_key] = model_labels
            log.info("[%s | %s] label distribution: %s", dataset_name, model_key,
                     {c: model_labels.count(c) for c in CATEGORIES})

        # Cohen's κ between qwen and smolvlm2 failure modes (aligned by sample order)
        qlen = len(aligned_labels.get("qwen", []))
        slen = len(aligned_labels.get("smolvlm2", []))
        if qlen > 0 and slen > 0:
            # Pad shorter list with "E" if they differ
            q_labels = aligned_labels["qwen"]
            s_labels = aligned_labels["smolvlm2"]
            max_len = max(qlen, slen)
            q_labels = q_labels + ["E"] * (max_len - qlen)
            s_labels = s_labels + ["E"] * (max_len - slen)
            kappa_result = compute_kappa(q_labels, s_labels)
            kappa_data[dataset_name]["qwen_vs_smolvlm2"] = kappa_result
            log.info("[%s] Cohen's κ (Qwen vs SmolVLM2) = %.4f (%s)",
                     dataset_name, kappa_result["kappa"], kappa_result["interpretation"])

        # Optional: Cohen's κ against human labels
        if HUMAN_LABELS_CSV.exists():
            log.info("Loading human labels from %s", HUMAN_LABELS_CSV)
            human_map = {}
            try:
                for hr in csv.DictReader(open(HUMAN_LABELS_CSV, encoding="utf-8")):
                    human_map[hr["row_id"]] = hr["human_label"]
            except Exception as e:
                log.warning("Could not load human labels: %s", e)
            for model_key in ["qwen", "smolvlm2"]:
                ml = aligned_labels.get(model_key, [])
                if not ml:
                    continue
                sample_ids = [r["image_id"] for r in
                              [r for r in rows if r["dataset_name"] == dataset_name
                               and r.get(f"{model_key}_correct", "").strip().lower()
                               not in ("true", "1", "yes")][:N_JUDGE_SAMPLES]]
                human_labels = [human_map.get(sid, "E") for sid in sample_ids]
                k_vs_human = compute_kappa(ml[:len(human_labels)], human_labels,
                                           label_name_a=model_key, label_name_b="human")
                kappa_data[dataset_name][f"{model_key}_vs_human"] = k_vs_human
                log.info("[%s] Cohen's κ (%s vs human) = %.4f (%s)",
                         dataset_name, model_key,
                         k_vs_human["kappa"], k_vs_human["interpretation"])

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  LLM-JUDGE ERROR TAXONOMY SUMMARY")
    print("=" * 68)
    for ds, kappa_items in kappa_data.items():
        for comparison, kres in kappa_items.items():
            print(f"\n  Dataset: {ds}  |  Comparison: {comparison}")
            print(f"    n={kres['n']}  κ={kres['kappa']:.4f}  ({kres['interpretation']})")
            for k, v in kres.items():
                if k.startswith("freq_"):
                    print(f"    {k}: {v}")
    print("=" * 68)

    # ── Save outputs ──────────────────────────────────────────────────────────
    OUTPUT_LABELS.write_text(json.dumps(all_labels, indent=2))
    log.info("Saved %d labelled examples to %s", len(all_labels), OUTPUT_LABELS)

    OUTPUT_KAPPA.write_text(json.dumps(kappa_data, indent=2))
    log.info("Saved kappa report to %s", OUTPUT_KAPPA)


if __name__ == "__main__":
    main()
