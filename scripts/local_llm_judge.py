"""
local_llm_judge.py
==================
Phase 6 (local): Error Taxonomy + distributional kappa using
facebook/bart-large-mnli zero-shot text classification.

Used as a substitute for llm_judge.py (which requires OPENAI_API_KEY) when
running in environments without API access.  BART-large-MNLI is a dedicated
NLI/classification model that is:
  - Not one of the two evaluated VLMs (no conflict of interest)
  - Fully deterministic at temperature=0
  - ~1.6 GB; runs on the available GPU

Methodology note (W5): The cross-failure-set kappa computed here measures
distributional overlap between the Qwen failure set and the SmolVLM2 failure
set, not inter-rater reliability.  See sec:judge in paper.tex.

Outputs: results/llm_judge_labels.json, results/cohen_kappa_report.json
"""

import csv
import json
import logging
import os
import random
import re
from pathlib import Path

import torch
from transformers import pipeline
from sklearn.metrics import cohen_kappa_score

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
RESULTS_CSV   = Path("results/vlm_inference_results.csv")
OUTPUT_LABELS = Path("results/llm_judge_labels.json")
OUTPUT_KAPPA  = Path("results/cohen_kappa_report.json")
LOG_FILE      = Path("llm_judge.log")

JUDGE_MODEL = "facebook/bart-large-mnli"
N_JUDGE_SAMPLES = 100   # failures per (model × dataset)
BATCH_SIZE = 16

CATEGORIES = ["A", "B", "C", "D", "E"]

# NLI hypothesis strings for each category -- applied to
# "premise = question + ground_truth + prediction" context
HYPOTHESES = {
    "A": "The model completely failed to identify or mention the main subject of the question.",
    "B": "The model correctly identified the subject but described it with wrong attributes, color, or action.",
    "C": "The model gave a generic, plausible answer based on common knowledge rather than the specific image content.",
    "D": "The model made an error about spatial relationships, count, position, or orientation.",
    "E": "The error does not clearly fit any specific failure pattern.",
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="a")],
)
log = logging.getLogger(__name__)
random.seed(SEED)

# ── Correctness helpers ───────────────────────────────────────────────────────
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


# ── Build premise for NLI ─────────────────────────────────────────────────────
def build_premise(dataset: str, question: str, ground_truth: str, prediction: str) -> str:
    return (
        f"Task: {question} "
        f"Correct answer: {ground_truth} "
        f"Model prediction: {prediction}"
    )


# ── Per-item zero-shot classification ────────────────────────────────────────
def classify_one_v2(classifier, premise: str) -> tuple[str, str]:
    """Zero-shot classify using all five hypotheses at once.

    hypothesis_template="{}" passes the candidate label string directly as the
    NLI hypothesis, rather than wrapping it in the default "This example is {}."
    template.  This is correct because HYPOTHESES already contain full sentences.
    """
    candidate_labels = [HYPOTHESES[c] for c in CATEGORIES]
    result = classifier(
        premise,
        candidate_labels=candidate_labels,
        hypothesis_template="{}",
        multi_label=False,
    )
    # result: {"sequence":..., "labels": [...sorted by score desc], "scores": [...]}
    top_hyp   = result["labels"][0]
    top_score = result["scores"][0]
    hyp_to_cat = {v: k for k, v in HYPOTHESES.items()}
    best_cat = hyp_to_cat.get(top_hyp, "E")
    reason = f"BART-MNLI top label '{best_cat}' (score={top_score:.3f})"
    return best_cat, reason


# ── Cohen's κ ─────────────────────────────────────────────────────────────────
def compute_kappa(labels_a, labels_b, name_a="qwen", name_b="smolvlm2"):
    n = min(len(labels_a), len(labels_b))
    labels_a, labels_b = labels_a[:n], labels_b[:n]
    if len(set(labels_a + labels_b)) <= 1:
        return {"n": n, "kappa": None,
                "interpretation": "undefined (degenerate — all labels identical)",
                f"freq_{name_a}": {c: labels_a.count(c) for c in CATEGORIES},
                f"freq_{name_b}": {c: labels_b.count(c) for c in CATEGORIES}}
    kappa = cohen_kappa_score(labels_a, labels_b, labels=CATEGORIES)
    def interpret(k):
        if k < 0:    return "less than chance"
        if k < 0.2:  return "slight"
        if k < 0.4:  return "fair"
        if k < 0.6:  return "moderate"
        if k < 0.8:  return "substantial"
        return "almost perfect"
    return {
        "n": n,
        "kappa": round(kappa, 4),
        "interpretation": interpret(kappa),
        f"freq_{name_a}": {c: labels_a.count(c) for c in CATEGORIES},
        f"freq_{name_b}": {c: labels_b.count(c) for c in CATEGORIES},
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  Local LLM Judge  —  BART-large-MNLI zero-shot")
    log.info("=" * 60)

    rows = list(csv.DictReader(open(RESULTS_CSV, encoding="utf-8")))
    log.info("Loaded %d rows from %s", len(rows), RESULTS_CSV)

    device = 0 if torch.cuda.is_available() else -1
    log.info("Loading %s on device=%s ...", JUDGE_MODEL, device)
    classifier = pipeline(
        "zero-shot-classification",
        model=JUDGE_MODEL,
        device=device,
    )
    log.info("Model loaded.")

    all_labels = []
    kappa_data = {}

    for dataset_name in ["vqav2", "coco_captions"]:
        dataset_rows = [r for r in rows if r["dataset_name"] == dataset_name]
        kappa_data[dataset_name] = {}
        aligned_labels = {}

        for model_key, pred_col in [("qwen", "qwen_output"), ("smolvlm2", "smol_output")]:
            failures = [r for r in dataset_rows
                        if not is_correct(dataset_name,
                                          r.get(pred_col, ""),
                                          r.get("ground_truth", ""))]
            random.shuffle(failures)
            sample = failures[:N_JUDGE_SAMPLES]
            log.info("[%s | %s] %d failures -> judging %d",
                     dataset_name, model_key, len(failures), len(sample))

            model_labels = []
            for i, row in enumerate(sample):
                question     = row.get("task_prompt", "")
                ground_truth = row.get("ground_truth", "")
                prediction   = row.get(pred_col, "")
                image_id     = row.get("image_id", "")

                premise = build_premise(dataset_name, question, ground_truth, prediction)
                category, reason = classify_one_v2(classifier, premise)

                model_labels.append(category)
                all_labels.append({
                    "dataset":         dataset_name,
                    "model":           model_key,
                    "image_id":        image_id,
                    "question":        question,
                    "ground_truth":    ground_truth,
                    "prediction":      prediction,
                    "is_empty_output": not bool(prediction.strip()),
                    "output_length":   len(prediction.split()),
                    "category":        category,
                    "reason":          reason,
                    "judge":           JUDGE_MODEL,
                })

                if (i + 1) % 20 == 0:
                    dist = {c: model_labels.count(c) for c in CATEGORIES}
                    log.info("  [%s|%s] %d/%d  dist=%s",
                             dataset_name, model_key, i + 1, len(sample), dist)

            aligned_labels[model_key] = model_labels
            dist = {c: model_labels.count(c) for c in CATEGORIES}
            log.info("[%s | %s] FINAL dist: %s", dataset_name, model_key, dist)

        # Cross-failure-set distributional kappa (see W5 / sec:judge)
        q = aligned_labels.get("qwen", [])
        s = aligned_labels.get("smolvlm2", [])
        max_len = max(len(q), len(s))
        q = q + ["E"] * (max_len - len(q))
        s = s + ["E"] * (max_len - len(s))
        kappa_result = compute_kappa(q, s)
        kappa_data[dataset_name]["qwen_vs_smolvlm2"] = kappa_result
        log.info("[%s] distributional κ = %s", dataset_name, kappa_result.get("kappa"))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  LOCAL LLM JUDGE — ERROR TAXONOMY SUMMARY")
    print(f"  Judge: {JUDGE_MODEL}")
    print("=" * 68)
    for ds, kappa_items in kappa_data.items():
        for comparison, kres in kappa_items.items():
            k = kres.get("kappa")
            k_str = f"{k:.4f}" if k is not None else "undefined"
            print(f"\n  Dataset: {ds}  |  {comparison}")
            print(f"    n={kres['n']}  κ={k_str}  ({kres['interpretation']})")
            for key, val in kres.items():
                if key.startswith("freq_"):
                    print(f"    {key}: {val}")
    print("=" * 68)

    OUTPUT_LABELS.write_text(json.dumps(all_labels, indent=2))
    log.info("Saved %d labelled examples → %s", len(all_labels), OUTPUT_LABELS)

    OUTPUT_KAPPA.write_text(json.dumps(kappa_data, indent=2))
    log.info("Saved kappa report → %s", OUTPUT_KAPPA)


if __name__ == "__main__":
    main()
