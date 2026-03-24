"""
heuristic_judge.py
==================
Rule-based error taxonomy classifier for VLM failure analysis.
Produces the same output format as llm_judge.py but without an external API.

Decision tree (applied in order):
  D  Spatial/Count   — question asks about position, count, or direction
  A  Object Blindness — key ground-truth token absent from prediction,
                        or prediction explicitly denies presence
  B  Semantic Drift  — GT object mentioned but with wrong attributes
  C  Prior Bias      — prediction is verbose/generic, question focus ignored
  E  Other           — fallthrough

Limitations vs GPT-4o: no access to the image; cannot verify visual claims
directly. Categories are inferred from the text of question, ground truth,
and prediction alone.  Results are disclosed as "rule-based heuristic" in the
paper (§3.4 / §4.1).

Outputs: results/llm_judge_labels.json, results/cohen_kappa_report.json
"""

import csv
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
RESULTS_CSV   = Path("results/vlm_inference_results.csv")
OUTPUT_LABELS = Path("results/llm_judge_labels.json")
OUTPUT_KAPPA  = Path("results/cohen_kappa_report.json")
LOG_FILE      = Path("llm_judge.log")
N_JUDGE_SAMPLES = 100
CATEGORIES = ["A", "B", "C", "D", "E"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="a")],
)
log = logging.getLogger(__name__)
random.seed(SEED)

# ── Stopwords to exclude from keyword matching ────────────────────────────────
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "it", "its", "he", "she", "they", "we", "you", "i", "me",
    "him", "her", "them", "us", "what", "which", "who", "how", "when",
    "where", "why", "there", "here", "with", "from", "into", "onto",
    "upon", "and", "or", "but", "not", "no", "in", "on", "at", "to",
    "for", "of", "by", "as", "up", "out", "yes", "any", "some",
}

# ── Spatial / count keyword lists ────────────────────────────────────────────
SPATIAL_Q_WORDS = {
    "how many", "count", "number of", "where is", "where are",
    "which side", "left", "right", "next to", "beside", "between",
    "in front", "behind", "above", "below", "under", "on top",
    "position", "location", "direction", "facing",
}

NEGATION_WORDS = {
    "no ", " no ", "not ", " not ", "cannot", "can't", "don't",
    "doesn't", "didn't", "none", "nothing", "nobody", "nowhere",
    "unable", "fail", "missing", "absent", "invisible",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_prediction(text: str) -> str:
    """Strip SmolVLM2 chat-template artefact 'Assistant:' prefix.

    batch_inference.py decodes the full sequence (input+output) for SmolVLM2
    and splits on the raw prompt text.  What remains starts with the role
    marker 'Assistant: ' emitted by apply_chat_template.  Stripping it
    ensures the taxonomy classifier does not treat 'assistant' as a content
    token in pred_tokens, which would suppress correct A/B classification.
    """
    stripped = text.strip()
    # Handle both 'Assistant: ...' and 'Assistant:\n...'
    if stripped.lower().startswith("assistant:"):
        stripped = stripped[len("assistant:"):].strip()
    return stripped


def is_correct(dataset_name: str, prediction: str, ground_truth: str) -> bool:
    pred = _norm(prediction)
    gt   = _norm(ground_truth)
    if not pred or not gt:
        return False
    if dataset_name == "vqav2":
        return gt in pred or pred in gt
    gt_words = [w for w in gt.split() if len(w) >= 3]
    return any(w in pred for w in gt_words)


def key_tokens(text: str, min_len: int = 3) -> list[str]:
    """Return content words longer than min_len not in stopwords."""
    return [w for w in _norm(text).split()
            if len(w) >= min_len and w not in STOPWORDS]


def any_in(needles: set, text: str) -> bool:
    return any(needle in text for needle in needles)


# ── Core taxonomy rule ────────────────────────────────────────────────────────
def classify_failure(dataset: str, question: str,
                     ground_truth: str, prediction: str) -> tuple[str, str]:
    q    = _norm(question)
    gt   = _norm(ground_truth)
    pred = _norm(prediction)

    gt_tokens   = key_tokens(gt)
    pred_tokens = key_tokens(pred)
    q_tokens    = key_tokens(q)

    # ── D: Spatial / Count ──────────────────────────────────────────────────
    if any_in(SPATIAL_Q_WORDS, q):
        return "D", f"Spatial/count question (q='{question[:60]}'); wrong answer"

    # ── A: Object Blindness ─────────────────────────────────────────────────
    # Case 1: explicit negation in prediction
    if any_in(NEGATION_WORDS, f" {pred} "):
        # Only classify A if the ground truth IS something (not a negation answer)
        if gt not in ("no", "yes", "none"):
            return "A", "Prediction contains explicit negation of the target object"

    # Case 2: ground-truth key tokens entirely absent from prediction
    if gt_tokens:
        overlap = sum(1 for tok in gt_tokens if tok in pred_tokens)
        if overlap == 0:
            # Also check: is the question's subject absent from the prediction?
            q_overlap = sum(1 for tok in q_tokens if tok in pred_tokens)
            if q_overlap == 0:
                return "A", (
                    f"No GT tokens ({gt_tokens[:3]}) or question tokens "
                    f"found in prediction"
                )

    # ── B: Semantic Drift ───────────────────────────────────────────────────
    # The right kind of object is described but with wrong attributes
    # Proxy: gt tokens partially present but answer is wrong
    if gt_tokens:
        overlap = sum(1 for tok in gt_tokens if tok in pred_tokens)
        # partial overlap = subject recognised, wrong details
        if overlap > 0:
            return "B", (
                f"GT tokens partially present ({overlap}/{len(gt_tokens)}) "
                "but answer incorrect — semantic attribute mismatch"
            )
        # Also: question subject present but gt mismatch
        q_in_pred = sum(1 for tok in q_tokens if tok in pred_tokens)
        if q_in_pred > 0:
            return "B", (
                "Question subject found in prediction but answer wrong "
                "— likely wrong attribute/category"
            )

    # ── C: Prior Bias ───────────────────────────────────────────────────────
    # Long, verbose prediction that ignores question focus
    pred_word_count = len(pred.split())
    if pred_word_count > 20:
        return "C", (
            f"Long prediction ({pred_word_count} words) with no GT token match "
            "— likely prior-biased verbose answer"
        )

    # Fallthrough
    return "E", "Did not match specific failure pattern A-D"


# ── Cohen's κ ─────────────────────────────────────────────────────────────────
def compute_kappa(labels_a, labels_b, name_a="qwen", name_b="smolvlm2"):
    n = min(len(labels_a), len(labels_b))
    la, lb = labels_a[:n], labels_b[:n]
    if len(set(la + lb)) <= 1:
        return {
            "n": n, "kappa": None,
            "interpretation": "undefined (degenerate — all labels identical)",
            f"freq_{name_a}": dict(Counter(la)),
            f"freq_{name_b}": dict(Counter(lb)),
        }
    kappa = cohen_kappa_score(la, lb, labels=CATEGORIES)
    def interpret(k):
        if k < 0:    return "less than chance"
        if k < 0.20: return "slight"
        if k < 0.40: return "fair"
        if k < 0.60: return "moderate"
        if k < 0.80: return "substantial"
        return "almost perfect"
    return {
        "n": n, "kappa": round(kappa, 4),
        "interpretation": interpret(kappa),
        f"freq_{name_a}": {c: la.count(c) for c in CATEGORIES},
        f"freq_{name_b}": {c: lb.count(c) for c in CATEGORIES},
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  Heuristic Judge — rule-based taxonomy (no API required)")
    log.info("=" * 60)

    rows = list(csv.DictReader(open(RESULTS_CSV, encoding="utf-8")))
    log.info("Loaded %d rows", len(rows))

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
            log.info("[%s | %s] %d failures → sampling %d",
                     dataset_name, model_key, len(failures), len(sample))

            model_labels = []
            for row in sample:
                question     = row.get("task_prompt", "")
                ground_truth = row.get("ground_truth", "")
                raw_pred     = row.get(pred_col, "")
                image_id     = row.get("image_id", "")
                # Strip SmolVLM2's 'Assistant: ' chat-template prefix before
                # taxonomy classification so the token 'assistant' does not
                # pollute pred_tokens and interfere with A/B/C decisions.
                prediction   = clean_prediction(raw_pred)

                category, reason = classify_failure(
                    dataset_name, question, ground_truth, prediction)
                model_labels.append(category)
                all_labels.append({
                    "dataset": dataset_name, "model": model_key,
                    "image_id": image_id, "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,           # cleaned
                    "raw_prediction": raw_pred,         # original from CSV
                    "is_empty_output": not bool(prediction.strip()),
                    "output_length": len(prediction.split()),
                    "category": category, "reason": reason,
                    "judge": "heuristic-rule-based",
                })

            aligned_labels[model_key] = model_labels
            dist = {c: model_labels.count(c) for c in CATEGORIES}
            log.info("[%s | %s] dist: %s", dataset_name, model_key, dist)

        q_labels = aligned_labels.get("qwen", [])
        s_labels = aligned_labels.get("smolvlm2", [])
        max_n = max(len(q_labels), len(s_labels))
        q_labels += ["E"] * (max_n - len(q_labels))
        s_labels += ["E"] * (max_n - len(s_labels))
        kappa_result = compute_kappa(q_labels, s_labels)
        kappa_data[dataset_name]["qwen_vs_smolvlm2"] = kappa_result
        log.info("[%s] distributional κ = %s (%s)",
                 dataset_name, kappa_result.get("kappa"),
                 kappa_result.get("interpretation"))

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  HEURISTIC JUDGE — TAXONOMY SUMMARY")
    print("=" * 68)
    for ds, kappa_items in kappa_data.items():
        for comparison, kres in kappa_items.items():
            k = kres.get("kappa")
            k_str = f"{k:.4f}" if k is not None else "undefined"
            print(f"\n  {ds}  |  {comparison}")
            print(f"    n={kres['n']}  κ={k_str}  ({kres['interpretation']})")
            for key, val in kres.items():
                if key.startswith("freq_"):
                    print(f"    {key}: {val}")
    print("=" * 68)

    OUTPUT_LABELS.write_text(json.dumps(all_labels, indent=2))
    log.info("Saved %d labels → %s", len(all_labels), OUTPUT_LABELS)
    OUTPUT_KAPPA.write_text(json.dumps(kappa_data, indent=2))
    log.info("Saved kappa report → %s", OUTPUT_KAPPA)


if __name__ == "__main__":
    main()
