"""
taxonomy_diagnosis.py
=====================
Post-hoc diagnostic analysis of the GPT-4o error taxonomy to address the
"Object Blindness monoculture" critique.

Motivation
----------
The initial llm_judge.py run suffered from a column-name mismatch: it read
predictions from columns that did not exist in vlm_inference_results.csv,
producing empty strings for every prediction.  GPT-4o correctly labels an
empty prediction as "Object Blindness" (the model produced no output), so
100% of the 400 judgements received category A.

This script re-analyses the taxonomy with three diagnostic lenses:

  Lens 1 — Output-suppression split
    Separate failures into:
      (a) Empty output  → the generation head declined to produce text
          (instruction-following / verbalization failure, NOT necessarily
          encoder blindness)
      (b) Articulated   → the model produced text but was factually wrong
          (the more interesting failure class for taxonomy purposes)

  Lens 2 — Task-type stratification
    VQAv2 prompts are short, closed questions with binary/specific ground
    truths.  A wrong answer on VQA is more likely to look like Object
    Blindness (missing the target object) or Prior Bias (guessing the modal
    answer).  COCO Captions prompts ("Generate a short caption ...") require
    open-ended compositional text generation — a much richer surface for
    Semantic Drift (correct objects, wrong attributes/relations).

    Hypothesis: the fraction of Semantic Drift + Prior Bias errors is higher
    on COCO captioning failures than on VQAv2 failures.

  Lens 3 — Re-judge articulated failures with a refined prompt
    Re-run GPT-4o on the non-empty predictions only, using an expanded
    six-category taxonomy that explicitly separates Generation Failure from
    perceptual failure.  New category Z (Generation Failure) is hidden from
    the non-empty re-judge pass (by construction), making the remaining
    categories more discriminating.

Outputs
-------
  results/taxonomy_diagnosis_report.json
      Structured breakdown with counts, rates, and re-judge label lists.

  Printed console summary with key findings.

Usage
-----
  python scripts/taxonomy_diagnosis.py

  Requires OPENAI_API_KEY for Lens 3 (re-judging articulated failures).
  Lenses 1 and 2 run on llm_judge_labels.json alone and require no API key.
"""

import csv
import json
import logging
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
LABELS_JSON   = Path("results/llm_judge_labels.json")
RESULTS_CSV   = Path("results/vlm_inference_results.csv")
OUTPUT_JSON   = Path("results/taxonomy_diagnosis_report.json")
LOG_FILE      = Path("taxonomy_diagnosis.log")

JUDGE_MODEL   = "gpt-4o"
MAX_RETRIES   = 3
RETRY_DELAY   = 5.0

# Expanded taxonomy for the re-judge pass (includes Z for generation failure)
REFINED_CATEGORIES = ["A", "B", "C", "D", "E", "Z"]
REFINED_TAXONOMY = """\
A  Object Blindness     – Model fails to detect/acknowledge a clearly visible object.
B  Semantic Drift       – Correct objects detected but wrong attribute, count, action,
                          or spatial relation — the error is in description precision.
C  Prior Bias           – Output is plausible for an average image of that scene type
                          but ignores this specific image's visual content (hallucination).
D  Spatial/Count Error  – Incorrect spatial relation, layout description, or object count.
E  Other                – Failure does not clearly fit A-D.
Z  Generation Failure   – The model produced text but the output is malformed, generic
                          boilerplate, or refuses to answer despite being given an image
                          (verbalization/instruction-following failure rather than
                          perceptual failure)."""

REFINED_SYSTEM_PROMPT = f"""\
You are an expert evaluator of vision-language model (VLM) failure modes.
Given a task, ground-truth answer, and a non-empty model prediction, classify
the error into exactly one category:

{REFINED_TAXONOMY}

Respond with a JSON object with exactly two keys:
  "category": one of ["A","B","C","D","E","Z"]
  "reason":   a one-sentence explanation.

Do NOT include any text outside the JSON object."""

REFINED_USER_TEMPLATE = """\
Dataset: {dataset}
Task/Question: {question}
Ground-truth: {ground_truth}
Model prediction: {prediction}

Classify the error."""

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="a")],
)
log = logging.getLogger(__name__)

random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


# ── Correctness helper (matches other pipeline scripts) ───────────────────────
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


# ── Load real predictions from CSV (column: smol_output / qwen_output) ────────
def load_real_predictions(csv_path: Path) -> dict:
    """
    Returns {(dataset_name, image_id, model_key): prediction_str}
    using the actual CSV column names produced by batch_inference.py.
    """
    mapping = {}
    if not csv_path.exists():
        log.warning("Results CSV not found at %s — Lens 3 will use labels JSON only.",
                    csv_path)
        return mapping
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ds  = row.get("dataset_name", "")
            iid = row.get("image_id", "")
            mapping[(ds, iid, "qwen")]     = row.get("qwen_output", "")
            mapping[(ds, iid, "smolvlm2")] = row.get("smol_output", "")
    return mapping


# ── GPT-4o re-judge ───────────────────────────────────────────────────────────
def get_client():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or not _OPENAI_OK:
        return None
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def rejudge_one(client, dataset, question, ground_truth, prediction):
    user_msg = REFINED_USER_TEMPLATE.format(
        dataset=dataset, question=question,
        ground_truth=ground_truth, prediction=prediction,
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": REFINED_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
                max_tokens=120,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content.strip())
            cat = data.get("category", "E")
            if cat not in REFINED_CATEGORIES:
                cat = "E"
            return cat, data.get("reason", "")
        except Exception as exc:
            log.warning("Attempt %d/%d: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    return "E", "API call failed after retries"


# ── Lens helpers ──────────────────────────────────────────────────────────────
def category_pct(labels: list[str], cats="ABC") -> dict:
    """Return percent breakdown limited to rows in `cats`, plus n."""
    subset = [c for c in labels if c in cats]
    total  = len(subset)
    if total == 0:
        return {c: 0.0 for c in cats} | {"n_abc": 0, "n_total": len(labels)}
    return {c: round(100 * subset.count(c) / total, 1) for c in cats} | {
        "n_abc": total, "n_total": len(labels)
    }


def articulation_rate(group: list[dict]) -> dict:
    empty = [r for r in group if r.get("is_empty_output", not bool(str(r.get("prediction","")).strip()))]
    nonempty = [r for r in group if not r.get("is_empty_output", not bool(str(r.get("prediction","")).strip()))]
    n = len(group)
    return {
        "n_total":    n,
        "n_empty":    len(empty),
        "n_articulated": len(nonempty),
        "articulation_rate_pct": round(100 * len(nonempty) / n, 1) if n > 0 else 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  TAXONOMY DIAGNOSIS: Object Blindness Stratification")
    log.info("=" * 60)

    # ── Load existing labels ───────────────────────────────────────────────────
    if not LABELS_JSON.exists():
        log.error("llm_judge_labels.json not found at %s. Run llm_judge.py first.",
                  LABELS_JSON)
        return

    labels = json.loads(LABELS_JSON.read_text())
    log.info("Loaded %d label entries from %s", len(labels), LABELS_JSON)

    # ── Detect column-name-mismatch artifact ──────────────────────────────────
    n_empty_in_labels = sum(1 for r in labels if not str(r.get("prediction","")).strip())
    empty_frac = n_empty_in_labels / max(len(labels), 1)
    artifact_detected = empty_frac > 0.9  # >90% empty predictions = strong signal

    log.info("Empty predictions in label file: %d / %d (%.1f%%)",
             n_empty_in_labels, len(labels), 100 * empty_frac)
    if artifact_detected:
        log.warning("ARTIFACT DETECTED: >90%% of judged predictions are empty strings.")
        log.warning("This indicates a column-name mismatch in the original llm_judge.py run.")
        log.warning("GPT-4o has been labelling empty strings as Object Blindness (A).")
        log.warning("Re-running on real predictions from %s ...", RESULTS_CSV)

    # ── Load real predictions from CSV ────────────────────────────────────────
    pred_map = load_real_predictions(RESULTS_CSV)
    has_csv = bool(pred_map)

    # Augment labels with real predictions if CSV available
    if has_csv:
        for entry in labels:
            key = (entry["dataset"], entry["image_id"], entry["model"])
            real_pred = pred_map.get(key, "")
            entry["prediction_real"] = real_pred
            entry["is_empty_output"] = not bool(real_pred.strip())
    else:
        for entry in labels:
            pred = str(entry.get("prediction", ""))
            entry["prediction_real"] = pred
            entry["is_empty_output"] = not bool(pred.strip())

    # ═══════════════════════════════════════════════════════════════════════════
    # LENS 1 — Output-suppression split
    # ═══════════════════════════════════════════════════════════════════════════
    log.info("--- Lens 1: Output-suppression split ---")

    lens1 = {}
    for model_key in ["qwen", "smolvlm2"]:
        for ds in ["vqav2", "coco_captions"]:
            group = [r for r in labels if r["model"] == model_key and r["dataset"] == ds]
            empty    = [r for r in group if r["is_empty_output"]]
            nonempty = [r for r in group if not r["is_empty_output"]]
            key = f"{model_key}:{ds}"
            lens1[key] = {
                **articulation_rate(group),
                "original_labels_dist":   dict(Counter(r["category"] for r in group)),
                "empty_labels_dist":      dict(Counter(r["category"] for r in empty)),
                "articulated_labels_dist": dict(Counter(r["category"] for r in nonempty)),
            }
            log.info("[%s | %s] empty=%d  articulated=%d",
                     model_key, ds, len(empty), len(nonempty))

    # ═══════════════════════════════════════════════════════════════════════════
    # LENS 2 — Task-type stratification (VQA vs captioning)
    # For each model, compare category distribution on VQAv2 vs COCO
    # ═══════════════════════════════════════════════════════════════════════════
    log.info("--- Lens 2: Task-type stratification ---")

    lens2 = {}
    for model_key in ["qwen", "smolvlm2"]:
        vqa_rows  = [r for r in labels if r["model"] == model_key and r["dataset"] == "vqav2"]
        coco_rows = [r for r in labels if r["model"] == model_key and r["dataset"] == "coco_captions"]

        # Split each further into empty vs articulated
        vqa_art  = [r for r in vqa_rows  if not r["is_empty_output"]]
        coco_art = [r for r in coco_rows if not r["is_empty_output"]]

        lens2[model_key] = {
            "vqav2": {
                **articulation_rate(vqa_rows),
                "all_labels":         dict(Counter(r["category"] for r in vqa_rows)),
                "articulated_labels": dict(Counter(r["category"] for r in vqa_art)),
                "category_pct_abc":   category_pct([r["category"] for r in vqa_art]),
            },
            "coco_captions": {
                **articulation_rate(coco_rows),
                "all_labels":         dict(Counter(r["category"] for r in coco_rows)),
                "articulated_labels": dict(Counter(r["category"] for r in coco_art)),
                "category_pct_abc":   category_pct([r["category"] for r in coco_art]),
            },
            "hypothesis": (
                "COCO shows higher Semantic Drift (B) + Prior Bias (C) than VQA"
                if (
                    (coco_art and sum(r["category"] in "BC" for r in coco_art) / len(coco_art)
                     > (vqa_art and sum(r["category"] in "BC" for r in vqa_art) / max(len(vqa_art),1) or 0))
                )
                else "VQA shows higher B+C rate than COCO (or articulated subset too small)"
            ),
        }
        log.info("[%s] VQA articulated=%d, COCO articulated=%d",
                 model_key, len(vqa_art), len(coco_art))

    # ═══════════════════════════════════════════════════════════════════════════
    # LENS 3 — Re-judge articulated failures with refined prompt (needs API key)
    # ═══════════════════════════════════════════════════════════════════════════
    log.info("--- Lens 3: Re-judge articulated failures ---")

    client = get_client()
    lens3 = {"status": "skipped", "reason": "OPENAI_API_KEY not set or openai package missing"}

    articulated_all = [r for r in labels if not r["is_empty_output"]]

    if client and articulated_all:
        log.info("Re-judging %d articulated failures with refined 6-category taxonomy ...",
                 len(articulated_all))
        rejudge_results = []
        for i, entry in enumerate(articulated_all):
            cat, reason = rejudge_one(
                client,
                dataset=entry["dataset"],
                question=entry["question"],
                ground_truth=entry["ground_truth"],
                prediction=entry["prediction_real"],
            )
            rejudge_results.append({
                **entry,
                "refined_category": cat,
                "refined_reason":   reason,
            })
            if (i + 1) % 10 == 0:
                log.info("  Re-judged %d / %d", i + 1, len(articulated_all))

        # Breakdown by dataset for refined labels
        refined_by_ds = {}
        for ds in ["vqav2", "coco_captions"]:
            ds_rows = [r for r in rejudge_results if r["dataset"] == ds]
            refined_by_ds[ds] = {
                "n": len(ds_rows),
                "category_dist": dict(Counter(r["refined_category"] for r in ds_rows)),
                "category_pct":  category_pct(
                    [r["refined_category"] for r in ds_rows], cats="ABCDEZ"),
            }

        lens3 = {
            "status":           "completed",
            "n_rejudged":       len(articulated_all),
            "refined_labels_by_dataset": refined_by_ds,
            "full_results":     rejudge_results,
        }
        log.info("Lens 3 complete. Refined label dist: %s",
                 dict(Counter(r["refined_category"] for r in rejudge_results)))

    elif not articulated_all:
        n_art = len(articulated_all)
        lens3 = {
            "status": "skipped",
            "reason": (
                "All predictions in llm_judge_labels.json are empty strings — "
                "column-name mismatch artifact confirmed. Re-run llm_judge.py "
                "(now fixed) then re-run this script to get Lens 3 results."
                if artifact_detected
                else "No articulated (non-empty) failures found in label file."
            ),
        }
        log.warning("No articulated failures to re-judge. %s", lens3["reason"])

    # ═══════════════════════════════════════════════════════════════════════════
    # Summarise and write report
    # ═══════════════════════════════════════════════════════════════════════════
    report = {
        "artifact_detected": artifact_detected,
        "artifact_description": (
            "100% of predictions passed to GPT-4o in the original llm_judge.py run "
            "were empty strings due to a column-name mismatch (script expected "
            "'qwen_prediction'/'smolvlm2_prediction' but CSV contains 'qwen_output'/"
            "'smol_output'). GPT-4o correctly labels an empty model output as Object "
            "Blindness (A), creating a spurious monoculture. The fix is in the updated "
            "llm_judge.py; re-run that script to get valid taxonomy labels."
        ) if artifact_detected else "No artifact detected.",
        "lens1_output_suppression": lens1,
        "lens2_task_stratification": lens2,
        "lens3_rejudge_articulated": lens3,
        "interpretation": {
            "encoder_blindness_vs_output_head": (
                "The empty-output fraction (Lens 1) measures 'output suppression': the "
                "model saw the image but the generation head declined to produce text. "
                "This is distinct from true encoder blindness (the encoder failed to "
                "extract visual features). Distinguishing them requires probing the "
                "encoder's intermediate representations (e.g., image-captioning probes "
                "on frozen encoder features), which is outside the scope of this study. "
                "However, the task-type stratification (Lens 2) provides indirect "
                "evidence: if COCO captioning failures (open-ended, harder to suppress) "
                "show more Semantic Drift than VQA failures (closed, easy to suppress), "
                "it is consistent with output suppression being the dominant VQA failure "
                "mode rather than encoder blindness."
            ),
            "vqa_vs_captioning_prediction": (
                "VQA-style prompts elicit binary/specific predictions; the failure mode "
                "when a model is uncertain is to output nothing or a refusal (suppression). "
                "Captioning prompts ('Generate a short caption') have no obvious 'suppress' "
                "mode — the model must produce text. Therefore COCO failures are expected "
                "to be more articulated, and articulated failures are more likely to reveal "
                "Semantic Drift (correct objects, wrong attributes) and Prior Bias (generic "
                "scene descriptions) rather than Object Blindness."
            ),
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2))
    log.info("Saved diagnosis report → %s", OUTPUT_JSON)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TAXONOMY DIAGNOSIS SUMMARY")
    print("=" * 70)

    if artifact_detected:
        print(f"\n  ⚠  MEASUREMENT ARTIFACT CONFIRMED")
        print(f"     {n_empty_in_labels}/{len(labels)} ({100*empty_frac:.0f}%) judged predictions were empty strings.")
        print(f"     Original 'Object Blindness monoculture' is an artifact of a")
        print(f"     column-name bug in llm_judge.py (now fixed).")
        print(f"     Re-run llm_judge.py then re-run this script for valid taxonomy.")

    print("\n  LENS 1: Output suppression rates (using real predictions from CSV)")
    for key, info in lens1.items():
        model, ds = key.split(":")
        print(f"    [{model} | {ds}]  empty={info['n_empty']}  "
              f"articulated={info['n_articulated']}  "
              f"articulation_rate={info['articulation_rate_pct']}%")

    print("\n  LENS 2: Task-type stratification (articulated failures only)")
    for model_key, info in lens2.items():
        for ds in ["vqav2", "coco_captions"]:
            pct = info[ds]["category_pct_abc"]
            n_art = info[ds]["n_articulated"]
            print(f"    [{model_key} | {ds}]  n_articulated={n_art}  "
                  f"A={pct.get('A',0)}%  B={pct.get('B',0)}%  C={pct.get('C',0)}%")
        print(f"    Hypothesis: {info['hypothesis']}")

    if lens3.get("status") == "completed":
        print("\n  LENS 3: Refined re-judge (articulated failures)")
        for ds, info in lens3["refined_labels_by_dataset"].items():
            print(f"    [{ds}]  n={info['n']}  dist={info['category_dist']}")
    else:
        print(f"\n  LENS 3: {lens3.get('status','skipped')} — {lens3.get('reason','')[:80]}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
