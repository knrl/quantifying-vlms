"""
update_report.py
================
Phase 6: Update REPORT.md with real experiment numbers.

Reads all output artefacts produced by the pipeline and replaces every
XX.X% / X.XXX placeholder in REPORT.md with actual measured values.

Artefacts consumed
------------------
  vlm_inference_results.csv      -> Tables 1 & 2 (accuracy, error counts)
  robustness_report.json         -> Table 4 (blur robustness)
  calibration_results.json       -> Table 3 (ECE)
  negation_probes_summary.json   -> Table 5 (negation probes)
  cohen_kappa_report.json        -> Table 2 footnote (Cohen's κ)
"""

import csv
import json
import logging
import re
import sys
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
REPORT_MD      = Path("docs/REPORT.md")
RESULTS_CSV    = Path("results/vlm_inference_results.csv")
BLUR_JSON      = Path("results/robustness_report.json")
CALIB_JSON     = Path("results/calibration_results.json")
NEGATION_JSON  = Path("results/negation_probes_summary.json")
KAPPA_JSON     = Path("results/cohen_kappa_report.json")
LOG_FILE       = Path("update_report.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="a")],
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    log.warning("Missing: %s", path)
    return {}


def pct(value, decimals=1):
    """Format a float in [0,1] as e.g. '62.3%'."""
    return f"{value * 100:.{decimals}f}%"


def fmt(value, decimals=3):
    """Format a float to fixed decimals."""
    return f"{value:.{decimals}f}"


# ── Table 1+2: accuracy + error taxonomy from CSV ────────────────────────────
def compute_accuracy_table(csv_path: Path):
    """
    Returns nested dict:
      stats[dataset][model] = {
          "n": int, "correct": int, "accuracy": float,
          "errors": {"qwen_only": int, "smol_only": int, "both_wrong": int, "both_right": int}
      }
    """
    if not csv_path.exists():
        log.warning("Missing %s", csv_path)
        return {}

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    log.info("Loaded %d rows from %s", len(rows), csv_path)

    stats = defaultdict(lambda: defaultdict(lambda: {"n": 0, "correct": 0}))
    taxonomy = defaultdict(lambda: {"qwen_only": 0, "smol_only": 0,
                                    "both_wrong": 0, "both_right": 0})

    for r in rows:
        ds = r.get("dataset_name", "")
        qwen_ok = r.get("qwen_correct", "").strip().lower() in ("true", "1", "yes")
        smol_ok = r.get("smolvlm2_correct", "").strip().lower() in ("true", "1", "yes")

        for model_key, ok in [("qwen", qwen_ok), ("smolvlm2", smol_ok)]:
            stats[ds][model_key]["n"] += 1
            if ok:
                stats[ds][model_key]["correct"] += 1

        if qwen_ok and smol_ok:
            taxonomy[ds]["both_right"] += 1
        elif qwen_ok and not smol_ok:
            taxonomy[ds]["qwen_only"] += 1
        elif not qwen_ok and smol_ok:
            taxonomy[ds]["smol_only"] += 1
        else:
            taxonomy[ds]["both_wrong"] += 1

    # Compute accuracy floats
    for ds in stats:
        for model in stats[ds]:
            d = stats[ds][model]
            d["accuracy"] = d["correct"] / d["n"] if d["n"] else 0.0

    return dict(stats), dict(taxonomy)


# ── String replacement engine ─────────────────────────────────────────────────
class ReportPatcher:
    """Applies named text-substitutions to REPORT.md using regex anchors."""

    def __init__(self, text: str):
        self.text = text
        self.applied: list[str] = []
        self.missed:  list[str] = []

    def replace(self, pattern: str, replacement: str, label: str, flags=re.IGNORECASE):
        """
        Replace `pattern` (regex) with `replacement`.
        Logs whether the replacement was found.
        """
        new_text, n = re.subn(pattern, replacement, self.text, flags=flags)
        if n:
            self.text = new_text
            self.applied.append(f"{label} ({n}x)")
        else:
            self.missed.append(label)

    def result(self):
        return self.text


def patch_report(report_text: str, stats: dict, taxonomy: dict,
                 blur: dict, calib: dict, negation: dict, kappa: dict) -> str:

    p = ReportPatcher(report_text)

    # ── Table 1: Baseline Accuracy ──────────────────────────────────────────
    # Pattern matches table cells like | XX.X% | for each model×dataset combo.
    # We replace the first occurrence for each slot using a unique anchor.

    for ds_tag, ds_key in [("VQAv2", "vqav2"), ("COCO", "coco_captions")]:
        for model_tag, model_key in [("Qwen2.5-VL-7B", "qwen"), ("SmolVLM2", "smolvlm2")]:
            if ds_key in stats and model_key in stats[ds_key]:
                acc = stats[ds_key][model_key]["accuracy"]
                n   = stats[ds_key][model_key]["n"]
                val = pct(acc)
                # Anchor: table row containing both model_tag and ds_tag column
                p.replace(
                    rf"(\|\s*{re.escape(model_tag)}[^|]*\|[^|]*XX\.X%[^|]*\|)",
                    lambda m, v=val: m.group(0).replace("XX.X%", v, 1),
                    f"Table1 {model_tag}×{ds_tag} accuracy",
                )
                log.info("Table 1 | %s | %s | n=%d | acc=%s", model_key, ds_key, n, val)

    # ── Table 2: Error Taxonomy ─────────────────────────────────────────────
    # Table 2 now uses llm_judge kappa data (per-category frequency counts)
    # rather than the old accuracy-based taxonomy dict.
    # Patch XX.X% cells for categories A-C (rows: Object Blindness, Semantic Drift, Prior Bias)
    # and (XX) integer cells for categories D and E, using freq_qwen / freq_smolvlm2 from kappa.
    cat_row_map = {
        "Object Blindness (A)": "A",
        "Semantic Drift (B)":   "B",
        "Prior Bias (C)":       "C",
    }
    for ds_key in ["vqav2", "coco_captions"]:
        kd = kappa.get(ds_key, {}).get("qwen_vs_smolvlm2", {})
        freq_q = kd.get("freq_qwen", {})
        freq_s = kd.get("freq_smolvlm2", {})
        if not freq_q and not freq_s:
            continue
        # Compute A+B+C totals for percentage base
        abc_q = sum(freq_q.get(c, 0) for c in ("A", "B", "C")) or 1
        abc_s = sum(freq_s.get(c, 0) for c in ("A", "B", "C")) or 1

        for row_label, cat_key in cat_row_map.items():
            val_q = pct(freq_q.get(cat_key, 0) / abc_q)
            val_s = pct(freq_s.get(cat_key, 0) / abc_s)
            # Row anchor: line containing row_label in the markdown table
            # Two XX.X% cells per row: first = SmolVLM2, second = Qwen
            p.replace(
                rf"(\|\s*{re.escape(row_label)}\s*\|[^|]*)(XX\.X%)([^|]*\|[^|]*)(XX\.X%)",
                lambda m, vs=val_s, vq=val_q: m.group(0)
                    .replace("XX.X%", vs, 1)
                    .replace("XX.X%", vq, 1),
                f"Table2 {row_label}×{ds_key}",
            )

        # (XX) integer cells for D and E rows
        for row_label, cat_key in [("Spatial Error (D)", "D"), ("Other (E)", "E")]:
            n_q = freq_q.get(cat_key, 0)
            n_s = freq_s.get(cat_key, 0)
            p.replace(
                rf"(\|\s*{re.escape(row_label)}\s*\|[^|]*)\(XX\)([^|]*\|[^|]*)\(XX\)",
                lambda m, ns=str(n_s), nq=str(n_q): m.group(0)
                    .replace("(XX)", f"({ns})", 1)
                    .replace("(XX)", f"({nq})", 1),
                f"Table2 {row_label} counts×{ds_key}",
            )

    # Cohen's κ — Table 2 kappa rows (X.XX placeholder)
    for ds_key in ["vqav2", "coco_captions"]:
        qv = (kappa.get(ds_key, {})
                   .get("qwen_vs_smolvlm2", {})
                   .get("kappa", None))
        interp = (kappa.get(ds_key, {})
                       .get("qwen_vs_smolvlm2", {})
                       .get("interpretation", ""))
        if qv is not None:
            ds_label = "VQAv2" if ds_key == "vqav2" else "COCO"
            p.replace(
                rf"(\|\s*qwen_vs_smolvlm2\s*\({re.escape(ds_label)}\)[^|]*\|[^|]*)X\.XX",
                lambda m, v=f"{qv:.2f}": m.group(0).replace("X.XX", v, 1),
                f"Table2 kappa {ds_key}",
            )
            p.replace(
                rf"(\|\s*qwen_vs_smolvlm2\s*\({re.escape(ds_label)}\)[^|]*\|[^|]*{re.escape(str(round(qv, 2)))}[^|]*\|[^|]*)\[Moderate[^\]]*\]",
                lambda m, v=interp.capitalize(): m.group(0).__class__(
                    m.group(0)[: m.group(0).rfind("[")] + f"[{v}]"),
                f"Table2 kappa interp {ds_key}",
            )

    # ── Table 3: ECE ────────────────────────────────────────────────────────
    for model_tag, model_key in [("Qwen2.5-VL-7B", "qwen"), ("SmolVLM2", "smolvlm2")]:
        for ds_tag, ds_key in [("VQAv2", "vqav2"), ("COCO", "coco_captions")]:
            ece = (calib.get(model_key, {})
                        .get(ds_key, {})
                        .get("ece", None))
            if ece is not None:
                val = fmt(ece, 4)
                p.replace(
                    rf"(\|\s*{re.escape(model_tag)}[^|]*\|[^|]*0\.XXX)",
                    lambda m, v=val: m.group(0).replace("0.XXX", v, 1),
                    f"Table3 ECE {model_tag}×{ds_tag}",
                )

    # ── Table 4: Blur Robustness ─────────────────────────────────────────────
    # robustness_report.json structure (per existing robustness_blur.py):
    # {dataset: {model: {original_acc, blurred_acc, drop_pp, ratio}}}
    for ds_tag, ds_key in [("VQAv2", "vqav2"), ("COCO", "coco_captions")]:
        for model_tag, model_key in [("Qwen", "qwen"), ("SmolVLM2", "smolvlm2")]:
            entry = blur.get(ds_key, {}).get(model_key, {})
            if entry:
                orig = entry.get("original_acc", entry.get("original_accuracy", None))
                blur_acc = entry.get("blurred_acc", entry.get("blurred_accuracy", None))
                drop = entry.get("drop_pp", entry.get("drop", None))
                ratio = entry.get("ratio", entry.get("relative_ratio", None))
                if orig is not None:
                    p.replace(
                        rf"(\|\s*{re.escape(model_tag)}[^|]*\|[^|]*XX\.X%[^|]*XX\.X%)",
                        lambda m, ov=pct(orig), bv=pct(blur_acc) if blur_acc else "?":
                            m.group(0).replace("XX.X%", ov, 1).replace("XX.X%", bv, 1),
                        f"Table4 {model_tag}×{ds_key} orig/blur",
                    )
                if drop is not None:
                    p.replace(
                        rf"(\|\s*{re.escape(model_tag)}[^|]*\|[^|]*[+-]XX\.X)",
                        lambda m, v=f"{drop:+.1f}":
                            m.group(0).replace("+XX.X", v, 1).replace("-XX.X", v, 1),
                        f"Table4 {model_tag}×{ds_key} drop",
                    )

    # ── Table 5: Negation Probes ──────────────────────────────────────────────
    for ds_tag, ds_key in [("VQAv2", "vqav2"), ("COCO", "coco_captions")]:
        for model_tag, model_key in [("Qwen", "qwen"), ("SmolVLM2", "smolvlm2")]:
            ds_neg = negation.get(ds_key, {})
            # aggregate success rate across all templates
            rates = []
            for tk, per_model in ds_neg.items():
                sr = per_model.get(model_key, {}).get("success_rate", None)
                if sr is not None:
                    rates.append(sr)
            if rates:
                avg_sr = sum(rates) / len(rates)
                val = pct(avg_sr)
                p.replace(
                    rf"(\|\s*{re.escape(model_tag)}[^|]*\|[^|]*XX\.X%)",
                    lambda m, v=val: m.group(0).replace("XX.X%", v, 1),
                    f"Table5 {model_tag}×{ds_key}",
                )

    # ── Generic fallback: remaining XX.X% placeholders ──────────────────────
    remaining = len(re.findall(r"XX\.X%", p.result()))
    if remaining:
        log.warning("%d XX.X%% placeholders remain unfilled", remaining)
    remaining_val = len(re.findall(r"0\.XXX", p.result()))
    if remaining_val:
        log.warning("%d 0.XXX placeholders remain unfilled", remaining_val)

    # Report
    log.info("Substitutions applied  : %s", ", ".join(p.applied) if p.applied else "none")
    if p.missed:
        log.warning("Patterns not found     : %s", ", ".join(p.missed))

    return p.result()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  Phase 6: Update REPORT.md with real results")
    log.info("=" * 60)

    if not REPORT_MD.exists():
        log.error("REPORT.md not found — nothing to update")
        sys.exit(1)

    report_text = REPORT_MD.read_text(encoding="utf-8")

    # Load all artefacts
    stats, taxonomy = ({}, {})
    if RESULTS_CSV.exists():
        stats, taxonomy = compute_accuracy_table(RESULTS_CSV)
    else:
        log.warning("Missing %s — Tables 1 & 2 will not be updated", RESULTS_CSV)

    blur     = load_json(BLUR_JSON)
    calib    = load_json(CALIB_JSON)
    negation = load_json(NEGATION_JSON)
    kappa    = load_json(KAPPA_JSON)

    # Patch report
    patched = patch_report(report_text, stats, taxonomy, blur, calib, negation, kappa)

    # Count remaining placeholders
    n_remaining_pct = len(re.findall(r"XX\.X%", patched))
    n_remaining_val = len(re.findall(r"0\.XXX|X\.XXX", patched))

    # Write back
    REPORT_MD.write_text(patched, encoding="utf-8")
    log.info("REPORT.md updated successfully")

    # Print summary
    print("\n" + "=" * 60)
    print("  REPORT UPDATE SUMMARY")
    print("=" * 60)

    # Accuracy table
    if stats:
        print(f"\n{'Model':<22} {'Dataset':<16} {'N':>6} {'Correct':>8} {'Accuracy':>10}")
        print("-" * 66)
        for ds_key in ["vqav2", "coco_captions"]:
            for model_key in ["qwen", "smolvlm2"]:
                label = "Qwen2.5-VL-7B" if model_key == "qwen" else "SmolVLM2-500M"
                d = stats.get(ds_key, {}).get(model_key, {})
                n = d.get("n", 0)
                c = d.get("correct", 0)
                acc = d.get("accuracy", 0.0)
                print(f"  {label:<20} {ds_key:<16} {n:>6} {c:>8} {acc:>9.1%}")

    # Taxonomy
    if taxonomy:
        print(f"\nError Taxonomy (fraction of all samples):")
        print(f"{'Dataset':<16} {'Qwen-only':>10} {'Smol-only':>10} {'Both-wrong':>11} {'Both-right':>11}")
        print("-" * 62)
        for ds_key in ["vqav2", "coco_captions"]:
            t = taxonomy.get(ds_key, {})
            total = sum(t.values()) or 1
            print(f"  {ds_key:<14} "
                  f"{t.get('qwen_only',0)/total:>9.1%} "
                  f"{t.get('smol_only',0)/total:>9.1%} "
                  f"{t.get('both_wrong',0)/total:>10.1%} "
                  f"{t.get('both_right',0)/total:>10.1%}")

    if n_remaining_pct or n_remaining_val:
        print(f"\nWARNING: {n_remaining_pct} XX.X% and {n_remaining_val} X.XXX placeholders "
              f"still unfilled (artefacts missing?)")
    else:
        print("\nAll placeholders filled successfully.")

    print("=" * 60)


if __name__ == "__main__":
    main()
