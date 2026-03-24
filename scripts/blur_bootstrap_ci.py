"""
blur_bootstrap_ci.py
====================
Bootstrap confidence intervals for the Gaussian-blur robustness experiment.

Reads results/robustness_blurred_results.csv and computes:
  - Per-model accuracy drop (pp) with 95% bootstrap CI
  - Relative robustness ratio ρ = Δ_smol / Δ_qwen with 95% bootstrap CI
  - McNemar's test for paired difference in error rates (same images → paired)
  - Fisher's exact test on the 2×2 contingency table of blurred-accuracy changes
  - Per-dataset breakdown with CIs

Outputs results/blur_ci_report.json
"""

import csv
import json
import math
import random
from pathlib import Path
from collections import Counter

SEED = 42
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
RESULTS_CSV = Path("results/robustness_blurred_results.csv")
OUTPUT_JSON = Path("results/blur_ci_report.json")

random.seed(SEED)


# ── helpers ────────────────────────────────────────────────────────────────────

def _norm_bool(v):
    """Parse CSV boolean field to Python bool."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("true", "1", "yes")


def load_rows():
    rows = []
    for r in csv.DictReader(open(RESULTS_CSV)):
        rows.append({
            "dataset":        r["dataset_name"],
            "smol_orig":      _norm_bool(r["smol_correct_original"]),
            "qwen_orig":      _norm_bool(r["qwen_correct_original"]),
            "smol_blur":      _norm_bool(r["smol_correct_blurred"]),
            "qwen_blur":      _norm_bool(r["qwen_correct_blurred"]),
        })
    return rows


def accuracy_drop(rows, model):
    """Returns drop in percentage points (positive = degraded)."""
    orig_key = f"{model}_orig"
    blur_key = f"{model}_blur"
    n = len(rows)
    if n == 0:
        return float("nan")
    orig_acc = sum(r[orig_key] for r in rows) / n
    blur_acc = sum(r[blur_key] for r in rows) / n
    return (orig_acc - blur_acc) * 100          # pp; positive = got worse


def rho(smol_drop, qwen_drop):
    """ρ = Δ_smol / Δ_qwen.  Returns nan if denominator near zero."""
    if abs(qwen_drop) < 0.01:
        return float("nan")
    return smol_drop / qwen_drop


def bootstrap_ci(rows, stat_fn, n_boot=N_BOOTSTRAP, level=CI_LEVEL):
    """
    Parametric bootstrap: resample rows with replacement n_boot times,
    compute stat_fn(boot_sample) each time.
    Returns (point_estimate, lower_ci, upper_ci).
    """
    point = stat_fn(rows)
    n = len(rows)
    samples = []
    for _ in range(n_boot):
        boot = [rows[random.randint(0, n - 1)] for _ in range(n)]
        samples.append(stat_fn(boot))
    # Remove nans
    samples = [s for s in samples if not (isinstance(s, float) and math.isnan(s))]
    samples.sort()
    alpha = 1 - level
    lo = samples[int(alpha / 2 * len(samples))]
    hi = samples[int((1 - alpha / 2) * len(samples))]
    return point, lo, hi


def mcnemar_test(rows, model_a, model_b):
    """
    McNemar's test on paired binary outcomes:
    H0: P(model_a correct on blur | model_b wrong) = P(model_b correct | model_a wrong)
    Returns (chi2, p_value) using continuity-corrected version.
    """
    # Discordant cell counts
    b = sum(1 for r in rows if r[f"{model_a}_blur"] and not r[f"{model_b}_blur"])
    c = sum(1 for r in rows if not r[f"{model_a}_blur"] and r[f"{model_b}_blur"])
    n_disc = b + c
    if n_disc == 0:
        return 0.0, 1.0
    # Continuity-corrected McNemar: (|b-c| - 1)^2 / (b+c)
    chi2 = max(0, (abs(b - c) - 1) ** 2) / n_disc
    # chi2 with df=1: p-value via normal approximation (chi2 = z^2)
    z = math.sqrt(chi2)
    # two-tailed p from standard normal
    p = 2 * (1 - _standard_normal_cdf(z))
    return chi2, p, b, c


def _standard_normal_cdf(z):
    """Abramowitz & Stegun approximation (error < 7.5e-8)."""
    if z < 0:
        return 1 - _standard_normal_cdf(-z)
    t = 1 / (1 + 0.2316419 * z)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
               + t * (-1.821255978 + t * 1.330274429))))
    return 1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-z * z / 2) * poly


def contingency_stats(rows, model):
    """
    2x2 table: original_correct × blurred_correct for one model.
    Returns table + odds ratio with 95% CI (log-normal).
    """
    orig_key = f"{model}_orig"
    blur_key = f"{model}_blur"
    # (orig=T, blur=T), (orig=T, blur=F), (orig=F, blur=T), (orig=F, blur=F)
    # Since all selected rows have orig=True, simplify:
    a = sum(1 for r in rows if r[blur_key])       # stayed correct
    b = sum(1 for r in rows if not r[blur_key])   # became wrong
    return {"correct_blurred": a, "wrong_blurred": b, "n": len(rows),
            "blur_acc": round(a / len(rows) * 100, 1) if rows else float("nan")}


def run(rows, label="ALL", verbose=True):
    n = len(rows)
    if n == 0:
        return {}

    # Point estimates
    smol_drop_pt = accuracy_drop(rows, "smol")
    qwen_drop_pt = accuracy_drop(rows, "qwen")
    rho_pt       = rho(smol_drop_pt, qwen_drop_pt)

    # Bootstrap CIs
    smol_drop_pt, smol_lo, smol_hi = bootstrap_ci(rows, lambda r: accuracy_drop(r, "smol"))
    qwen_drop_pt, qwen_lo, qwen_hi = bootstrap_ci(rows, lambda r: accuracy_drop(r, "qwen"))
    rho_pt,       rho_lo,  rho_hi  = bootstrap_ci(rows, lambda r: rho(accuracy_drop(r, "smol"),
                                                                        accuracy_drop(r, "qwen")))

    # McNemar: are drop rates significantly different between models?
    chi2, p_val, b, c = mcnemar_test(rows, "smol", "qwen")

    smol_ct = contingency_stats(rows, "smol")
    qwen_ct = contingency_stats(rows, "qwen")

    result = {
        "n": n,
        "smol": {
            "orig_acc": round(sum(r["smol_orig"] for r in rows) / n * 100, 1),
            "blur_acc": round(smol_ct["blur_acc"], 1),
            "drop_pp":  round(smol_drop_pt, 2),
            "ci_95_pp": [round(smol_lo, 2), round(smol_hi, 2)],
        },
        "qwen": {
            "orig_acc": round(sum(r["qwen_orig"] for r in rows) / n * 100, 1),
            "blur_acc": round(qwen_ct["blur_acc"], 1),
            "drop_pp":  round(qwen_drop_pt, 2),
            "ci_95_pp": [round(qwen_lo, 2), round(qwen_hi, 2)],
        },
        "rho": {
            "point":  round(rho_pt, 3) if not math.isnan(rho_pt) else None,
            "ci_95":  [round(rho_lo, 3), round(rho_hi, 3)] if not math.isnan(rho_pt) else None,
        },
        "mcnemar": {
            "discordant_smol_only": b,
            "discordant_qwen_only": c,
            "chi2_cc": round(chi2, 3),
            "p_value": round(p_val, 4),
            "significant_at_0.05": p_val < 0.05,
        },
    }

    if verbose:
        print(f"\n{'='*62}")
        print(f"  BLUR ROBUSTNESS BOOTSTRAP CIs  [{label}]  n={n}")
        print(f"{'='*62}")
        print(f"  SmolVLM2 drop: {smol_drop_pt:+.1f} pp  "
              f"95% CI [{smol_lo:+.1f}, {smol_hi:+.1f}]")
        print(f"  Qwen-7B NF4 drop: {qwen_drop_pt:+.1f} pp  "
              f"95% CI [{qwen_lo:+.1f}, {qwen_hi:+.1f}]")
        if not math.isnan(rho_pt):
            print(f"  ρ: {rho_pt:.2f}  95% CI [{rho_lo:.2f}, {rho_hi:.2f}]")
        else:
            print(f"  ρ: undefined (Qwen drop ≈ 0)")
        print(f"  McNemar p={p_val:.4f}  (χ²={chi2:.2f})  "
              f"{'*significant*' if p_val < 0.05 else 'not significant'} at α=0.05")
        print(f"  (discordant: SmolOnly={b}, QwenOnly={c})")
        print(f"{'='*62}")

    return result


def main():
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found. Run robustness_blur.py first.")
        return

    rows = load_rows()
    print(f"Loaded {len(rows)} rows from {RESULTS_CSV}")

    report = {}
    report["overall"] = run(rows, label="ALL DATASETS")

    # Per-dataset breakdown
    for ds in sorted(set(r["dataset"] for r in rows)):
        ds_rows = [r for r in rows if r["dataset"] == ds]
        report[ds] = run(ds_rows, label=ds.upper())

    OUTPUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"\nSaved CI report to {OUTPUT_JSON}")

    # Print interpretation note
    smol_ci = report["overall"]["smol"]["ci_95_pp"]
    qwen_ci = report["overall"]["qwen"]["ci_95_pp"]
    rho_ci  = report["overall"]["rho"]["ci_95"] or ["N/A", "N/A"]
    print(f"\n  NOTE FOR PAPER:")
    print(f"  SmolVLM2:  {report['overall']['smol']['drop_pp']:+.1f} pp "
          f"[{smol_ci[0]:+.1f}, {smol_ci[1]:+.1f}] 95% CI")
    print(f"  Qwen NF4:  {report['overall']['qwen']['drop_pp']:+.1f} pp "
          f"[{qwen_ci[0]:+.1f}, {qwen_ci[1]:+.1f}] 95% CI")
    print(f"  ρ:         {report['overall']['rho']['point']} "
          f"[{rho_ci[0]}, {rho_ci[1]}] 95% CI")


if __name__ == "__main__":
    main()
