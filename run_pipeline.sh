#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh
# Edge Reliability Gap in Vision-Language Models — Full Experiment Pipeline
#
# Usage:
#   bash run_pipeline.sh [OPTIONS]
#
# Options:
#   --skip-batch        Skip Phase 1 (batch inference) if CSV already exists
#   --skip-blur         Skip Phase 2 (Gaussian blur robustness)
#   --skip-precision    Skip Phase 2b (FP16 vs NF4 precision ablation)
#   --skip-strat-rob    Skip Phase 2c (stratified blur robustness across all 4 strata)
#   --skip-ece          Skip Phase 3 (calibration ECE)
#   --skip-negation     Skip Phase 4 (negation probes)
#   --skip-llm          Skip Phase 5 (LLM-as-judge; also skipped if no API key)
#   --skip-report       Skip Phase 6 (update REPORT.md with real numbers)
#   --skip-taxonomy-diag Skip Phase 6b (taxonomy diagnosis; Lens 1-2 need no key)
#   --results-csv PATH  Use a custom path for vlm_inference_results.csv
#                       (default: vlm_inference_results.csv)
#   --work-dir PATH     Working directory (default: script location)
#   --no-color          Disable ANSI color output
#   --no-push           Commit after each phase but do NOT push to remote
#   --no-git            Disable all git operations (commit + push)
#   -h, --help          Show this help
#
# Environment Variables:
#   OPENAI_API_KEY      Required for Phase 5 (LLM-as-judge). If unset,
#                       Phase 5 is skipped automatically.
#   HF_HOME             Optional: Hugging Face cache directory override.
#   GIT_REMOTE          Remote name to push to (default: origin).
#   GIT_BRANCH          Branch to push to (default: current branch).
#
# Exit Codes:
#   0  All requested phases completed successfully
#   1  A required phase failed (subsequent phases are skipped)
#   2  Bad arguments
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
_USE_COLOR=1
_col() { [ "$_USE_COLOR" -eq 1 ] && printf "\033[%sm" "$1" || true; }
RED=$(_col "31"); GRN=$(_col "32"); YLW=$(_col "33")
BLU=$(_col "34"); MAG=$(_col "35"); CYN=$(_col "36"); RST=$(_col "0")

log()  { printf "${BLU}[pipeline]${RST} %s\n" "$*"; }
ok()   { printf "${GRN}[  OK  ]${RST} %s\n" "$*"; }
warn() { printf "${YLW}[ WARN ]${RST} %s\n" "$*"; }
fail() { printf "${RED}[ FAIL ]${RST} %s\n" "$*" >&2; }
sep()  { printf "${CYN}%s${RST}\n" "══════════════════════════════════════════════════════════════"; }

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR"
RESULTS_CSV="results/vlm_inference_results.csv"

SKIP_BATCH=0
SKIP_BLUR=0
SKIP_PRECISION=0
SKIP_STRAT_ROB=0
SKIP_ECE=0
SKIP_NEGATION=0
SKIP_LLM=0
SKIP_REPORT=0
SKIP_TAXONOMY_DIAG=0
NO_PUSH=0
NO_GIT=0
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-}"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-batch)    SKIP_BATCH=1 ;;
        --skip-blur)     SKIP_BLUR=1 ;;
        --skip-precision) SKIP_PRECISION=1 ;;
        --skip-strat-rob) SKIP_STRAT_ROB=1 ;;
        --skip-ece)      SKIP_ECE=1 ;;
        --skip-negation) SKIP_NEGATION=1 ;;
        --skip-llm)      SKIP_LLM=1 ;;
        --skip-report)   SKIP_REPORT=1 ;;
        --skip-taxonomy-diag) SKIP_TAXONOMY_DIAG=1 ;;
        --results-csv)   RESULTS_CSV="$2"; shift ;;
        --work-dir)      WORK_DIR="$2"; shift ;;
        --no-color)      _USE_COLOR=0 ;;
        --no-push)       NO_PUSH=1 ;;
        --no-git)        NO_GIT=1 ;;
        -h|--help)
            sed -n '/^# =/,/^# =/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) fail "Unknown option: $1"; exit 2 ;;
    esac
    shift
done

# ── Environment checks ────────────────────────────────────────────────────────
cd "$WORK_DIR"
PYTHON=${PYTHON:-python}

sep
log "Edge Reliability Gap — Experiment Pipeline"
log "Working dir : $WORK_DIR"
log "Python      : $($PYTHON --version 2>&1)"
log "Results CSV : $RESULTS_CSV"
log "Date        : $(date '+%Y-%m-%d %H:%M:%S')"
sep

# Check GPU
if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))")
    VRAM=$($PYTHON -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1024**3,1))")
    ok "GPU: $GPU  (${VRAM} GiB)"
else
    warn "No CUDA GPU detected — inference will run on CPU (very slow)"
fi

# Check required packages
MISSING=""
for pkg in torch transformers datasets bitsandbytes accelerate num2words pillow matplotlib sklearn; do
    $PYTHON -c "import ${pkg//-/_}" 2>/dev/null || MISSING="$MISSING $pkg"
done
if [[ -n "$MISSING" ]]; then
    fail "Missing Python packages:$MISSING"
    fail "Run: pip install$MISSING"
    exit 1
fi
ok "All required Python packages present"

# ── Git helpers ───────────────────────────────────────────────────────────────
# Resolve push branch once (use current branch if GIT_BRANCH not set)
if [[ -z "$GIT_BRANCH" ]]; then
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
fi

git_commit_phase() {
    # Usage: git_commit_phase <phase_num> <phase_name> [files_to_add...]
    local num="$1" name="$2"
    shift 2
    local files=("$@")

    [[ "$NO_GIT" -eq 1 ]] && return 0

    # Stage specific artefact files if provided, else stage everything new
    if [[ ${#files[@]} -gt 0 ]]; then
        local existing=()
        for f in "${files[@]}"; do
            [[ -f "$f" ]] && existing+=("$f")
        done
        [[ ${#existing[@]} -gt 0 ]] && git add -- "${existing[@]}"
    else
        git add -A
    fi

    # Only commit if there is actually something staged
    if git diff --cached --quiet; then
        warn "Phase $num: nothing new to commit"
        return 0
    fi

    local msg="results(phase-$num): $name [$(date '+%Y-%m-%d %H:%M')]"
    if git commit -m "$msg"; then
        ok "Committed: $msg"
    else
        warn "git commit failed for phase $num (non-fatal)"
        return 0
    fi

    if [[ "$NO_PUSH" -eq 1 ]]; then
        warn "Phase $num: --no-push set, skipping git push"
        return 0
    fi

    if git push "$GIT_REMOTE" "$GIT_BRANCH"; then
        ok "Pushed to $GIT_REMOTE/$GIT_BRANCH"
    else
        warn "git push failed for phase $num (non-fatal — results are committed locally)"
    fi
}


PHASE_PASS=0
PHASE_FAIL=0
PHASE_SKIP=0
declare -A PHASE_STATUS
declare -A PHASE_DURATION

run_phase() {
    local num="$1" name="$2" script="$3"
    shift 3
    local extra_args=("$@")

    sep
    log "Phase $num: $name"

    if [[ ! -f "$script" ]]; then
        fail "$script not found — aborting"
        PHASE_STATUS[$num]="MISSING"
        ((PHASE_FAIL++)) || true
        return 1
    fi

    local t0=$SECONDS
    if $PYTHON "$script" "${extra_args[@]}"; then
        local dt=$(( SECONDS - t0 ))
        ok "Phase $num completed in ${dt}s"
        PHASE_STATUS[$num]="OK"
        PHASE_DURATION[$num]="${dt}s"
        ((PHASE_PASS++)) || true
        # Git commit is called by the caller with phase-specific file list
    else
        local code=$?
        local dt=$(( SECONDS - t0 ))
        fail "Phase $num FAILED (exit $code) after ${dt}s"
        PHASE_STATUS[$num]="FAIL"
        PHASE_DURATION[$num]="${dt}s"
        ((PHASE_FAIL++)) || true
        return 1
    fi
}

skip_phase() {
    local num="$1" name="$2" reason="$3"
    warn "Phase $num SKIPPED — $reason ($name)"
    PHASE_STATUS[$num]="SKIP"
    PHASE_DURATION[$num]="—"
    ((PHASE_SKIP++)) || true
}

# ── Phase 1: Batch Inference ──────────────────────────────────────────────────
if [[ "$SKIP_BATCH" -eq 1 ]]; then
    skip_phase 1 "Batch Inference" "--skip-batch flag"
elif [[ -f "$RESULTS_CSV" ]]; then
    NROWS=$(wc -l < "$RESULTS_CSV")
    if [[ "$NROWS" -ge 4001 ]]; then   # header + 4000 data rows
        skip_phase 1 "Batch Inference" "CSV already has $((NROWS-1)) rows"
    else
        warn "CSV exists but only has $((NROWS-1)) rows — re-running (checkpoint resume)"
        run_phase 1 "Batch Inference" "scripts/batch_inference.py" && \
            git_commit_phase 1 "Batch Inference" \
                results/vlm_inference_results.csv results/vlm_inference_checkpoint.json \
                vlm_skipped_indices.log batch_inference.log
    fi
else
    run_phase 1 "Batch Inference" "scripts/batch_inference.py" && \
        git_commit_phase 1 "Batch Inference" \
            results/vlm_inference_results.csv results/vlm_inference_checkpoint.json \
            vlm_skipped_indices.log batch_inference.log
fi

# Guard: remaining phases all need the CSV
if [[ ! -f "$RESULTS_CSV" ]]; then
    fail "results/vlm_inference_results.csv missing — cannot continue"
    exit 1
fi

# ── Phase 2: Gaussian Blur Robustness ────────────────────────────────────────
if [[ "$SKIP_BLUR" -eq 1 ]]; then
    skip_phase 2 "Blur Robustness" "--skip-blur flag"
else
    run_phase 2 "Blur Robustness" "scripts/robustness_blur.py" && \
        git_commit_phase 2 "Blur Robustness" \
            results/robustness_blurred_results.csv results/robustness_report.json \
            robustness_blur.log
fi

# ── Phase 2b: Precision Ablation (FP16 vs NF4) ───────────────────────────────
# Optional but strongly recommended: isolates the quantisation confound.
# Requires robustness_blurred_results.csv (Phase 2) to be present.
if [[ "$SKIP_PRECISION" -eq 1 ]]; then
    skip_phase "2b" "Precision Ablation" "--skip-precision flag"
elif [[ ! -f "results/robustness_blurred_results.csv" ]]; then
    skip_phase "2b" "Precision Ablation" "robustness_blurred_results.csv not found (run Phase 2 first)"
else
    run_phase "2b" "Precision Ablation (FP16 vs NF4)" "scripts/precision_ablation.py" && \
        git_commit_phase "2b" "Precision Ablation" \
            results/precision_ablation_results.json results/precision_ablation_rows.csv \
            precision_ablation.log
fi
# ── Phase 2c: Stratified Blur Robustness (all 4 correctness strata) ────────────────
# Addresses "both-correct" bias: measures blur sensitivity on A=both-correct,
# B=Qwen-only, C=Smol-only, D=both-wrong strata to test whether rho=1.50
# is driven by image-difficulty selection or reflects architectural differences.
if [[ "${SKIP_STRAT_ROB:-0}" -eq 1 ]]; then
    skip_phase "2c" "Stratified Robustness" "--skip-strat-rob flag"
else
    run_phase "2c" "Stratified Robustness" "scripts/stratified_robustness.py" && \
        git_commit_phase "2c" "Stratified Robustness" \
            results/stratified_robustness_results.csv \
            results/stratified_robustness_report.json \
            stratified_robustness.log
fi
# ── Phase 3: Calibration / ECE ───────────────────────────────────────────────
if [[ "$SKIP_ECE" -eq 1 ]]; then
    skip_phase 3 "Calibration ECE" "--skip-ece flag"
else
    run_phase 3 "Calibration ECE" "scripts/calibration_ece.py" && \
        git_commit_phase 3 "Calibration ECE" \
            results/calibration_results.json \
            figures/calibration_reliability_diagram_qwen_vqav2.png \
            figures/calibration_reliability_diagram_qwen_coco_captions.png \
            figures/calibration_reliability_diagram_smolvlm2_vqav2.png \
            figures/calibration_reliability_diagram_smolvlm2_coco_captions.png \
            calibration_ece.log
fi

# ── Phase 4: Negation Probes ─────────────────────────────────────────────────
if [[ "$SKIP_NEGATION" -eq 1 ]]; then
    skip_phase 4 "Negation Probes" "--skip-negation flag"
else
    run_phase 4 "Negation Probes" "scripts/negation_probes.py" && \
        git_commit_phase 4 "Negation Probes" \
            results/negation_probes_results.csv results/negation_probes_summary.json \
            negation_probes.log
fi

# ── Phase 5: LLM-as-Judge ────────────────────────────────────────────────────
if [[ "$SKIP_LLM" -eq 1 ]]; then
    skip_phase 5 "LLM-as-Judge" "--skip-llm flag"
elif [[ -z "${OPENAI_API_KEY:-}" ]]; then
    skip_phase 5 "LLM-as-Judge" "OPENAI_API_KEY not set"
else
    run_phase 5 "LLM-as-Judge" "scripts/llm_judge.py" && \
        git_commit_phase 5 "LLM-as-Judge" \
            results/llm_judge_labels.json results/cohen_kappa_report.json llm_judge.log
fi

# ── Phase 6: Update REPORT.md ────────────────────────────────────────────────
if [[ "$SKIP_REPORT" -eq 1 ]]; then
    skip_phase 6 "Report Update" "--skip-report flag"
else
    run_phase 6 "Report Update" "scripts/update_report.py" && \
        git_commit_phase 6 "Report Update" \
            docs/REPORT.md update_report.log
fi

# ── Phase 6b: Taxonomy Diagnosis (optional — requires no API key for Lens 1-2)
# Lens 3 (GPT-4o re-judge) is automatically skipped when OPENAI_API_KEY is unset.
if [[ "${SKIP_TAXONOMY_DIAG:-0}" -eq 1 ]]; then
    skip_phase "6b" "Taxonomy Diagnosis" "--skip-taxonomy-diag flag"
else
    run_phase "6b" "Taxonomy Diagnosis" "scripts/taxonomy_diagnosis.py" && \
        git_commit_phase "6b" "Taxonomy Diagnosis" \
            results/taxonomy_diagnosis_report.json taxonomy_diagnosis.log
fi

# ── Final Summary ─────────────────────────────────────────────────────────────
sep
log "Pipeline complete"
printf "\n"
printf "  %-6s %-24s %-8s %s\n" "Phase" "Name" "Status" "Duration"
printf "  %-6s %-24s %-8s %s\n" "-----" "----" "------" "--------"

declare -A PHASE_NAMES=(
    [1]="Batch Inference"
    [2]="Blur Robustness"
    [2b]="Precision Ablation"
    [2c]="Stratified Robustness"
    [3]="Calibration ECE"
    [4]="Negation Probes"
    [5]="LLM-as-Judge"
    [6]="Report Update"
    [6b]="Taxonomy Diagnosis"
)

for i in 1 2 "2b" "2c" 3 4 5 6 "6b"; do
    STATUS="${PHASE_STATUS[$i]:-SKIP}"
    DURATION="${PHASE_DURATION[$i]:-—}"
    NAME="${PHASE_NAMES[$i]}"
    case "$STATUS" in
        OK)      COLOR="$GRN" ;;
        FAIL|MISSING) COLOR="$RED" ;;
        *)       COLOR="$YLW" ;;
    esac
    printf "  %-6s %-24s ${COLOR}%-8s${RST} %s\n" "$i" "$NAME" "$STATUS" "$DURATION"
done

printf "\n"
log "Passed: $PHASE_PASS | Failed: $PHASE_FAIL | Skipped: $PHASE_SKIP"

# List generated outputs
printf "\n"
log "Generated output files:"
for f in \
    results/vlm_inference_results.csv \
    results/precision_ablation_results.json \
    results/precision_ablation_rows.csv \
    results/robustness_blurred_results.csv \
    results/robustness_report.json \
    results/stratified_robustness_results.csv \
    results/stratified_robustness_report.json \
    results/calibration_results.json \
    figures/calibration_reliability_diagram_qwen_vqav2.png \
    figures/calibration_reliability_diagram_qwen_coco_captions.png \
    figures/calibration_reliability_diagram_smolvlm2_vqav2.png \
    figures/calibration_reliability_diagram_smolvlm2_coco_captions.png \
    results/negation_probes_results.csv \
    results/negation_probes_summary.json \
    results/llm_judge_labels.json \
    results/cohen_kappa_report.json \
    results/taxonomy_diagnosis_report.json \
    docs/REPORT.md; do
    if [[ -f "$f" ]]; then
        SIZE=$(du -sh "$f" 2>/dev/null | cut -f1)
        ok "  $f  ($SIZE)"
    else
        warn "  $f  (not generated)"
    fi
done

sep

[[ "$PHASE_FAIL" -eq 0 ]] && exit 0 || exit 1
