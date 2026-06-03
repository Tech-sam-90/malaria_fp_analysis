#!/bin/bash
# =============================================================================
# Job A — GPU: Baseline Evaluation + FP Extraction
#
# Depends on: Model B training job (passed via --dependency=afterok:<jobid>)
#
# Runs (in order):
#   Phase 2: baseline_metrics_and_size_stratified_ap.py   (model.val — GPU)
#   Phase 2: confusion_matrix_multi_iou_thresholds.py     (model.val — GPU)
#   Phase 3: extract_predictions_and_label_tp_fp.py       (model.predict — GPU)
#   Phase 3: tide_error_decomposition.py                  (model.val — GPU)
#   Phase 3: hard_negative_audit_and_visualization.py     (inference — GPU)
#
# Outputs needed by Job B:
#   outputs/analysis_results/phase_2/*.csv
#   outputs/analysis_results/phase_3/predictions_tp_fp_labeled_model_a.csv
#   outputs/analysis_results/phase_3/predictions_tp_fp_labeled_model_b.csv
#   outputs/analysis_results/phase_3/tide_error_decomposition.csv
# =============================================================================

#SBATCH --account=rpp-uanazodo_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=phase_2_3_eval
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail   # -e removed so individual step failures don't abort the job

echo "============================================"
echo "Job A: GPU Evaluation + FP Extraction"
echo "Started : $(date)"
echo "Node    : $(hostname)"
echo "Job ID  : $SLURM_JOB_ID"
echo "============================================"

# --- Environment ---
module load gcc opencv python/3.11 cuda/12.2
source ~/scratch/venv/bin/activate

# Ensure optional analysis packages are installed
pip install -q tidecv mlxtend esda libpysal 2>/dev/null || true

export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0

cd ~/scratch/Practicum_Main_Project

# Helper: log step with timestamp
step() { echo; echo ">>> [$(date '+%H:%M:%S')] $1"; }

# Helper: run step, warn on failure but don't abort the whole job
run_step() {
    local label="$1"; shift
    step "$label"
    if python "$@"; then
        echo "    Done."
    else
        echo "    WARNING: $label failed — continuing pipeline."
    fi
}

# =============================================================================
# PHASE 2
# =============================================================================
# Skip if outputs already exist from a previous run
if [ -f outputs/analysis_results/phase_2/baseline_metrics_comparison.csv ]; then
    step "Phase 2: Baseline metrics — already exists, skipping."
else
    run_step "Phase 2: Baseline metrics and size-stratified AP" \
        Notebooks/phase_2_baseline_evaluation/baseline_metrics_and_size_stratified_ap.py
fi

if ls outputs/analysis_results/phase_2/confusion_matrix_model_a_iou*.png 2>/dev/null | grep -q .; then
    step "Phase 2: Confusion matrices — already exist, skipping."
else
    run_step "Phase 2: Confusion matrices at multiple IoU thresholds" \
        Notebooks/phase_2_baseline_evaluation/confusion_matrix_multi_iou_thresholds.py
fi

# =============================================================================
# PHASE 3
# =============================================================================
if [ -f outputs/analysis_results/phase_3/predictions_tp_fp_labeled_model_a.csv ] && \
   [ -f outputs/analysis_results/phase_3/predictions_tp_fp_labeled_model_b.csv ]; then
    step "Phase 3: TP/FP labels — already exist, skipping."
else
    run_step "Phase 3: Extract predictions and label TP/FP  [CENTRAL ARTIFACT]" \
        Notebooks/phase_3_fp_extraction/extract_predictions_and_label_tp_fp.py
fi

run_step "Phase 3: TIDE error decomposition" \
    Notebooks/phase_3_fp_extraction/tide_error_decomposition.py

run_step "Phase 3: Hard negative audit and visualization" \
    Notebooks/phase_3_fp_extraction/hard_negative_audit_and_visualization.py

echo
echo "============================================"
echo "Job A COMPLETE — $(date)"
echo "============================================"
