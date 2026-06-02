#!/bin/bash
# =============================================================================
# Job B — CPU: Spatial Analysis + Statistical Comparison + Calibration
#
# Depends on: Job A (afterok) — requires predictions_tp_fp_labeled_*.csv
#
# Runs (in order):
#   Phase 2: per_confidence_fp_rate_curves.py       (reads Phase 3 CSV)
#
#   Phase 4: mannwhitney_distance_to_nearest_wbc.py
#   Phase 4: distance_stratification_by_wbc_proximity.py
#   Phase 4: ripley_cross_k_function_fp_wbc_attraction.py
#   Phase 4: kde_correlation_fp_density_vs_wbc_density.py
#   Phase 4: morans_i_spatial_autocorrelation.py
#   Phase 4: overdispersion_negative_binomial_test.py
#   Phase 4: dbscan_fp_cluster_analysis.py
#
#   Phase 5: paired_bootstrap_map_comparison.py
#   Phase 5: wilcoxon_signedrank_per_image_ap.py
#   Phase 5: mcnemar_test_and_stratified_wbc_proximity.py
#
#   Phase 6: ece_mce_reliability_diagrams.py
#   Phase 6: temperature_scaling_calibration.py
#   Phase 6: multivariate_position_aware_calibration.py
#
# Outputs needed by Job C:
#   outputs/analysis_results/phase_2–6/**/*.csv, *.png
# =============================================================================

#SBATCH --account=def-uanazodo-ab
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --job-name=phase_4_5_6_analysis
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail   # removed -e so individual step failures don't abort the job

echo "============================================"
echo "Job B: Spatial + Statistical + Calibration"
echo "Started : $(date)"
echo "Node    : $(hostname)"
echo "Job ID  : $SLURM_JOB_ID"
echo "============================================"

module load gcc opencv python/3.11
source ~/scratch/venv/bin/activate

# Install optional packages — failures are non-fatal
pip install -q esda libpysal mlxtend stambo 2>/dev/null || true

export WANDB_MODE=disabled

cd ~/scratch/Practicum_Main_Project

step() { echo; echo ">>> [$(date '+%H:%M:%S')] $1"; }

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
# PHASE 2 (remainder — reads Phase 3 CSV)
# =============================================================================
if [ -f outputs/analysis_results/phase_2/confidence_threshold_curves_model_a.csv ]; then
    step "Phase 2: FP rate curves — already exist, skipping."
else
    run_step "Phase 2: Per-confidence FP rate curves" \
        Notebooks/phase_2_baseline_evaluation/per_confidence_fp_rate_curves.py
fi

# =============================================================================
# PHASE 4 — SPATIAL ANALYSIS
# =============================================================================
if [ -f outputs/analysis_results/phase_4/mannwhitney_results.csv ]; then
    step "Phase 4: Mann-Whitney — already exists, skipping."
else
    run_step "Phase 4: Mann-Whitney distance to nearest WBC" \
        Notebooks/phase_4_spatial_analysis/mannwhitney_distance_to_nearest_wbc.py
fi

if [ -f outputs/analysis_results/phase_4/distance_stratification_results.csv ]; then
    step "Phase 4: Distance stratification — already exists, skipping."
else
    run_step "Phase 4: Distance stratification by WBC proximity" \
        Notebooks/phase_4_spatial_analysis/distance_stratification_by_wbc_proximity.py
fi

if [ -f outputs/analysis_results/phase_4/ripley_cross_k_pooled_results.csv ]; then
    step "Phase 4: Ripley Cross-K — already exists, skipping."
else
    run_step "Phase 4: Ripley Cross-K function (FP-WBC spatial attraction)" \
        Notebooks/phase_4_spatial_analysis/ripley_cross_k_function_fp_wbc_attraction.py
fi

if [ -f outputs/analysis_results/phase_4/kde_correlation_summary.csv ]; then
    step "Phase 4: KDE correlation — already exists, skipping."
else
    run_step "Phase 4: KDE correlation — FP density vs WBC density" \
        Notebooks/phase_4_spatial_analysis/kde_correlation_fp_density_vs_wbc_density.py
fi

run_step "Phase 4: Moran's I spatial autocorrelation" \
    Notebooks/phase_4_spatial_analysis/morans_i_spatial_autocorrelation.py

run_step "Phase 4: Overdispersion test (Poisson vs Negative Binomial)" \
    Notebooks/phase_4_spatial_analysis/overdispersion_negative_binomial_test.py

run_step "Phase 4: DBSCAN FP cluster analysis" \
    Notebooks/phase_4_spatial_analysis/dbscan_fp_cluster_analysis.py

# =============================================================================
# PHASE 5 — MODEL COMPARISON
# =============================================================================
run_step "Phase 5: Paired bootstrap mAP comparison" \
    Notebooks/phase_5_model_comparison/paired_bootstrap_map_comparison.py

run_step "Phase 5: Wilcoxon signed-rank per-image AP" \
    Notebooks/phase_5_model_comparison/wilcoxon_signedrank_per_image_ap.py

run_step "Phase 5: McNemar test + stratified WBC proximity" \
    Notebooks/phase_5_model_comparison/mcnemar_test_and_stratified_wbc_proximity.py

# =============================================================================
# PHASE 6 — CALIBRATION
# =============================================================================
run_step "Phase 6: ECE/MCE reliability diagrams" \
    Notebooks/phase_6_calibration/ece_mce_reliability_diagrams.py

run_step "Phase 6: Temperature scaling calibration" \
    Notebooks/phase_6_calibration/temperature_scaling_calibration.py

run_step "Phase 6: Multivariate position-aware calibration" \
    Notebooks/phase_6_calibration/multivariate_position_aware_calibration.py

echo
echo "============================================"
echo "Job B COMPLETE — $(date)"
echo "============================================"
