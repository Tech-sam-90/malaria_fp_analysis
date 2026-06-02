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

#SBATCH --account=rpp-uanazodo
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --job-name=phase_4_5_6_analysis
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "============================================"
echo "Job B: Spatial + Statistical + Calibration"
echo "Started : $(date)"
echo "Node    : $(hostname)"
echo "Job ID  : $SLURM_JOB_ID"
echo "============================================"

module load gcc opencv python/3.11
source ~/scratch/venv/bin/activate

export WANDB_MODE=disabled

cd ~/scratch/Practicum_Main_Project

step() { echo; echo ">>> [$(date '+%H:%M:%S')] $1"; }

# =============================================================================
# PHASE 2 (remainder — reads Phase 3 CSV)
# =============================================================================
step "Phase 2: Per-confidence FP rate curves"
python Notebooks/phase_2_baseline_evaluation/per_confidence_fp_rate_curves.py
echo "    Done."

# =============================================================================
# PHASE 4 — SPATIAL ANALYSIS
# =============================================================================
step "Phase 4: Mann-Whitney distance to nearest WBC"
python Notebooks/phase_4_spatial_analysis/mannwhitney_distance_to_nearest_wbc.py
echo "    Done."

step "Phase 4: Distance stratification by WBC proximity"
python Notebooks/phase_4_spatial_analysis/distance_stratification_by_wbc_proximity.py
echo "    Done."

step "Phase 4: Ripley Cross-K function (FP-WBC spatial attraction)"
python Notebooks/phase_4_spatial_analysis/ripley_cross_k_function_fp_wbc_attraction.py
echo "    Done."

step "Phase 4: KDE correlation — FP density vs WBC density"
python Notebooks/phase_4_spatial_analysis/kde_correlation_fp_density_vs_wbc_density.py
echo "    Done."

step "Phase 4: Moran's I spatial autocorrelation"
python Notebooks/phase_4_spatial_analysis/morans_i_spatial_autocorrelation.py
echo "    Done."

step "Phase 4: Overdispersion test (Poisson vs Negative Binomial)"
python Notebooks/phase_4_spatial_analysis/overdispersion_negative_binomial_test.py
echo "    Done."

step "Phase 4: DBSCAN FP cluster analysis"
python Notebooks/phase_4_spatial_analysis/dbscan_fp_cluster_analysis.py
echo "    Done."

# =============================================================================
# PHASE 5 — MODEL COMPARISON
# =============================================================================
step "Phase 5: Paired bootstrap mAP comparison"
python Notebooks/phase_5_model_comparison/paired_bootstrap_map_comparison.py
echo "    Done."

step "Phase 5: Wilcoxon signed-rank per-image AP"
python Notebooks/phase_5_model_comparison/wilcoxon_signedrank_per_image_ap.py
echo "    Done."

step "Phase 5: McNemar test + stratified WBC proximity"
python Notebooks/phase_5_model_comparison/mcnemar_test_and_stratified_wbc_proximity.py
echo "    Done."

# =============================================================================
# PHASE 6 — CALIBRATION
# =============================================================================
step "Phase 6: ECE/MCE reliability diagrams"
python Notebooks/phase_6_calibration/ece_mce_reliability_diagrams.py
echo "    Done."

step "Phase 6: Temperature scaling calibration"
python Notebooks/phase_6_calibration/temperature_scaling_calibration.py
echo "    Done."

step "Phase 6: Multivariate position-aware calibration"
python Notebooks/phase_6_calibration/multivariate_position_aware_calibration.py
echo "    Done."

echo
echo "============================================"
echo "Job B COMPLETE — $(date)"
echo "============================================"
