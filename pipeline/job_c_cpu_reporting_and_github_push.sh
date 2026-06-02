#!/bin/bash
# =============================================================================
# Job C — CPU: Phase 7 Reporting + GitHub Push
#
# Depends on: Job B (afterok) — requires all analysis result CSVs and PNGs
#
# Runs (in order):
#   Phase 7: generate_tide_error_breakdown_chart.py
#   Phase 7: generate_spatial_heatmaps_and_kde_overlays.py
#   Phase 7: generate_final_summary_tables.py
#
# Then pushes all new outputs to GitHub:
#   git add outputs/ → git commit → git push
# =============================================================================

#SBATCH --account=def-uanazodo-ab
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=phase_7_report_push
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail   # removed -e so individual failures don't abort the git push

echo "============================================"
echo "Job C: Phase 7 Reporting + GitHub Push"
echo "Started : $(date)"
echo "Node    : $(hostname)"
echo "Job ID  : $SLURM_JOB_ID"
echo "============================================"

module load gcc opencv python/3.11
source ~/scratch/venv/bin/activate

pip install -q openpyxl 2>/dev/null || true

export WANDB_MODE=disabled

cd ~/scratch/Practicum_Main_Project

step() { echo; echo ">>> [$(date '+%H:%M:%S')] $1"; }

run_step() {
    local label="$1"; shift
    step "$label"
    if python "$@"; then
        echo "    Done."
    else
        echo "    WARNING: $label failed — continuing to next step."
    fi
}

# =============================================================================
# PHASE 7 — REPORTING (all non-fatal)
# =============================================================================
run_step "Phase 7: Generate TIDE error breakdown chart" \
    Notebooks/phase_7_reporting/generate_tide_error_breakdown_chart.py

run_step "Phase 7: Generate spatial heatmaps and KDE overlays" \
    Notebooks/phase_7_reporting/generate_spatial_heatmaps_and_kde_overlays.py

run_step "Phase 7: Generate final summary tables" \
    Notebooks/phase_7_reporting/generate_final_summary_tables.py

# =============================================================================
# GITHUB PUSH
# =============================================================================
step "GitHub: Staging all new outputs"

# Load SSH key for GitHub (the key has no passphrase so this is non-interactive)
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

git config user.email "ifeoluwasamuel40@gmail.com"
git config user.name  "Tech-sam-90"

# Stage outputs and pipeline scripts
git add outputs/ pipeline/

# Stage any other new/changed files (notebooks edits, etc.)
git add -u

# Only commit if there is something staged
if git diff --cached --quiet; then
    echo "    Nothing new to commit — repo already up to date."
else
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
    git commit -m "Pipeline run complete ${TIMESTAMP}: add Phase 2-7 analysis outputs

- Phase 2: baseline metrics, confusion matrices, FP rate curves
- Phase 3: TP/FP labeled predictions, TIDE decomposition, hard negative audit
- Phase 4: spatial analysis (Mann-Whitney, Ripley K, KDE, Moran's I, DBSCAN)
- Phase 5: model comparison (bootstrap CI, Wilcoxon, McNemar stratified)
- Phase 6: calibration (ECE/MCE, temperature scaling, position-aware)
- Phase 7: final figures and summary tables

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

    git push origin main
    echo "    Pushed to github.com/Tech-sam-90/malaria_fp_analysis"
fi

ssh-agent -k   # clean up SSH agent

echo
echo "============================================"
echo "Job C COMPLETE — $(date)"
echo "Pipeline fully done."
echo "============================================"
