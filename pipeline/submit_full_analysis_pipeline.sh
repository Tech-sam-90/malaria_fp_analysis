#!/bin/bash
# =============================================================================
# Master Orchestration Script — Submit Full Analysis Pipeline
#
# Usage:
#   bash pipeline/submit_full_analysis_pipeline.sh
#   bash pipeline/submit_full_analysis_pipeline.sh <model_b_job_id>
#
# If no job ID is supplied, the script auto-detects the running model_b_train
# job from squeue. If Model B has already finished, Jobs A–C are submitted
# immediately (no dependency needed).
#
# Job dependency chain:
#
#   [Model B training]
#         ↓ afterok
#   [Job A — GPU]  Phase 2 (model.val) + Phase 3 (model.predict / TIDE)
#         ↓ afterok
#   [Job B — CPU]  Phase 2 FP curves + Phase 4 + Phase 5 + Phase 6
#         ↓ afterok
#   [Job C — CPU]  Phase 7 reporting + git push to GitHub
#
# =============================================================================

set -euo pipefail
cd ~/scratch/Practicum_Main_Project

echo "========================================================"
echo "  Malaria FP Analysis — Pipeline Submission"
echo "  $(date)"
echo "========================================================"

# -----------------------------------------------------------------------------
# 1. Resolve Model B job ID
# -----------------------------------------------------------------------------
MODEL_B_JOBID="${1:-}"

if [[ -z "$MODEL_B_JOBID" ]]; then
    # Auto-detect from squeue
    MODEL_B_JOBID=$(squeue -u "$USER" --noheader -o "%i %j" \
                    | awk '$2 == "model_b_train" {print $1; exit}')
fi

if [[ -n "$MODEL_B_JOBID" ]]; then
    echo "Model B job detected: $MODEL_B_JOBID"
    echo "Jobs A–C will run after it completes (afterok)."
    DEPENDENCY_A="--dependency=afterok:${MODEL_B_JOBID}"
else
    echo "Model B job not found in queue — submitting Jobs A–C immediately."
    DEPENDENCY_A=""
fi

# -----------------------------------------------------------------------------
# 2. Submit Job A (GPU: Phase 2 + Phase 3)
# -----------------------------------------------------------------------------
JOB_A_ID=$(sbatch \
    $DEPENDENCY_A \
    pipeline/job_a_gpu_evaluation_and_fp_extraction.sh \
    | awk '{print $NF}')

echo
echo "Job A submitted  → ID: $JOB_A_ID"
echo "  Name   : phase_2_3_eval"
echo "  Depends: ${MODEL_B_JOBID:-none}"
echo "  Tasks  : Phase 2 (baseline + confusion matrices) + Phase 3 (FP extraction + TIDE)"

# -----------------------------------------------------------------------------
# 3. Submit Job B (CPU: Phase 4 + Phase 5 + Phase 6)
# -----------------------------------------------------------------------------
JOB_B_ID=$(sbatch \
    --dependency=afterok:${JOB_A_ID} \
    pipeline/job_b_cpu_spatial_and_statistical_analysis.sh \
    | awk '{print $NF}')

echo
echo "Job B submitted  → ID: $JOB_B_ID"
echo "  Name   : phase_4_5_6_analysis"
echo "  Depends: $JOB_A_ID (Job A)"
echo "  Tasks  : Phase 2 FP curves + Phase 4 (spatial) + Phase 5 (stats) + Phase 6 (calibration)"

# -----------------------------------------------------------------------------
# 4. Submit Job C (CPU: Phase 7 + git push)
# -----------------------------------------------------------------------------
JOB_C_ID=$(sbatch \
    --dependency=afterok:${JOB_B_ID} \
    pipeline/job_c_cpu_reporting_and_github_push.sh \
    | awk '{print $NF}')

echo
echo "Job C submitted  → ID: $JOB_C_ID"
echo "  Name   : phase_7_report_push"
echo "  Depends: $JOB_B_ID (Job B)"
echo "  Tasks  : Phase 7 (figures + tables) + git push to GitHub"

# -----------------------------------------------------------------------------
# 5. Summary
# -----------------------------------------------------------------------------
echo
echo "========================================================"
echo "  Pipeline submitted successfully!"
echo "========================================================"
echo
echo "  Dependency chain:"
if [[ -n "$MODEL_B_JOBID" ]]; then
echo "    Model B ($MODEL_B_JOBID)"
echo "         ↓ afterok"
fi
echo "    Job A  ($JOB_A_ID)  — GPU eval + FP extraction"
echo "         ↓ afterok"
echo "    Job B  ($JOB_B_ID)  — Spatial / statistical / calibration"
echo "         ↓ afterok"
echo "    Job C  ($JOB_C_ID)  — Reporting + GitHub push"
echo
echo "  Monitor progress:"
echo "    squeue -u \$USER"
echo
echo "  Watch live output of Job A:"
echo "    tail -f phase_2_3_eval-${JOB_A_ID}.out"
echo
echo "  Watch live output of Job B:"
echo "    tail -f phase_4_5_6_analysis-${JOB_B_ID}.out"
echo
echo "  Watch live output of Job C:"
echo "    tail -f phase_7_report_push-${JOB_C_ID}.out"
echo "========================================================"
