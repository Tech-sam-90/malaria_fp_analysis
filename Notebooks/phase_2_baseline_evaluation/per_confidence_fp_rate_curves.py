# =============================================================================
# Phase 2 | Per-Confidence FP Rate Curves
#
# Sweeps confidence thresholds from 0.05 to 0.95 and plots how Precision,
# Recall, F1, and False Positive rate change for both models. Identifies the
# clinically relevant operating threshold where FP rate becomes acceptable.
#
# Unlike standard PR curves (which vary recall by threshold), this analysis
# shows the absolute FP COUNT per image vs. threshold — a metric that maps
# directly to clinical over-diagnosis risk.
#
# Paper reference:
#   "Confidence Score: The Forgotten Dimension of Object Detection Performance
#    Evaluation" (MDPI Sensors 2021, doi:10.3390/s21134350)
#
# Prerequisite: Phase 3 extract_predictions_and_label_tp_fp.py must have run
#   (reads predictions_tp_fp_labeled_model_a.csv and _model_b.csv)
#   Alternatively, runs inference here at each threshold if CSVs are absent.
#
# Outputs (saved to outputs/analysis_results/phase_2/):
#   confidence_threshold_curves_model_a.csv
#   confidence_threshold_curves_model_b.csv
#   fp_rate_vs_confidence_threshold.png
#   precision_recall_vs_confidence_threshold.png
# =============================================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================
project_root = Path("/home/sadeniji/scratch/Practicum_Main_Project")
results_dir  = project_root / "outputs/analysis_results"
phase2_dir   = results_dir / "phase_2"
phase3_dir   = results_dir / "phase_3"
phase2_dir.mkdir(parents=True, exist_ok=True)

pred_csv_a = phase3_dir / "predictions_tp_fp_labeled_model_a.csv"
pred_csv_b = phase3_dir / "predictions_tp_fp_labeled_model_b.csv"

for name, p in [("Model A predictions CSV", pred_csv_a), ("Model B predictions CSV", pred_csv_b)]:
    if not p.exists():
        sys.exit(f"ERROR: {name} not found at {p}.\n"
                 f"Run phase_3/extract_predictions_and_label_tp_fp.py first.")

# =============================================================================
# SWEEP THRESHOLDS
# =============================================================================
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)

def threshold_sweep(df_preds, model_label, n_gt_total, n_images):
    records = []
    for thresh in THRESHOLDS:
        above = df_preds[df_preds['conf'] >= thresh]
        tp = above['is_tp'].sum()
        fp = above['is_fp'].sum()
        total = len(above)
        precision   = tp / max(total, 1)
        recall      = tp / max(n_gt_total, 1)
        f1          = 2 * precision * recall / max(precision + recall, 1e-9)
        fp_per_img  = fp / max(n_images, 1)
        records.append({
            'model': model_label,
            'threshold': thresh,
            'n_predictions': int(total),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'fp_per_image': round(fp_per_img, 4),
        })
    return pd.DataFrame(records)


print("Loading labeled predictions...")
df_a = pd.read_csv(pred_csv_a)
df_b = pd.read_csv(pred_csv_b)

n_images_a = df_a['image_id'].nunique()
n_images_b = df_b['image_id'].nunique()

# Ground-truth totals from the TP count at the lowest threshold
n_gt_a = df_a[df_a['conf'] >= 0.05]['is_tp'].sum()  # approximate lower bound
n_gt_b = df_b[df_b['conf'] >= 0.05]['is_tp'].sum()

curves_a = threshold_sweep(df_a, 'Model_A', n_gt_a, n_images_a)
curves_b = threshold_sweep(df_b, 'Model_B', n_gt_b, n_images_b)

curves_a.to_csv(phase2_dir / "confidence_threshold_curves_model_a.csv", index=False)
curves_b.to_csv(phase2_dir / "confidence_threshold_curves_model_b.csv", index=False)
print(f"Threshold curve CSVs saved to {phase2_dir}")

# =============================================================================
# PLOT 1 — FP COUNT PER IMAGE vs CONFIDENCE THRESHOLD
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(curves_a['threshold'], curves_a['fp_per_image'], 'b-o', markersize=5,
        label='Model A (parasite only)', linewidth=2)
ax.plot(curves_b['threshold'], curves_b['fp_per_image'], 'r-s', markersize=5,
        label='Model B (parasite + WBC)', linewidth=2)
ax.set_xlabel('Confidence Threshold', fontsize=12)
ax.set_ylabel('False Positives per Image', fontsize=12)
ax.set_title('FP Rate vs Confidence Threshold', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0.05, 0.95])
plt.tight_layout()
plt.savefig(phase2_dir / "fp_rate_vs_confidence_threshold.png", dpi=150, bbox_inches='tight')

# =============================================================================
# PLOT 2 — PRECISION, RECALL, F1 vs THRESHOLD (4-panel)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Precision / Recall / F1 vs Confidence Threshold', fontsize=13)

for curves, label, color in [(curves_a, 'Model A', 'steelblue'), (curves_b, 'Model B', 'darkorange')]:
    axes[0, 0].plot(curves['threshold'], curves['precision'], color=color, label=label, linewidth=2)
    axes[0, 1].plot(curves['threshold'], curves['recall'],    color=color, label=label, linewidth=2)
    axes[1, 0].plot(curves['threshold'], curves['f1'],        color=color, label=label, linewidth=2)
    axes[1, 1].plot(curves['threshold'], curves['n_predictions'], color=color, label=label, linewidth=2)

for ax, title, ylabel in [
    (axes[0, 0], 'Precision vs Threshold',     'Precision'),
    (axes[0, 1], 'Recall vs Threshold',         'Recall'),
    (axes[1, 0], 'F1 Score vs Threshold',       'F1'),
    (axes[1, 1], 'Total Predictions vs Threshold', 'N Predictions'),
]:
    ax.set_title(title); ax.set_xlabel('Threshold'); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim([0.05, 0.95])

plt.tight_layout()
plt.savefig(phase2_dir / "precision_recall_vs_confidence_threshold.png", dpi=150, bbox_inches='tight')

# =============================================================================
# PRINT SUMMARY TABLE
# =============================================================================
print("\n--- Model A: Key Thresholds ---")
for row in curves_a[curves_a['threshold'].isin([0.25, 0.50, 0.75])].itertuples():
    print(f"  conf={row.threshold:.2f}  P={row.precision:.3f}  R={row.recall:.3f}  "
          f"F1={row.f1:.3f}  FP/img={row.fp_per_image:.2f}")

print("\n--- Model B: Key Thresholds ---")
for row in curves_b[curves_b['threshold'].isin([0.25, 0.50, 0.75])].itertuples():
    print(f"  conf={row.threshold:.2f}  P={row.precision:.3f}  R={row.recall:.3f}  "
          f"F1={row.f1:.3f}  FP/img={row.fp_per_image:.2f}")

print("\nPhase 2 — Confidence threshold curves complete.")
