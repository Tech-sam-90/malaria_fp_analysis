# =============================================================================
# Phase 2 | Confusion Matrix at Multiple IoU Thresholds
#
# Generates confusion matrices for both models at IoU thresholds:
#   0.10, 0.30, 0.50 (standard PASCAL), 0.70, 0.90
#
# For Model B (two-class), the WBC→Parasite off-diagonal cell directly
# quantifies how many WBC detections are mislabeled as parasites — this is
# the primary WBC-confusion metric for Objective 2.
#
# At loose IoU (0.10), all correctly-localized detections pass → reveals
# pure class-confusion errors. At strict IoU (0.70+), failures are mainly
# localization errors. The delta between loose and strict IoU matrices shows
# how much of Model B's error is localization vs. classification.
#
# Outputs (saved to outputs/analysis_results/phase_2/):
#   confusion_matrix_model_a_iou{X}.png  (for each IoU threshold)
#   confusion_matrix_model_b_iou{X}.png
#   confusion_matrix_summary.csv          (off-diagonal rates per IoU)
# =============================================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix

# =============================================================================
# PATHS
# =============================================================================
project_root    = Path("/home/sadeniji/scratch/Practicum_Main_Project")
model_a_weights = project_root / "outputs/model_a/model_a_parasite_detection/weights/best.pt"
model_b_weights = project_root / "outputs/model_b/model_b_parasite_wbc_detection/weights/best.pt"
data_yaml_a     = project_root / "model_a/data_model_a.yaml"
data_yaml_b     = project_root / "model_b/data_model_b.yaml"
out_dir         = project_root / "outputs/analysis_results/phase_2"
out_dir.mkdir(parents=True, exist_ok=True)

for name, p in [("Model A", model_a_weights), ("Model B", model_b_weights)]:
    if not p.exists():
        sys.exit(f"ERROR: {name} weights not found at {p}")

IOUs = [0.10, 0.30, 0.50, 0.70, 0.90]

# =============================================================================
# HELPER — run val and extract confusion matrix
# =============================================================================
def run_val_at_iou(model_path, data_yaml, iou_thresh, conf_thresh=0.25):
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split='test',
        iou=iou_thresh,
        conf=conf_thresh,
        verbose=False,
    )
    return metrics


def plot_confusion_matrix(cm_array, class_names, title, save_path):
    cm = cm_array.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=12)

    for ax, data, fmt, subtitle in [
        (axes[0], cm,      'd',   'Raw counts'),
        (axes[1], cm_norm, '.2f', 'Normalized (row)'),
    ]:
        im = ax.imshow(data, interpolation='nearest', cmap='Blues')
        ax.set_title(subtitle)
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticks(tick_marks); ax.set_yticklabels(class_names)
        ax.set_ylabel('True label'); ax.set_xlabel('Predicted label')
        thresh = data.max() / 2
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, format(data[i, j], fmt),
                        ha='center', va='center',
                        color='white' if data[i, j] > thresh else 'black', fontsize=9)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# RUN FOR MODEL A
# =============================================================================
summary_rows = []
print("Running Model A confusion matrices at multiple IoU thresholds...")
for iou in IOUs:
    print(f"  IoU = {iou:.2f}", end=' ... ')
    try:
        metrics = run_val_at_iou(model_a_weights, data_yaml_a, iou)
        if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
            cm = metrics.confusion_matrix.matrix
            class_names_a = ['parasite', 'background']
            title = f"Model A — Confusion Matrix (IoU={iou:.2f}, conf=0.25)"
            save_path = out_dir / f"confusion_matrix_model_a_iou{int(iou*100):02d}.png"
            plot_confusion_matrix(cm, class_names_a, title, save_path)
            print(f"saved to {save_path.name}")
            summary_rows.append({'model': 'Model_A', 'iou': iou,
                                  'TP': int(cm[0, 0]), 'FP_bkg': int(cm[1, 0]),
                                  'FN': int(cm[0, 1])})
        else:
            print("confusion matrix not available")
    except Exception as e:
        print(f"ERROR: {e}")

# =============================================================================
# RUN FOR MODEL B
# =============================================================================
print("\nRunning Model B confusion matrices at multiple IoU thresholds...")
for iou in IOUs:
    print(f"  IoU = {iou:.2f}", end=' ... ')
    try:
        metrics = run_val_at_iou(model_b_weights, data_yaml_b, iou)
        if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
            cm = metrics.confusion_matrix.matrix
            class_names_b = ['parasite', 'WBC', 'background']
            title = f"Model B — Confusion Matrix (IoU={iou:.2f}, conf=0.25)"
            save_path = out_dir / f"confusion_matrix_model_b_iou{int(iou*100):02d}.png"
            plot_confusion_matrix(cm, class_names_b, title, save_path)
            print(f"saved to {save_path.name}")
            # WBC→Parasite confusion: cm[0,1] = predicted parasite, true WBC
            wbc_as_parasite = int(cm[0, 1]) if cm.shape[0] > 1 else 0
            summary_rows.append({'model': 'Model_B', 'iou': iou,
                                  'TP_parasite': int(cm[0, 0]),
                                  'WBC_as_parasite': wbc_as_parasite,
                                  'FP_bkg': int(cm[-1, 0]) if cm.shape[0] > 2 else 0})
        else:
            print("confusion matrix not available")
    except Exception as e:
        print(f"ERROR: {e}")

# =============================================================================
# SAVE SUMMARY
# =============================================================================
if summary_rows:
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "confusion_matrix_summary.csv", index=False)
    print(f"\nConfusion matrix summary:\n{df_summary.to_string(index=False)}")

print("\nPhase 2 — Confusion matrix analysis complete.")
