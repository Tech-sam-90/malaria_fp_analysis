# =============================================================================
# Confusion Matrices at Multiple IoU Thresholds — Model A and Model B
# Generates normalised confusion matrices at IoU = {0.10, 0.30, 0.50, 0.70, 0.90}
# Output: outputs/analysis_results/phase_2/confusion_matrix_multi_iou_MODEL.png
# =============================================================================
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO

project_root    = Path("/home/sadeniji/scratch/Practicum_Main_Project")
model_a_weights = project_root / "outputs/model_a/model_a_parasite_detection-7/weights/best.pt"
model_b_weights = project_root / "outputs/model_b/model_b_parasite_wbc_detection/weights/best.pt"
data_yaml_a     = project_root / "model_a/data_model_a.yaml"
data_yaml_b     = project_root / "model_b/data_model_b.yaml"
out_dir         = project_root / "outputs/analysis_results/phase_2"
out_dir.mkdir(parents=True, exist_ok=True)

IOU_THRESHOLDS = [0.10, 0.30, 0.50, 0.70, 0.90]

def run_multi_iou(model_path, data_yaml, class_names, out_name):
    model = YOLO(str(model_path))
    results_per_iou = {}

    for iou in IOU_THRESHOLDS:
        print(f"  IoU={iou:.2f} ...", end=' ', flush=True)
        metrics = model.val(data=str(data_yaml), split='test', iou=iou,
                            verbose=False, plots=False)
        cm = metrics.confusion_matrix.matrix
        results_per_iou[iou] = cm
        print("done")

    labels = class_names + ['Background']
    fig, axes = plt.subplots(1, len(IOU_THRESHOLDS), figsize=(5 * len(IOU_THRESHOLDS), 5))
    fig.suptitle(f'Normalised Confusion Matrices — {out_name}',
                 fontsize=12, fontweight='bold')

    for ax, iou in zip(axes, IOU_THRESHOLDS):
        cm = results_per_iou[iou]
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / np.maximum(row_sums, 1), 0)

        ax.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues')
        ax.set_title(f'IoU = {iou:.2f}', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('True', fontsize=9)

        for i in range(len(labels)):
            for j in range(len(labels)):
                val = cm_norm[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    safe_name = out_name.lower().replace(' ', '_').replace('+', 'plus')
    out_path = out_dir / f"confusion_matrix_multi_iou_{safe_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_path}")

print("Model A — multi-IoU confusion matrices...")
run_multi_iou(model_a_weights, data_yaml_a, ['parasite'], 'Model A (Parasite Only)')

print("\nModel B — multi-IoU confusion matrices...")
run_multi_iou(model_b_weights, data_yaml_b, ['parasite', 'WBC'], 'Model B (Parasite + WBC)')

print("\nDone.")
