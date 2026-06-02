# =============================================================================
# Phase 3 | TIDE Error Decomposition
#
# Uses the TIDE (Task-Independent Detection Evaluation) framework to decompose
# ALL detection errors for both models into 6 mutually exclusive categories:
#
#   Bkg  — Background FP: detection overlaps no GT box at any IoU
#   Cls  — Classification error: correct location, wrong class label
#   Loc  — Localization error: correct class, IoU just below threshold (0.45–0.50)
#   Both — Wrong class AND poor localization
#   Dup  — Duplicate: GT already matched by a higher-confidence detection
#   Miss — Missed GT: no detection matched this GT object at any confidence
#
# Key insight for this project:
#   Single-class model (A): all FPs are Bkg — no classification confusion possible
#   Two-class model (B): Cls errors = predictions localizing a WBC correctly but
#                        labeling it as parasite → direct WBC-confusion count
#
# Install: pip install tidecv
#
# Outputs (saved to outputs/analysis_results/phase_3/):
#   tide_errors_model_a.csv
#   tide_errors_model_b.csv
#   tide_error_decomposition_comparison.png
# =============================================================================

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import tidecv
    from tidecv import TIDE, datasets
except ImportError:
    sys.exit("ERROR: tidecv not installed. Run: pip install tidecv")

# =============================================================================
# PATHS
# =============================================================================
project_root    = Path("/home/sadeniji/scratch/Practicum_Main_Project")
model_a_weights = project_root / "outputs/model_a/model_a_parasite_detection/weights/best.pt"
model_b_weights = project_root / "outputs/model_b/model_b_parasite_wbc_detection/weights/best.pt"
data_yaml_a     = project_root / "model_a/data_model_a.yaml"
data_yaml_b     = project_root / "model_b/data_model_b.yaml"
out_dir         = project_root / "outputs/analysis_results/phase_3"
out_dir.mkdir(parents=True, exist_ok=True)

# TIDE requires COCO-format JSON annotations and predictions
# We generate these by running model.val(save_json=True) which produces
# predictions in COCO format, then build GT JSON from label files.
val_dir_a = out_dir / "tide_val_a"
val_dir_b = out_dir / "tide_val_b"
val_dir_a.mkdir(exist_ok=True)
val_dir_b.mkdir(exist_ok=True)

IMG_W, IMG_H = 1024, 1024

# =============================================================================
# HELPERS — build COCO-format JSON from YOLO labels
# =============================================================================
def yolo_labels_to_coco_json(labels_dir, images_dir, class_names, out_path):
    images_list, annotations_list = [], []
    ann_id = 1

    for img_idx, img_path in enumerate(sorted(Path(images_dir).glob("*.jpg")) +
                                        sorted(Path(images_dir).glob("*.png")), 1):
        images_list.append({'id': img_idx, 'file_name': img_path.name,
                            'width': IMG_W, 'height': IMG_H})

        lf = Path(labels_dir) / (img_path.stem + ".txt")
        if not lf.exists():
            continue
        for line in lf.read_text().strip().split('\n'):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc = float(parts[1])*IMG_W, float(parts[2])*IMG_H
            bw, bh = float(parts[3])*IMG_W, float(parts[4])*IMG_H
            x_min, y_min = xc - bw/2, yc - bh/2
            annotations_list.append({
                'id': ann_id, 'image_id': img_idx,
                'category_id': cls_id + 1,  # COCO uses 1-indexed categories
                'bbox': [round(x_min, 2), round(y_min, 2), round(bw, 2), round(bh, 2)],
                'area': round(bw * bh, 2),
                'iscrowd': 0,
            })
            ann_id += 1

    categories = [{'id': i+1, 'name': n} for i, n in enumerate(class_names)]
    coco_dict = {'images': images_list, 'annotations': annotations_list,
                 'categories': categories}
    with open(out_path, 'w') as f:
        json.dump(coco_dict, f)
    return out_path


def run_val_save_json(model_path, data_yaml, out_subdir):
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), split='test', save_json=True,
                        project=str(out_subdir), name='val', verbose=False)
    # Ultralytics writes predictions_<split>.json in the run directory
    pred_json = list(out_subdir.glob("**/predictions.json"))
    if not pred_json:
        pred_json = list(out_subdir.glob("**/*.json"))
    return pred_json[0] if pred_json else None


# =============================================================================
# BUILD GT JSONs
# =============================================================================
print("Building COCO GT JSON for Model A...")
gt_json_a = yolo_labels_to_coco_json(
    labels_dir=project_root / "model_a/test/labels",
    images_dir=project_root / "model_a/test/images",
    class_names=['parasite'],
    out_path=val_dir_a / "gt_coco.json"
)
print(f"  Saved to {gt_json_a}")

print("Building COCO GT JSON for Model B...")
gt_json_b = yolo_labels_to_coco_json(
    labels_dir=project_root / "model_b/test/labels",
    images_dir=project_root / "model_b/test/images",
    class_names=['parasite', 'WBC'],
    out_path=val_dir_b / "gt_coco.json"
)
print(f"  Saved to {gt_json_b}")

# =============================================================================
# GET PREDICTION JSONs
# =============================================================================
print("\nRunning val to get COCO prediction JSONs...")
pred_json_a = run_val_save_json(model_a_weights, data_yaml_a, val_dir_a)
pred_json_b = run_val_save_json(model_b_weights, data_yaml_b, val_dir_b)

if not pred_json_a or not pred_json_b:
    sys.exit("ERROR: Could not find prediction JSON files. Check val output directories.")

print(f"  Model A predictions: {pred_json_a}")
print(f"  Model B predictions: {pred_json_b}")

# =============================================================================
# RUN TIDE
# =============================================================================
def run_tide(gt_json, pred_json, model_label):
    tide = TIDE()
    tide.evaluate(datasets.COCO(str(gt_json)), datasets.COCOResult(str(pred_json)),
                  mode=TIDE.BOX)
    tide.summarize()

    errors = tide.errors
    row = {'model': model_label}
    error_names = ['Cls', 'Loc', 'Both', 'Dup', 'Bkg', 'Miss']
    for name in error_names:
        attr = name.lower() + '_error'
        row[name] = float(getattr(errors, attr, 0) or 0)
    return row


print("\n" + "="*55)
print("Running TIDE for Model A...")
row_a = run_tide(gt_json_a, pred_json_a, 'Model_A')

print("\n" + "="*55)
print("Running TIDE for Model B...")
row_b = run_tide(gt_json_b, pred_json_b, 'Model_B')

# =============================================================================
# SAVE AND PLOT
# =============================================================================
df_tide = pd.DataFrame([row_a, row_b])
df_tide.to_csv(out_dir / "tide_error_decomposition.csv", index=False)
print(f"\nTIDE results:\n{df_tide.to_string(index=False)}")

error_cols = ['Bkg', 'Cls', 'Loc', 'Dup', 'Both', 'Miss']
colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#9b59b6', '#3498db']

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(error_cols))
width = 0.35

vals_a = [row_a.get(c, 0) for c in error_cols]
vals_b = [row_b.get(c, 0) for c in error_cols]

bars_a = ax.bar(x - width/2, vals_a, width, label='Model A', color='steelblue', alpha=0.85)
bars_b = ax.bar(x + width/2, vals_b, width, label='Model B', color='darkorange', alpha=0.85)

ax.set_title('TIDE Error Decomposition: Model A vs Model B', fontsize=13)
ax.set_ylabel('AP Error Contribution')
ax.set_xticks(x)
ax.set_xticklabels(['Background\n(Bkg)', 'Class\nConfusion\n(Cls)',
                     'Localization\n(Loc)', 'Duplicate\n(Dup)',
                     'Both\n(Cls+Loc)', 'Missed\n(Miss)'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bar in list(bars_a) + list(bars_b):
    h = bar.get_height()
    if h > 0:
        ax.annotate(f'{h:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(out_dir / "tide_error_decomposition_comparison.png", dpi=150, bbox_inches='tight')
print(f"TIDE chart saved to: {out_dir}/tide_error_decomposition_comparison.png")

print("\n--- Key Finding ---")
print(f"  Model A Bkg FP error : {row_a.get('Bkg', 0):.4f}  "
      f"(all FPs are background — no class confusion possible)")
print(f"  Model B Bkg FP error : {row_b.get('Bkg', 0):.4f}")
print(f"  Model B Cls error    : {row_b.get('Cls', 0):.4f}  "
      f"(WBC-as-parasite confusion)")

print("\nPhase 3 — TIDE decomposition complete.")
