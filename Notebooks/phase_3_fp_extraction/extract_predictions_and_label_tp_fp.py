# =============================================================================
# Phase 3 | Extract All Predictions and Label Each as TP or FP
#
# Runs model.predict() on the test set for both models at a low confidence
# threshold (0.10) to capture all detections. Each prediction is matched
# against ground-truth annotations using IoU >= 0.50 (PASCAL VOC standard)
# and labeled as TP (True Positive) or FP (False Positive).
#
# The resulting CSVs are the central data artifact used by ALL downstream
# phases (4 spatial analysis, 5 model comparison, 6 calibration, 7 reporting).
#
# Output columns per prediction:
#   image_id, x_ctr, y_ctr, w, h, conf, cls,
#   pred_x_min, pred_y_min, pred_x_max, pred_y_max,
#   is_tp, is_fp, best_iou, matched_gt_class
#
# Outputs (saved to outputs/analysis_results/phase_3/):
#   predictions_tp_fp_labeled_model_a.csv
#   predictions_tp_fp_labeled_model_b.csv
#   fp_extraction_summary.csv
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

# =============================================================================
# PATHS
# =============================================================================
project_root    = Path("/home/sadeniji/scratch/Practicum_Main_Project")
model_a_weights = project_root / "outputs/model_a/model_a_parasite_detection-7/weights/best.pt"
model_b_weights = project_root / "outputs/model_b/model_b_parasite_wbc_detection/weights/best.pt"
test_images_a   = project_root / "model_a/test/images"
test_images_b   = project_root / "model_b/test/images"
test_labels_a   = project_root / "model_a/test/labels"
test_labels_b   = project_root / "model_b/test/labels"
out_dir         = project_root / "outputs/analysis_results/phase_3"
out_dir.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.10   # low threshold to capture all detections
IOU_THRESHOLD  = 0.50   # PASCAL standard for TP matching
IMG_W, IMG_H   = 2048, 2048   # actual test image dimensions (YOLO predictions are in original-image space)

for name, p in [("Model A weights", model_a_weights), ("Model B weights", model_b_weights),
                ("Test images A", test_images_a), ("Test labels A", test_labels_a),
                ("Test images B", test_images_b), ("Test labels B", test_labels_b)]:
    if not p.exists():
        print(f"WARNING: {name} not found at {p}")

# =============================================================================
# HELPERS
# =============================================================================
def load_ground_truth(labels_dir, img_w=IMG_W, img_h=IMG_H):
    gt = {}
    for lf in sorted(Path(labels_dir).glob("*.txt")):
        boxes = []
        for line in lf.read_text().strip().split('\n'):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = float(parts[1])*img_w, float(parts[2])*img_h, \
                              float(parts[3])*img_w, float(parts[4])*img_h
            boxes.append([xc - bw/2, yc - bh/2, xc + bw/2, yc + bh/2, cls_id])
        gt[lf.stem] = boxes
    return gt


def iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / max(union, 1e-9)


def label_predictions(results, ground_truth, iou_thresh=IOU_THRESHOLD):
    records = []
    for r in results:
        img_id  = Path(r.path).stem
        gt_boxes = ground_truth.get(img_id, [])
        matched  = [False] * len(gt_boxes)

        # Sort detections by confidence (high first) to greedily assign TPs
        boxes_sorted = sorted(r.boxes, key=lambda b: float(b.conf[0]), reverse=True)

        for box in boxes_sorted:
            xyxy = box.xyxy[0].tolist()
            xywh = box.xywh[0].tolist()
            conf = float(box.conf[0])
            cls  = int(box.cls[0])

            best_iou_val, best_gt_idx, best_gt_cls = 0.0, -1, -1
            for gi, gt in enumerate(gt_boxes):
                if matched[gi]:
                    continue
                ov = iou(xyxy, gt[:4])
                if ov > best_iou_val:
                    best_iou_val, best_gt_idx, best_gt_cls = ov, gi, gt[4]

            is_tp = best_iou_val >= iou_thresh
            if is_tp and best_gt_idx >= 0:
                matched[best_gt_idx] = True

            records.append({
                'image_id':       img_id,
                'x_ctr':          xywh[0],
                'y_ctr':          xywh[1],
                'w':              xywh[2],
                'h':              xywh[3],
                'conf':           conf,
                'cls':            cls,
                'pred_x_min':     xyxy[0],
                'pred_y_min':     xyxy[1],
                'pred_x_max':     xyxy[2],
                'pred_y_max':     xyxy[3],
                'is_tp':          is_tp,
                'is_fp':          not is_tp,
                'best_iou':       round(best_iou_val, 4),
                'matched_gt_cls': best_gt_cls,
            })
    return pd.DataFrame(records)


def print_summary(df, model_label):
    n_tp = df['is_tp'].sum()
    n_fp = df['is_fp'].sum()
    total = len(df)
    print(f"\n{'='*55}")
    print(f"{model_label}")
    print(f"{'='*55}")
    print(f"  Total predictions : {total}")
    print(f"  True Positives    : {n_tp}  ({n_tp/max(total,1)*100:.1f}%)")
    print(f"  False Positives   : {n_fp}  ({n_fp/max(total,1)*100:.1f}%)")
    print(f"  Images            : {df['image_id'].nunique()}")
    fp_df = df[df['is_fp']]
    tp_df = df[df['is_tp']]
    print(f"\n  FP confidence  — mean: {fp_df['conf'].mean():.4f}  "
          f"median: {fp_df['conf'].median():.4f}  std: {fp_df['conf'].std():.4f}")
    print(f"  TP confidence  — mean: {tp_df['conf'].mean():.4f}  "
          f"median: {tp_df['conf'].median():.4f}  std: {tp_df['conf'].std():.4f}")

# =============================================================================
# MODEL A
# =============================================================================
print("Loading ground truth for Model A...")
gt_a = load_ground_truth(test_labels_a)
print(f"  {len(gt_a)} images, {sum(len(v) for v in gt_a.values())} GT boxes")

print(f"\nRunning Model A inference (conf>={CONF_THRESHOLD})...")
model_a = YOLO(str(model_a_weights))
results_a = model_a.predict(source=str(test_images_a), conf=CONF_THRESHOLD,
                             save=False, verbose=False)

df_a = label_predictions(results_a, gt_a)
df_a.to_csv(out_dir / "predictions_tp_fp_labeled_model_a.csv", index=False)
print_summary(df_a, "Model A (parasite only)")

# =============================================================================
# MODEL B
# =============================================================================
print("\nLoading ground truth for Model B...")
gt_b = load_ground_truth(test_labels_b)
print(f"  {len(gt_b)} images, {sum(len(v) for v in gt_b.values())} GT boxes")

print(f"\nRunning Model B inference (conf>={CONF_THRESHOLD})...")
model_b = YOLO(str(model_b_weights))
results_b = model_b.predict(source=str(test_images_b), conf=CONF_THRESHOLD,
                             save=False, verbose=False)

df_b = label_predictions(results_b, gt_b)
df_b.to_csv(out_dir / "predictions_tp_fp_labeled_model_b.csv", index=False)
print_summary(df_b, "Model B (parasite + WBC)")

# =============================================================================
# COMBINED SUMMARY CSV
# =============================================================================
summary = []
for df, label in [(df_a, 'Model_A'), (df_b, 'Model_B')]:
    fp = df[df['is_fp']]
    tp = df[df['is_tp']]
    summary.append({
        'model': label,
        'n_total': len(df),
        'n_tp': int(df['is_tp'].sum()),
        'n_fp': int(df['is_fp'].sum()),
        'precision_at_0.25': round(len(df[df['conf']>=0.25]['is_tp']) /
                                    max(len(df[df['conf']>=0.25]), 1), 4),
        'fp_mean_conf': round(fp['conf'].mean(), 4),
        'tp_mean_conf': round(tp['conf'].mean(), 4),
        'n_images': df['image_id'].nunique(),
    })

pd.DataFrame(summary).to_csv(out_dir / "fp_extraction_summary.csv", index=False)
print(f"\nSummary saved to: {out_dir}/fp_extraction_summary.csv")
print("\nPhase 3 — FP extraction complete. CSVs ready for Phases 4–7.")
