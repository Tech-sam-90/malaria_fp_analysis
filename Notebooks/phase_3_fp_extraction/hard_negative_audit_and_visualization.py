# =============================================================================
# Phase 3 | Hard Negative Audit and Visualization
#
# Identifies and visually inspects the "hardest" false positives — the
# high-confidence FPs that the model is most certain about yet are wrong.
# These are the most informative for understanding model failure modes.
#
# Three audit categories are produced:
#   1. High-confidence FPs (conf >= 0.50): model is very confident but wrong
#   2. WBC-proximal FPs: FPs within 60px of an annotated WBC center
#   3. Isolated FPs: FPs far (>200px) from any WBC — background/artifact errors
#
# For each category, sample images are saved with FP boxes drawn in red and
# nearby GT boxes drawn in green, enabling rapid visual inspection.
#
# Prerequisite: Phase 3 extract_predictions_and_label_tp_fp.py
#
# Outputs (saved to outputs/analysis_results/phase_3/hard_negatives/):
#   hard_negatives_high_confidence.csv
#   hard_negatives_wbc_proximal.csv
#   hard_negatives_isolated.csv
#   sample_images/*.jpg  (annotated image crops for each category)
# =============================================================================

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# =============================================================================
# PATHS
# =============================================================================
project_root  = Path("/home/sadeniji/scratch/Practicum_Main_Project")
phase3_dir    = project_root / "outputs/analysis_results/phase_3"
out_dir       = phase3_dir / "hard_negatives"
sample_dir    = out_dir / "sample_images"
out_dir.mkdir(parents=True, exist_ok=True)
sample_dir.mkdir(parents=True, exist_ok=True)

pred_csv_a = phase3_dir / "predictions_tp_fp_labeled_model_a.csv"
pred_csv_b = phase3_dir / "predictions_tp_fp_labeled_model_b.csv"
test_images_a = project_root / "model_a/test/images"
test_images_b = project_root / "model_b/test/images"
test_labels_b = project_root / "model_b/test/labels"   # used to get WBC GT positions

for name, p in [("Model A CSV", pred_csv_a), ("Model B CSV", pred_csv_b)]:
    if not p.exists():
        sys.exit(f"ERROR: {name} not found. Run extract_predictions_and_label_tp_fp.py first.")

IMG_W, IMG_H = 1024, 1024
TOP_N = 50          # number of hard negatives to keep per category
WBC_PROX_PX = 60    # pixels — FP within this distance is "WBC-proximal"
ISOLATED_PX = 200   # pixels — FP further than this is "isolated"

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading labeled predictions...")
df_a = pd.read_csv(pred_csv_a)
df_b = pd.read_csv(pred_csv_b)

fps_a = df_a[df_a['is_fp']].copy().reset_index(drop=True)
fps_b = df_b[df_b['is_fp']].copy().reset_index(drop=True)

# Load WBC GT centers from Model B labels
def load_wbc_centers(labels_dir):
    wbc_rows = []
    for lf in sorted(Path(labels_dir).glob("*.txt")):
        for line in lf.read_text().strip().split('\n'):
            parts = line.strip().split()
            if len(parts) < 5 or int(parts[0]) != 1:  # class 1 = WBC
                continue
            xc, yc = float(parts[1])*IMG_W, float(parts[2])*IMG_H
            wbc_rows.append({'image_id': lf.stem, 'wbc_x': xc, 'wbc_y': yc})
    return pd.DataFrame(wbc_rows)

df_wbc = load_wbc_centers(test_labels_b)
print(f"WBC GT centers loaded: {len(df_wbc)} across {df_wbc['image_id'].nunique()} images")

# =============================================================================
# COMPUTE DISTANCE FROM EACH FP TO NEAREST WBC
# =============================================================================
def add_wbc_distance(fps_df, wbc_df):
    dists = []
    for img_id, group in fps_df.groupby('image_id'):
        wbc_in_img = wbc_df[wbc_df['image_id'] == img_id]
        fp_coords = group[['x_ctr', 'y_ctr']].values
        if len(wbc_in_img) == 0:
            dists.extend([np.nan] * len(group))
        else:
            tree = cKDTree(wbc_in_img[['wbc_x', 'wbc_y']].values)
            d, _ = tree.query(fp_coords)
            dists.extend(d.tolist())
    fps_df = fps_df.copy()
    fps_df['dist_to_nearest_wbc'] = dists
    return fps_df

print("Computing FP-to-WBC distances...")
fps_b = add_wbc_distance(fps_b, df_wbc)

# =============================================================================
# CATEGORIZE HARD NEGATIVES
# =============================================================================
def get_top_n(df, sort_col, ascending, n=TOP_N):
    return df.sort_values(sort_col, ascending=ascending).head(n).copy()

# Category 1: highest-confidence FPs (model most wrong)
hn_high_conf = get_top_n(fps_b, 'conf', ascending=False)
hn_high_conf.to_csv(out_dir / "hard_negatives_high_confidence.csv", index=False)
print(f"\nHigh-confidence FPs (top {TOP_N}): mean conf = {hn_high_conf['conf'].mean():.4f}")

# Category 2: WBC-proximal FPs (near annotated WBC centers)
fps_b_valid = fps_b[fps_b['dist_to_nearest_wbc'].notna()]
hn_wbc_prox = fps_b_valid[fps_b_valid['dist_to_nearest_wbc'] <= WBC_PROX_PX]
hn_wbc_prox = get_top_n(hn_wbc_prox, 'conf', ascending=False)
hn_wbc_prox.to_csv(out_dir / "hard_negatives_wbc_proximal.csv", index=False)
print(f"WBC-proximal FPs (dist<={WBC_PROX_PX}px): {len(hn_wbc_prox)} sampled")

# Category 3: isolated FPs (far from any WBC — background/artifact errors)
hn_isolated = fps_b_valid[fps_b_valid['dist_to_nearest_wbc'] > ISOLATED_PX]
hn_isolated = get_top_n(hn_isolated, 'conf', ascending=False)
hn_isolated.to_csv(out_dir / "hard_negatives_isolated.csv", index=False)
print(f"Isolated FPs (dist>{ISOLATED_PX}px): {len(hn_isolated)} sampled")

# =============================================================================
# HELPER — draw annotated image crop around an FP
# =============================================================================
def draw_fp_on_image(img_path, fp_row, gt_boxes, wbc_centers, save_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw all GT parasite boxes in green
    for box in gt_boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)

    # Draw WBC centers in blue
    for _, wbc in wbc_centers.iterrows():
        cx, cy = int(wbc['wbc_x']), int(wbc['wbc_y'])
        cv2.circle(img, (cx, cy), 20, (0, 100, 255), 2)

    # Draw the FP box in red
    x1 = int(fp_row['pred_x_min']) if 'pred_x_min' in fp_row else int(fp_row['x_ctr'] - fp_row['w']/2)
    y1 = int(fp_row['pred_y_min']) if 'pred_y_min' in fp_row else int(fp_row['y_ctr'] - fp_row['h']/2)
    x2 = int(fp_row['pred_x_max']) if 'pred_x_max' in fp_row else int(fp_row['x_ctr'] + fp_row['w']/2)
    y2 = int(fp_row['pred_y_max']) if 'pred_y_max' in fp_row else int(fp_row['y_ctr'] + fp_row['h']/2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(img, f"FP conf={fp_row['conf']:.2f}", (x1, max(y1-8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


# =============================================================================
# SAVE SAMPLE IMAGES FOR EACH CATEGORY
# =============================================================================
def load_gt_for_image(labels_dir, img_id, cls_filter=None):
    lf = Path(labels_dir) / (img_id + ".txt")
    boxes = []
    if not lf.exists():
        return boxes
    for line in lf.read_text().strip().split('\n'):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        if cls_filter is not None and cls_id not in cls_filter:
            continue
        xc, yc = float(parts[1])*IMG_W, float(parts[2])*IMG_H
        bw, bh = float(parts[3])*IMG_W, float(parts[4])*IMG_H
        boxes.append([xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2, cls_id])
    return boxes


SAVE_N = 20   # save images for top 20 per category
print(f"\nSaving annotated sample images (top {SAVE_N} per category)...")

for category, hn_df, label in [
    ('high_conf',   hn_high_conf, 'high_confidence'),
    ('wbc_proximal', hn_wbc_prox, 'wbc_proximal'),
    ('isolated',    hn_isolated,  'isolated'),
]:
    cat_dir = sample_dir / label
    cat_dir.mkdir(exist_ok=True)
    for i, (_, row) in enumerate(hn_df.head(SAVE_N).iterrows()):
        img_id = row['image_id']
        img_files = list(test_images_b.glob(f"{img_id}.*"))
        if not img_files:
            continue
        gt_boxes = load_gt_for_image(test_labels_b, img_id, cls_filter=[0])
        wbc_in_img = df_wbc[df_wbc['image_id'] == img_id]
        save_path = cat_dir / f"{i:02d}_{img_id[:30]}_conf{row['conf']:.2f}.jpg"
        try:
            draw_fp_on_image(img_files[0], row, gt_boxes, wbc_in_img, save_path)
        except Exception as e:
            print(f"  Could not save image {img_id}: {e}")
    print(f"  {label}: saved to {cat_dir}")

print("\nPhase 3 — Hard negative audit complete.")
