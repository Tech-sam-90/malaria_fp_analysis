# =============================================================================
# Sample Detection Figure — TP (green) and FP (red) boxes on real thick smear
# Selects images with a mix of TPs and FPs for representative visualisation.
# Output: outputs/analysis_results/phase_3/sample_detection_figure.png
# =============================================================================
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

project_root  = Path("/home/sadeniji/scratch/Practicum_Main_Project")
phase3_dir    = project_root / "outputs/analysis_results/phase_3"
images_dir_a  = project_root / "model_a/test/images"
images_dir_b  = project_root / "model_b/test/images"
labels_dir_a  = project_root / "model_a/test/labels"
out_dir       = phase3_dir

IMG_W, IMG_H = 2048, 2048

df_a = pd.read_csv(phase3_dir / "predictions_tp_fp_labeled_model_a.csv")
df_b = pd.read_csv(phase3_dir / "predictions_tp_fp_labeled_model_b.csv")

# Pick 3 images: (1) balanced TP/FP, (2) high FP, (3) clean
stats = df_a.groupby('image_id').agg(
    n_tp=('is_tp', 'sum'), n_fp=('is_fp', 'sum')
).reset_index()
stats['total'] = stats['n_tp'] + stats['n_fp']
stats = stats[stats['total'] >= 5]

# Image 1: balanced (close FP/TP ratio, decent count)
stats['balance'] = abs(stats['n_tp'] - stats['n_fp'])
img1_id = stats[(stats['n_tp'] >= 3) & (stats['n_fp'] >= 3)].sort_values('balance').iloc[0]['image_id']
# Image 2: high FP rate
img2_id = stats[stats['n_fp'] >= 4].sort_values('n_fp', ascending=False).iloc[0]['image_id']
# Image 3: mostly TP (clean)
img3_id = stats[stats['n_tp'] >= 5].sort_values('n_tp', ascending=False).iloc[2]['image_id']

chosen = [img1_id, img2_id, img3_id]
titles = ['Balanced TP/FP', 'High FP image', 'High TP image']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model A Detection Examples on Thick Giemsa Blood Smear\n'
             'Green = True Positive  |  Red = False Positive  |  Blue dashed = Ground Truth',
             fontsize=12, fontweight='bold')

def load_gt(img_id, labels_dir):
    lf = Path(labels_dir) / (img_id + ".txt")
    boxes = []
    if lf.exists():
        for line in lf.read_text().strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 5 and int(parts[0]) == 0:
                xc, yc = float(parts[1])*IMG_W, float(parts[2])*IMG_H
                bw, bh = float(parts[3])*IMG_W, float(parts[4])*IMG_H
                boxes.append((xc - bw/2, yc - bh/2, bw, bh))
    return boxes

for ax, img_id, title in zip(axes, chosen, titles):
    # Find image file
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        p = images_dir_a / (img_id + ext)
        if p.exists():
            img_path = p
            break
    if img_path is None:
        ax.set_visible(False)
        continue

    img = Image.open(img_path)
    ax.imshow(img, cmap='gray' if img.mode == 'L' else None)

    # Draw GT boxes (blue dashed)
    for (x, y, w, h) in load_gt(img_id, labels_dir_a):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="square,pad=0", linewidth=1.2,
            edgecolor='#3498db', facecolor='none', linestyle='--')
        ax.add_patch(rect)

    # Draw predictions
    preds = df_a[df_a['image_id'] == img_id]
    for _, row in preds.iterrows():
        color = '#2ecc71' if row['is_tp'] else '#e74c3c'
        lw = 1.8
        rect = mpatches.FancyBboxPatch(
            (row['pred_x_min'], row['pred_y_min']),
            row['pred_x_max'] - row['pred_x_min'],
            row['pred_y_max'] - row['pred_y_min'],
            boxstyle="square,pad=0", linewidth=lw,
            edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # Confidence label for high-confidence FPs
        if row['is_fp'] and row['conf'] >= 0.40:
            ax.text(row['pred_x_min'], row['pred_y_min'] - 8,
                    f"{row['conf']:.2f}", color='#e74c3c',
                    fontsize=6, fontweight='bold')

    n_tp = int(preds['is_tp'].sum())
    n_fp = int(preds['is_fp'].sum())
    ax.set_title(f'{title}\nTP={n_tp}  FP={n_fp}', fontsize=10, fontweight='bold')
    ax.axis('off')

# Legend
legend_elements = [
    mpatches.Patch(facecolor='none', edgecolor='#2ecc71', linewidth=2, label='True Positive'),
    mpatches.Patch(facecolor='none', edgecolor='#e74c3c', linewidth=2, label='False Positive'),
    mpatches.Patch(facecolor='none', edgecolor='#3498db', linewidth=1.5,
                   linestyle='--', label='Ground Truth'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
out_path = out_dir / "sample_detection_figure.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
print(f"Selected images: {chosen}")
