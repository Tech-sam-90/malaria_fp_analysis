# =============================================================================
# Phase 5 | Wilcoxon Signed-Rank Test on Per-Image AP50
#
# Non-parametric paired test comparing Model A vs Model B per-image AP50.
# Preferred over paired t-test because per-image AP50 is:
#   - Non-normal (many values are 0 or 1 — floor/ceiling effects)
#   - Paired (same test images for both models)
#
# Also includes the per-image AP scatterplot (A vs B) and a signed-rank
# chart showing which images Model B improved vs. degraded compared to A.
#
# Prerequisite: Phase 5 paired_bootstrap_map_comparison.py must have run
#   (reuses its per-image AP computation logic)
#
# Outputs (saved to outputs/analysis_results/phase_5/):
#   wilcoxon_per_image_ap_results.csv
#   per_image_ap_scatterplot.png
#   per_image_ap_delta_distribution.png
# =============================================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# =============================================================================
# PATHS
# =============================================================================
project_root  = Path("/home/sadeniji/scratch/Practicum_Main_Project")
phase3_dir    = project_root / "outputs/analysis_results/phase_3"
out_dir       = project_root / "outputs/analysis_results/phase_5"
out_dir.mkdir(parents=True, exist_ok=True)

pred_csv_a    = phase3_dir / "predictions_tp_fp_labeled_model_a.csv"
pred_csv_b    = phase3_dir / "predictions_tp_fp_labeled_model_b.csv"
test_labels_a = project_root / "model_a/test/labels"

CONF_THRESH = 0.25
IOU_THRESH  = 0.50
IMG_W, IMG_H = 1024, 1024

# =============================================================================
# COMPUTE PER-IMAGE AP50 (same logic as bootstrap script)
# =============================================================================
def compute_per_image_ap(pred_df, gt_dir, iou_thresh=IOU_THRESH, conf_thresh=CONF_THRESH):
    ap_per_image = {}
    for img_id in pred_df['image_id'].unique():
        lf = Path(gt_dir) / (img_id + ".txt")
        gt_boxes = []
        if lf.exists():
            for line in lf.read_text().strip().split('\n'):
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 0:
                    xc, yc = float(parts[1])*IMG_W, float(parts[2])*IMG_H
                    bw, bh = float(parts[3])*IMG_W, float(parts[4])*IMG_H
                    gt_boxes.append([xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2])
        n_gt = len(gt_boxes)
        if n_gt == 0:
            ap_per_image[img_id] = np.nan; continue

        preds = pred_df[(pred_df['image_id'] == img_id) &
                        (pred_df['conf'] >= conf_thresh) &
                        (pred_df['cls'] == 0)].sort_values('conf', ascending=False)
        if len(preds) == 0:
            ap_per_image[img_id] = 0.0; continue

        tp_cum = np.cumsum(preds['is_tp'].values.astype(float))
        fp_cum = np.cumsum(preds['is_fp'].values.astype(float))
        precision = tp_cum / (tp_cum + fp_cum)
        recall    = tp_cum / n_gt
        ap = sum(precision[recall >= t].max() if (recall >= t).any() else 0
                 for t in np.linspace(0, 1, 11)) / 11.0
        ap_per_image[img_id] = ap
    return ap_per_image


print("Computing per-image AP50...")
df_a = pd.read_csv(pred_csv_a)
df_b = pd.read_csv(pred_csv_b)
ap_a = compute_per_image_ap(df_a, test_labels_a)
ap_b = compute_per_image_ap(df_b, test_labels_a)

shared = sorted(set(ap_a) & set(ap_b))
shared = [i for i in shared if not (np.isnan(ap_a[i]) or np.isnan(ap_b[i]))]
a_arr  = np.array([ap_a[i] for i in shared])
b_arr  = np.array([ap_b[i] for i in shared])
delta  = b_arr - a_arr

print(f"Images: {len(shared)}  |  mean AP A={a_arr.mean():.4f}  B={b_arr.mean():.4f}")

# =============================================================================
# WILCOXON SIGNED-RANK TEST
# =============================================================================
stat, p_val = wilcoxon(a_arr, b_arr, alternative='two-sided')

# Effect size: rank-biserial correlation
n = len(delta[delta != 0])
r_rb = 1 - (2 * stat) / (n * (n + 1))

print(f"\nWilcoxon Signed-Rank Test (two-sided)")
print(f"  W statistic  : {stat:.2f}")
print(f"  p-value      : {p_val:.6f}")
print(f"  Rank-biserial r : {r_rb:.4f}")

if p_val < 0.05:
    better = "B" if b_arr.mean() > a_arr.mean() else "A"
    print(f"  *** SIGNIFICANT: Model {better} has higher per-image AP50 (p < 0.05) ***")
else:
    print("  NOT SIGNIFICANT at p=0.05")

df_image_ap = pd.DataFrame({'image_id': shared, 'ap_model_a': a_arr,
                              'ap_model_b': b_arr, 'delta_b_minus_a': delta})
df_image_ap.to_csv(out_dir / "per_image_ap50_model_a_vs_b.csv", index=False)

pd.DataFrame([{
    'W_statistic': stat, 'p_value_two_sided': p_val,
    'rank_biserial_r': r_rb,
    'mean_ap_a': a_arr.mean(), 'mean_ap_b': b_arr.mean(),
    'n_images': len(shared),
}]).to_csv(out_dir / "wilcoxon_per_image_ap_results.csv", index=False)

# =============================================================================
# PLOT 1 — SCATTER: Model A AP vs Model B AP per image
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
colors = np.where(delta > 0, 'green', np.where(delta < 0, 'red', 'grey'))
ax.scatter(a_arr, b_arr, c=colors, alpha=0.4, s=25, edgecolors='none')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='y = x (equal performance)')
ax.set_xlabel('Per-Image AP50: Model A')
ax.set_ylabel('Per-Image AP50: Model B')
ax.set_title('Per-Image AP50: Model A vs Model B\n(green=B better, red=A better)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# =============================================================================
# PLOT 2 — DELTA DISTRIBUTION
# =============================================================================
ax = axes[1]
ax.hist(delta, bins=40, color='steelblue', edgecolor='white')
ax.axvline(0, color='black', linewidth=1.5, label='No difference')
ax.axvline(delta.mean(), color='red', linewidth=2,
           label=f'Mean delta = {delta.mean():+.4f}')
pct_improved  = (delta > 0).mean() * 100
pct_degraded  = (delta < 0).mean() * 100
ax.set_title(f'Per-Image AP50 Delta (B − A)\n'
             f'{pct_improved:.1f}% improved, {pct_degraded:.1f}% degraded')
ax.set_xlabel('Delta (Model B AP50 − Model A AP50)')
ax.set_ylabel('Number of Images')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "per_image_ap_scatterplot_and_delta.png", dpi=150, bbox_inches='tight')

print(f"\nPlots saved to: {out_dir}/per_image_ap_scatterplot_and_delta.png")
print("\nPhase 5 — Wilcoxon signed-rank test complete.")
