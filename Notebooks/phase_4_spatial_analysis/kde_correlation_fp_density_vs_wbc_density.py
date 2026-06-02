# =============================================================================
# Phase 4 | KDE Correlation: FP Density Surface vs WBC Density Surface
#
# For each test image, builds two Kernel Density Estimation (KDE) surfaces:
#   - Surface 1: FP prediction density (where are FPs concentrated?)
#   - Surface 2: WBC annotation density (where are WBCs concentrated?)
#
# Computes the pixel-wise Pearson correlation between the two surfaces.
# A high positive r across images means FP density spatially tracks WBC
# density, supporting the WBC-proximity hypothesis.
#
# Also produces pooled density heatmaps by averaging all per-image surfaces
# to reveal systematic FP-dense regions (e.g., image center vs borders).
#
# Implementation: scipy.stats.gaussian_kde on a downsampled grid (64×64)
#   for computational efficiency. The bandwidth (sigma) is set by
#   Scott's rule (scipy default) per image.
#
# Outputs (saved to outputs/analysis_results/phase_4/):
#   kde_correlation_per_image.csv
#   kde_correlation_summary.csv
#   kde_pooled_density_heatmaps.png
#   kde_correlation_histogram.png
# =============================================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr, spearmanr

# =============================================================================
# PATHS
# =============================================================================
project_root  = Path("/home/sadeniji/scratch/Practicum_Main_Project")
phase3_dir    = project_root / "outputs/analysis_results/phase_3"
out_dir       = project_root / "outputs/analysis_results/phase_4"
out_dir.mkdir(parents=True, exist_ok=True)

pred_csv_b    = phase3_dir / "predictions_tp_fp_labeled_model_b.csv"
test_labels_b = project_root / "model_b/test/labels"

if not pred_csv_b.exists():
    sys.exit("ERROR: Run Phase 3 extract_predictions_and_label_tp_fp.py first.")

IMG_W, IMG_H = 1024, 1024
CONF_THRESH  = 0.25
GRID_SIZE    = 64    # downsample to 64×64 grid for efficiency

# =============================================================================
# LOAD DATA
# =============================================================================
def load_wbc_gt(labels_dir):
    rows = []
    for lf in sorted(Path(labels_dir).glob("*.txt")):
        for line in lf.read_text().strip().split('\n'):
            parts = line.strip().split()
            if len(parts) < 5 or int(parts[0]) != 1:
                continue
            rows.append({'image_id': lf.stem,
                         'x': float(parts[1])*IMG_W,
                         'y': float(parts[2])*IMG_H})
    return pd.DataFrame(rows)


df_b   = pd.read_csv(pred_csv_b)
df_b   = df_b[df_b['conf'] >= CONF_THRESH].copy()
df_wbc = load_wbc_gt(test_labels_b)
fps    = df_b[df_b['is_fp']].copy()

# =============================================================================
# BUILD KDE SURFACE FOR A SET OF POINTS
# =============================================================================
xgrid = np.linspace(0, IMG_W, GRID_SIZE)
ygrid = np.linspace(0, IMG_H, GRID_SIZE)
xx, yy = np.meshgrid(xgrid, ygrid)
grid_coords = np.vstack([xx.ravel(), yy.ravel()])


def kde_surface(points_xy, grid=grid_coords, shape=(GRID_SIZE, GRID_SIZE)):
    if len(points_xy) < 2:
        return np.zeros(shape)
    try:
        kde = gaussian_kde(points_xy.T)
        density = kde(grid).reshape(shape)
        return density
    except Exception:
        return np.zeros(shape)


# =============================================================================
# COMPUTE PER-IMAGE CORRELATIONS
# =============================================================================
shared_images = set(fps['image_id'].unique()) & set(df_wbc['image_id'].unique())
print(f"Images with FPs and WBC GT: {len(shared_images)}")

per_image_rows = []
pooled_fp_surfaces  = []
pooled_wbc_surfaces = []

for img_id in sorted(shared_images):
    fp_pts  = fps[fps['image_id'] == img_id][['x_ctr', 'y_ctr']].values
    wbc_pts = df_wbc[df_wbc['image_id'] == img_id][['x', 'y']].values

    if len(fp_pts) < 2 or len(wbc_pts) < 1:
        continue

    surf_fp  = kde_surface(fp_pts)
    surf_wbc = kde_surface(wbc_pts)

    flat_fp  = surf_fp.ravel()
    flat_wbc = surf_wbc.ravel()

    try:
        r_pearson, p_pearson   = pearsonr(flat_fp, flat_wbc)
        r_spearman, p_spearman = spearmanr(flat_fp, flat_wbc)
    except Exception:
        r_pearson = p_pearson = r_spearman = p_spearman = np.nan

    per_image_rows.append({
        'image_id': img_id,
        'n_fp': len(fp_pts),
        'n_wbc': len(wbc_pts),
        'pearson_r': round(r_pearson, 4),
        'pearson_p': round(p_pearson, 6) if not np.isnan(p_pearson) else np.nan,
        'spearman_r': round(r_spearman, 4),
        'spearman_p': round(p_spearman, 6) if not np.isnan(p_spearman) else np.nan,
    })

    pooled_fp_surfaces.append(surf_fp)
    pooled_wbc_surfaces.append(surf_wbc)

df_per_img = pd.DataFrame(per_image_rows)
df_per_img.to_csv(out_dir / "kde_correlation_per_image.csv", index=False)

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
r_vals = df_per_img['pearson_r'].dropna()
summary = {
    'n_images_analyzed': len(df_per_img),
    'mean_pearson_r': round(r_vals.mean(), 4),
    'median_pearson_r': round(r_vals.median(), 4),
    'std_pearson_r': round(r_vals.std(), 4),
    'pct_positive_r': round((r_vals > 0).mean() * 100, 1),
    'pct_r_above_0.3': round((r_vals > 0.3).mean() * 100, 1),
    'pct_r_above_0.5': round((r_vals > 0.5).mean() * 100, 1),
}
pd.DataFrame([summary]).to_csv(out_dir / "kde_correlation_summary.csv", index=False)

print("\n--- KDE Correlation Summary ---")
for k, v in summary.items():
    print(f"  {k}: {v}")

# =============================================================================
# POOLED DENSITY HEATMAPS
# =============================================================================
pooled_fp  = np.mean(pooled_fp_surfaces,  axis=0)
pooled_wbc = np.mean(pooled_wbc_surfaces, axis=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Pooled KDE Density Surfaces (averaged over test images)', fontsize=12)

im0 = axes[0].imshow(pooled_fp,  origin='lower', cmap='Reds',   aspect='auto')
axes[0].set_title('FP Detection Density'); plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(pooled_wbc, origin='lower', cmap='Blues',  aspect='auto')
axes[1].set_title('WBC Annotation Density'); plt.colorbar(im1, ax=axes[1])

diff = pooled_fp - pooled_fp.mean() - (pooled_wbc - pooled_wbc.mean())
im2 = axes[2].imshow(diff, origin='lower', cmap='RdBu_r', aspect='auto')
axes[2].set_title('FP − WBC density (red = excess FPs)')
plt.colorbar(im2, ax=axes[2])

for ax in axes:
    ax.set_xlabel('X'); ax.set_ylabel('Y')

plt.tight_layout()
plt.savefig(out_dir / "kde_pooled_density_heatmaps.png", dpi=150, bbox_inches='tight')

# =============================================================================
# HISTOGRAM OF PER-IMAGE PEARSON R
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df_per_img['pearson_r'].dropna(), bins=30, color='steelblue', edgecolor='white')
ax.axvline(r_vals.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean r = {r_vals.mean():.3f}')
ax.axvline(0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Pearson r (FP density vs WBC density)')
ax.set_ylabel('Number of Images')
ax.set_title('Per-Image Correlation: FP Density vs WBC Density')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "kde_correlation_histogram.png", dpi=150, bbox_inches='tight')

print(f"\nKDE correlation plots saved to: {out_dir}")
print("\nPhase 4 — KDE correlation analysis complete.")
