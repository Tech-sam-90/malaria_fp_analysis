# =============================================================================
# Phase 7 | Generate Spatial Heatmaps and KDE Overlays on Sample Images
#
# Produces the main spatial analysis figures for the report:
#   1. Per-image FP spatial heatmap (KDE) overlaid on actual test images
#   2. FP vs WBC density comparison on example images
#   3. Distance stratification summary bar chart
#
# Prerequisite: Phase 4 analyses must have run
#
# Outputs (saved to outputs/analysis_results/phase_7/figures/):
#   spatial_heatmap_kde_overlays.png
#   distance_stratification_final.png
# =============================================================================

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# =============================================================================
# PATHS
# =============================================================================
project_root = Path("/home/sadeniji/scratch/Practicum_Main_Project")
phase3_dir   = project_root / "outputs/analysis_results/phase_3"
phase4_dir   = project_root / "outputs/analysis_results/phase_4"
out_dir      = project_root / "outputs/analysis_results/phase_7/figures"
out_dir.mkdir(parents=True, exist_ok=True)

pred_csv_b    = phase3_dir / "predictions_tp_fp_labeled_model_b.csv"
test_images_b = project_root / "model_b/test/images"
test_labels_b = project_root / "model_b/test/labels"
strat_csv     = phase4_dir / "distance_stratification_results.csv"

IMG_W, IMG_H  = 1024, 1024
CONF_THRESH   = 0.25
GRID_DENSITY  = 128   # KDE grid resolution


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


def kde_surface(points_xy, grid_n=GRID_DENSITY):
    xg = np.linspace(0, IMG_W, grid_n)
    yg = np.linspace(0, IMG_H, grid_n)
    xx, yy = np.meshgrid(xg, yg)
    gcoords = np.vstack([xx.ravel(), yy.ravel()])
    if len(points_xy) < 2:
        return np.zeros((grid_n, grid_n))
    try:
        return gaussian_kde(points_xy.T)(gcoords).reshape(grid_n, grid_n)
    except Exception:
        return np.zeros((grid_n, grid_n))


# =============================================================================
# PLOT 1 — KDE OVERLAY ON SAMPLE IMAGES
# =============================================================================
df_b = pd.read_csv(pred_csv_b)
df_b = df_b[df_b['conf'] >= CONF_THRESH].copy()
fps  = df_b[df_b['is_fp']].copy()
tps  = df_b[df_b['is_tp']].copy()
df_wbc = load_wbc_gt(test_labels_b)

# Pick images with a mix of FPs and WBCs
candidate_imgs = (fps.groupby('image_id').size()
                     .reset_index(name='n_fp')
                     .sort_values('n_fp', ascending=False))
candidate_imgs = candidate_imgs[candidate_imgs['image_id'].isin(df_wbc['image_id'].unique())]
sample_ids = candidate_imgs.head(6)['image_id'].tolist()

fig, axes = plt.subplots(3, 6, figsize=(20, 10))
fig.suptitle('FP and WBC KDE Overlays on Sample Test Images', fontsize=13)
col_titles = ['Test Image', 'FP KDE Heatmap', 'WBC KDE Heatmap',
              'FP Density', 'WBC Density', 'FP−WBC Density Diff']

for j, title in enumerate(col_titles):
    axes[0, j].set_title(title, fontsize=9)

for row_idx, img_id in enumerate(sample_ids[:3]):
    img_files = sorted(test_images_b.glob(f"{img_id}.*"))
    fp_pts  = fps[fps['image_id'] == img_id][['x_ctr', 'y_ctr']].values
    wbc_pts = df_wbc[df_wbc['image_id'] == img_id][['x', 'y']].values

    surf_fp  = kde_surface(fp_pts)
    surf_wbc = kde_surface(wbc_pts)

    if img_files:
        img = cv2.imread(str(img_files[0]))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[row_idx, 0].imshow(img, aspect='auto')

    axes[row_idx, 1].imshow(surf_fp,  origin='lower', cmap='Reds',  aspect='auto')
    axes[row_idx, 2].imshow(surf_wbc, origin='lower', cmap='Blues', aspect='auto')

    axes[row_idx, 3].scatter(fp_pts[:, 0],  IMG_H - fp_pts[:, 1],
                              c='red', s=15, alpha=0.5)
    axes[row_idx, 4].scatter(wbc_pts[:, 0], IMG_H - wbc_pts[:, 1],
                              c='blue', s=30, alpha=0.5, marker='^')
    diff = surf_fp - surf_wbc
    axes[row_idx, 5].imshow(diff, origin='lower', cmap='RdBu_r', aspect='auto')

    for j in range(6):
        axes[row_idx, j].set_xticks([]); axes[row_idx, j].set_yticks([])
    axes[row_idx, 0].set_ylabel(img_id[:20], fontsize=7)

plt.tight_layout()
plt.savefig(out_dir / "spatial_heatmap_kde_overlays.png", dpi=130, bbox_inches='tight')
print(f"KDE overlay figure saved to: {out_dir}/spatial_heatmap_kde_overlays.png")

# =============================================================================
# PLOT 2 — DISTANCE STRATIFICATION FINAL CHART
# =============================================================================
if strat_csv.exists():
    df_strat = pd.read_csv(strat_csv)
    strat_a  = df_strat[df_strat['model'] == 'Model_A']
    strat_b  = df_strat[df_strat['model'] == 'Model_B']

    band_labels = strat_a['band'].tolist()
    x = np.arange(len(band_labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('FP Spatial Distribution by WBC-Proximity Band', fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.bar(x - width/2, strat_a['fp_rate'], width, label='Model A', color='steelblue', alpha=0.85)
    ax.bar(x + width/2, strat_b['fp_rate'], width, label='Model B', color='darkorange', alpha=0.85)
    ax.set_title('FP Rate per Band (FP / Total predictions)')
    ax.set_xticks(x); ax.set_xticklabels(band_labels, fontsize=9)
    ax.set_ylabel('FP Rate'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x - width/2, strat_a['pct_of_all_fp'], width, label='Model A', color='steelblue', alpha=0.85)
    ax.bar(x + width/2, strat_b['pct_of_all_fp'], width, label='Model B', color='darkorange', alpha=0.85)
    ax.set_title('% of All FPs per Band')
    ax.set_xticks(x); ax.set_xticklabels(band_labels, fontsize=9)
    ax.set_ylabel('% of All FPs'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_dir / "distance_stratification_final.png", dpi=150, bbox_inches='tight')
    print(f"Distance stratification chart saved to: {out_dir}/distance_stratification_final.png")

print("\nPhase 7 — Spatial heatmap figures generated.")
