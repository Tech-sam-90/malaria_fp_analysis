# =============================================================================
# Phase 4 | Moran's I: Spatial Autocorrelation of FP Counts
#
# Tests whether FPs are spatially autocorrelated — i.e., do high-FP tiles
# cluster together, or are FPs randomly distributed across image regions?
#
# Method:
#   1. Divide each test image into a grid of 50×50 px tiles
#   2. Count FPs per tile → spatial count surface
#   3. Compute Global Moran's I with queen-contiguity weights
#      I > 0: clustered; I < 0: dispersed; I ≈ 0: random
#   4. Compute Local Moran's I (LISA) to identify FP hot-spot tiles
#   5. Compare hot-spot tiles against WBC density per tile
#
# Positive global Moran's I confirms that FPs cluster spatially (not random),
# motivating the WBC-proximity analysis. LISA maps show where FP clusters
# are — these are expected to overlap with WBC-dense regions.
#
# Implementation: esda + libpysal (PySAL ecosystem)
#   Install: pip install esda libpysal
#
# Outputs (saved to outputs/analysis_results/phase_4/):
#   morans_i_global_results.csv
#   morans_i_local_hotspots.csv
#   morans_i_global_distribution.png
#   morans_i_hotspot_map_mean.png
# =============================================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import esda
    import libpysal
    from esda.moran import Moran, Moran_Local
    from libpysal.weights import lat2W
except ImportError:
    sys.exit("ERROR: esda or libpysal not installed.\n"
             "Run: pip install esda libpysal")

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
TILE_SIZE    = 50    # pixels per grid tile → 1024/50 ≈ 20 tiles per side

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

N_TILES_X = IMG_W // TILE_SIZE
N_TILES_Y = IMG_H // TILE_SIZE
N_TILES   = N_TILES_X * N_TILES_Y
print(f"Grid: {N_TILES_X} × {N_TILES_Y} = {N_TILES} tiles of {TILE_SIZE}×{TILE_SIZE} px")

# Queen-contiguity weights matrix (reused for all images)
W = lat2W(N_TILES_Y, N_TILES_X, rook=False)
W.transform = 'r'   # row-standardize

# =============================================================================
# HELPER — count points in each tile
# =============================================================================
def points_to_tile_counts(x_coords, y_coords, n_x=N_TILES_X, n_y=N_TILES_Y,
                           tile=TILE_SIZE, w=IMG_W, h=IMG_H):
    counts = np.zeros((n_y, n_x), dtype=float)
    for x, y in zip(x_coords, y_coords):
        tx = min(int(x / tile), n_x - 1)
        ty = min(int(y / tile), n_y - 1)
        counts[ty, tx] += 1
    return counts.ravel()   # flatten to 1D in row-major order (matches lat2W)

# =============================================================================
# COMPUTE MORAN'S I PER IMAGE
# =============================================================================
global_rows   = []
local_hotspot_surfaces = []
local_wbc_surfaces     = []

images_with_fps = fps['image_id'].unique()
print(f"Computing Moran's I for {len(images_with_fps)} images...")

for img_id in sorted(images_with_fps):
    fp_pts  = fps[fps['image_id'] == img_id][['x_ctr', 'y_ctr']]
    wbc_pts = df_wbc[df_wbc['image_id'] == img_id][['x', 'y']]

    fp_counts  = points_to_tile_counts(fp_pts['x_ctr'], fp_pts['y_ctr'])
    wbc_counts = points_to_tile_counts(wbc_pts['x'],    wbc_pts['y'])

    if fp_counts.sum() < 2:
        continue

    # Global Moran's I
    try:
        mi = Moran(fp_counts, W, permutations=199)
        global_rows.append({
            'image_id': img_id,
            'n_fp': int(fp_counts.sum()),
            'n_wbc': int(wbc_counts.sum()),
            'moran_I': round(mi.I, 4),
            'expected_I': round(mi.EI, 6),
            'z_score': round(mi.z_sim, 4),
            'p_value': round(mi.p_sim, 4),
            'significant_p05': mi.p_sim < 0.05,
        })
    except Exception as e:
        print(f"  [{img_id}] Moran error: {e}")
        continue

    # Local Moran's I (LISA) — identify hot-spot tiles
    try:
        lmi = Moran_Local(fp_counts, W, permutations=99)
        hotspot_mask = (lmi.q == 1) & (lmi.p_sim < 0.05)   # HH cluster (hot-spot)
        local_hotspot_surfaces.append(hotspot_mask.reshape(N_TILES_Y, N_TILES_X).astype(float))
        local_wbc_surfaces.append(wbc_counts.reshape(N_TILES_Y, N_TILES_X))
    except Exception:
        pass

# =============================================================================
# SAVE GLOBAL RESULTS
# =============================================================================
df_global = pd.DataFrame(global_rows)
df_global.to_csv(out_dir / "morans_i_global_results.csv", index=False)

print(f"\n--- Global Moran's I Summary ---")
print(f"  Images analyzed            : {len(df_global)}")
print(f"  Mean I                     : {df_global['moran_I'].mean():.4f}")
print(f"  Median I                   : {df_global['moran_I'].median():.4f}")
print(f"  % significant (p<0.05)     : {df_global['significant_p05'].mean()*100:.1f}%")
print(f"  % positive I               : {(df_global['moran_I'] > 0).mean()*100:.1f}%")

# =============================================================================
# MEAN HOT-SPOT MAP vs MEAN WBC MAP
# =============================================================================
if local_hotspot_surfaces:
    mean_hotspot = np.mean(local_hotspot_surfaces, axis=0)
    mean_wbc     = np.mean(local_wbc_surfaces,     axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Moran's I LISA Hot-Spot Analysis (averaged over images)", fontsize=12)

    im0 = axes[0].imshow(mean_hotspot, cmap='hot', aspect='auto', origin='lower')
    axes[0].set_title('Mean FP Hot-Spot Frequency\n(HH cluster, p<0.05)')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(mean_wbc, cmap='Blues', aspect='auto', origin='lower')
    axes[1].set_title('Mean WBC Count per Tile')
    plt.colorbar(im1, ax=axes[1])

    overlap = mean_hotspot * (mean_wbc / max(mean_wbc.max(), 1))
    im2 = axes[2].imshow(overlap, cmap='Purples', aspect='auto', origin='lower')
    axes[2].set_title('Overlap: Hot-Spots × WBC Density')
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel('Tile X'); ax.set_ylabel('Tile Y')

    plt.tight_layout()
    plt.savefig(out_dir / "morans_i_hotspot_map_mean.png", dpi=150, bbox_inches='tight')
    print(f"\nHot-spot map saved to: {out_dir}/morans_i_hotspot_map_mean.png")

# =============================================================================
# DISTRIBUTION OF GLOBAL MORAN'S I
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df_global['moran_I'], bins=30, color='steelblue', edgecolor='white')
ax.axvline(df_global['moran_I'].mean(), color='red', linestyle='--', linewidth=2,
           label=f"Mean I = {df_global['moran_I'].mean():.4f}")
ax.axvline(df_global['expected_I'].mean(), color='black', linestyle='-', linewidth=1,
           label=f"Expected I (CSR) = {df_global['expected_I'].mean():.4f}")
ax.set_xlabel("Moran's I"); ax.set_ylabel('Number of Images')
ax.set_title("Distribution of Global Moran's I Across Test Images")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "morans_i_global_distribution.png", dpi=150, bbox_inches='tight')

print("\nPhase 4 — Moran's I spatial autocorrelation complete.")
