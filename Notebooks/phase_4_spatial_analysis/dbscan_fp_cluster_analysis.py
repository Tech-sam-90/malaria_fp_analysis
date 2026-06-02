# =============================================================================
# Phase 4 | DBSCAN: Spatial Clustering of FP Predictions
#
# Applies DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# to the FP prediction centroids to identify whether FPs form dense spatial
# clusters and whether those clusters co-locate with annotated WBC positions.
#
# DBSCAN is preferred over k-means here because:
#   - No need to specify the number of clusters in advance
#   - Handles clusters of arbitrary shape
#   - Explicitly marks sparse/noise points (label = -1)
#
# Parameters:
#   eps = 60 px  (approx. 1–2 WBC diameters at 1024px resolution)
#   min_samples = 3  (minimum 3 FPs to form a core cluster point)
#
# For each cluster identified:
#   - Compute cluster centroid
#   - Find nearest annotated WBC center (distance)
#   - Classify: WBC-associated (dist < 80px) vs background cluster
#
# Outputs (saved to outputs/analysis_results/phase_4/):
#   dbscan_cluster_summary.csv
#   dbscan_per_image_stats.csv
#   dbscan_cluster_wbc_proximity.png
#   dbscan_sample_image_overlay.png   (first 6 images with clusters drawn)
# =============================================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

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

IMG_W, IMG_H  = 1024, 1024
CONF_THRESH   = 0.25
EPS           = 60      # pixel radius for neighborhood
MIN_SAMPLES   = 3       # minimum FPs to form a cluster core point
WBC_ASSOC_PX  = 80      # cluster centroid within this dist of WBC = WBC-associated

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
# RUN DBSCAN PER IMAGE
# =============================================================================
per_image_rows = []
cluster_rows   = []

for img_id, group in fps.groupby('image_id'):
    coords = group[['x_ctr', 'y_ctr']].values
    if len(coords) < MIN_SAMPLES:
        continue

    labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(coords).labels_

    n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise     = (labels == -1).sum()
    n_clustered = (labels != -1).sum()

    wbc_in_img  = df_wbc[df_wbc['image_id'] == img_id][['x', 'y']].values
    wbc_tree    = cKDTree(wbc_in_img) if len(wbc_in_img) > 0 else None

    per_image_rows.append({
        'image_id': img_id,
        'n_fp': len(coords),
        'n_clusters': n_clusters,
        'n_noise_fps': int(n_noise),
        'n_clustered_fps': int(n_clustered),
        'cluster_ratio': round(n_clustered / max(len(coords), 1), 4),
        'n_wbc_annotations': len(wbc_in_img),
    })

    for cid in range(n_clusters):
        mask = labels == cid
        pts  = coords[mask]
        ctr  = pts.mean(axis=0)

        dist_to_wbc = np.nan
        if wbc_tree is not None:
            dist_to_wbc = float(wbc_tree.query([ctr])[0])

        cluster_rows.append({
            'image_id': img_id,
            'cluster_id': cid,
            'n_fp_in_cluster': int(mask.sum()),
            'centroid_x': round(ctr[0], 2),
            'centroid_y': round(ctr[1], 2),
            'dist_to_nearest_wbc': round(dist_to_wbc, 2) if not np.isnan(dist_to_wbc) else np.nan,
            'wbc_associated': dist_to_wbc < WBC_ASSOC_PX if not np.isnan(dist_to_wbc) else False,
        })

df_per_image = pd.DataFrame(per_image_rows)
df_clusters  = pd.DataFrame(cluster_rows)

df_per_image.to_csv(out_dir / "dbscan_per_image_stats.csv", index=False)
df_clusters.to_csv(out_dir / "dbscan_cluster_summary.csv", index=False)

# =============================================================================
# PRINT SUMMARY
# =============================================================================
print(f"\n{'='*55}")
print("DBSCAN Clustering Summary")
print(f"{'='*55}")
print(f"  eps={EPS}px, min_samples={MIN_SAMPLES}")
print(f"  Images with >= {MIN_SAMPLES} FPs analyzed : {len(df_per_image)}")
print(f"  Total DBSCAN clusters found   : {len(df_clusters)}")
print(f"  Mean clusters per image       : {df_per_image['n_clusters'].mean():.2f}")
print(f"  Mean FP clustering ratio      : {df_per_image['cluster_ratio'].mean():.4f}")

if len(df_clusters) > 0:
    wbc_assoc = df_clusters['wbc_associated'].sum()
    total_c = len(df_clusters)
    print(f"\n  WBC-associated clusters (<{WBC_ASSOC_PX}px) : "
          f"{wbc_assoc}/{total_c} ({wbc_assoc/total_c*100:.1f}%)")
    valid_dists = df_clusters['dist_to_nearest_wbc'].dropna()
    print(f"  Mean dist cluster→WBC         : {valid_dists.mean():.1f} px")
    print(f"  Median dist cluster→WBC       : {valid_dists.median():.1f} px")

# =============================================================================
# PLOT 1 — CLUSTER WBC PROXIMITY DISTRIBUTION
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('DBSCAN FP Cluster Analysis', fontsize=12)

ax = axes[0]
valid_d = df_clusters['dist_to_nearest_wbc'].dropna()
ax.hist(valid_d, bins=30, color='steelblue', edgecolor='white')
ax.axvline(WBC_ASSOC_PX, color='red', linestyle='--', linewidth=2,
           label=f'WBC-assoc threshold ({WBC_ASSOC_PX}px)')
ax.axvline(valid_d.mean(), color='darkorange', linestyle='-', linewidth=2,
           label=f'Mean = {valid_d.mean():.0f}px')
ax.set_xlabel('Distance: Cluster Centroid → Nearest WBC (px)')
ax.set_ylabel('Number of Clusters')
ax.set_title('Cluster Proximity to WBC Annotations')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(df_clusters['n_fp_in_cluster'],
           df_clusters['dist_to_nearest_wbc'],
           c=df_clusters['wbc_associated'].map({True: 'red', False: 'steelblue'}),
           alpha=0.6, s=40)
ax.set_xlabel('FPs in Cluster')
ax.set_ylabel('Distance to Nearest WBC (px)')
ax.set_title('Cluster Size vs WBC Distance')
red_patch   = mpatches.Patch(color='red',       label=f'WBC-associated (<{WBC_ASSOC_PX}px)')
blue_patch  = mpatches.Patch(color='steelblue', label='Background cluster')
ax.legend(handles=[red_patch, blue_patch], fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "dbscan_cluster_wbc_proximity.png", dpi=150, bbox_inches='tight')

# =============================================================================
# PLOT 2 — SAMPLE IMAGE OVERLAYS (first 6 images with >= 2 clusters)
# =============================================================================
images_with_clusters = df_clusters[df_clusters['n_fp_in_cluster'] >= 2]['image_id'].unique()[:6]

if len(images_with_clusters) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('DBSCAN FP Clusters on Sample Images', fontsize=12)
    axes = axes.ravel()

    cmap = plt.cm.get_cmap('tab10')

    for ax_idx, img_id in enumerate(images_with_clusters):
        ax = axes[ax_idx]
        fp_img   = fps[fps['image_id'] == img_id][['x_ctr', 'y_ctr']].values
        wbc_img  = df_wbc[df_wbc['image_id'] == img_id][['x', 'y']].values
        c_img    = df_clusters[df_clusters['image_id'] == img_id]

        if len(fp_img) >= MIN_SAMPLES:
            labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(fp_img).labels_
        else:
            labels = np.full(len(fp_img), -1)

        noise_mask = labels == -1
        ax.scatter(fp_img[noise_mask, 0], fp_img[noise_mask, 1],
                   c='grey', s=20, alpha=0.4, label='Noise FPs', marker='x')

        n_c = labels.max() + 1
        for cid in range(n_c):
            mask = labels == cid
            color = cmap(cid % 10)
            ax.scatter(fp_img[mask, 0], fp_img[mask, 1],
                       c=[color], s=40, alpha=0.8)
            ctr = fp_img[mask].mean(axis=0)
            ax.plot(*ctr, '*', color=color, markersize=15)

        if len(wbc_img) > 0:
            ax.scatter(wbc_img[:, 0], wbc_img[:, 1],
                       c='blue', s=80, alpha=0.6, marker='^', label='GT WBC')

        ax.set_xlim([0, IMG_W]); ax.set_ylim([IMG_H, 0])
        ax.set_title(f"{img_id[:25]}  ({n_c} clusters)", fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_dir / "dbscan_sample_image_overlay.png", dpi=130, bbox_inches='tight')
    print(f"\nSample image overlay saved to: {out_dir}/dbscan_sample_image_overlay.png")

print("\nPhase 4 — DBSCAN cluster analysis complete.")
