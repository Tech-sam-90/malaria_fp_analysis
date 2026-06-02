# =============================================================================
# Phase 4 | Mann-Whitney U Test: FP Distance to Nearest WBC vs TP Distance
#
# Tests the core hypothesis of Objective 3:
#   H1: False positive predictions are significantly closer to annotated WBC
#       centers than true positive predictions are.
#
# Method:
#   1. For each FP centroid: find the nearest WBC annotation center (cKDTree)
#   2. For each TP centroid: find the nearest WBC annotation center
#   3. Mann-Whitney U test (one-tailed, alternative='less'):
#      tests if FP distance distribution is stochastically smaller than TP
#   4. Report mean, median, std distances and the U statistic / p-value
#
# This analysis is run separately for Model A (using GT WBC positions from
# Model B labels) and for Model B (using its own WBC predictions as a proxy).
#
# Prerequisite: Phase 3 extract_predictions_and_label_tp_fp.py
#
# Outputs (saved to outputs/analysis_results/phase_4/):
#   mannwhitney_results.csv
#   distance_distribution_fp_vs_tp.png
# =============================================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

# =============================================================================
# PATHS
# =============================================================================
project_root  = Path("/home/sadeniji/scratch/Practicum_Main_Project")
phase3_dir    = project_root / "outputs/analysis_results/phase_3"
out_dir       = project_root / "outputs/analysis_results/phase_4"
out_dir.mkdir(parents=True, exist_ok=True)

pred_csv_a    = phase3_dir / "predictions_tp_fp_labeled_model_a.csv"
pred_csv_b    = phase3_dir / "predictions_tp_fp_labeled_model_b.csv"
test_labels_b = project_root / "model_b/test/labels"

for name, p in [("Model A CSV", pred_csv_a), ("Model B CSV", pred_csv_b)]:
    if not p.exists():
        sys.exit(f"ERROR: {name} not found. Run Phase 3 first.")

IMG_W, IMG_H = 1024, 1024
CONF_THRESH  = 0.25   # filter predictions at clinical operating threshold

# =============================================================================
# LOAD WBC GROUND-TRUTH CENTERS (from Model B annotations)
# =============================================================================
def load_wbc_gt(labels_dir):
    rows = []
    for lf in sorted(Path(labels_dir).glob("*.txt")):
        for line in lf.read_text().strip().split('\n'):
            parts = line.strip().split()
            if len(parts) < 5 or int(parts[0]) != 1:
                continue
            rows.append({
                'image_id': lf.stem,
                'x': float(parts[1]) * IMG_W,
                'y': float(parts[2]) * IMG_H,
            })
    return pd.DataFrame(rows)


print("Loading WBC ground-truth centers...")
df_wbc = load_wbc_gt(test_labels_b)
print(f"  {len(df_wbc)} WBC annotations across {df_wbc['image_id'].nunique()} images")

# =============================================================================
# COMPUTE DISTANCES PER IMAGE
# =============================================================================
def compute_nearest_wbc_distances(preds_df, wbc_df, label):
    results = []
    n_no_wbc = 0
    for img_id, group in preds_df.groupby('image_id'):
        wbc_img = wbc_df[wbc_df['image_id'] == img_id]
        if len(wbc_img) == 0:
            n_no_wbc += 1
            continue
        tree = cKDTree(wbc_img[['x', 'y']].values)
        coords = group[['x_ctr', 'y_ctr']].values
        dists, _ = tree.query(coords)
        for i, (_, row) in enumerate(group.iterrows()):
            results.append({
                'image_id': img_id,
                'x_ctr': row['x_ctr'],
                'y_ctr': row['y_ctr'],
                'conf': row['conf'],
                'is_tp': row['is_tp'],
                'is_fp': row['is_fp'],
                'dist_to_wbc': dists[i],
                'model': label,
            })
    if n_no_wbc > 0:
        print(f"  [{label}] {n_no_wbc} images had no WBC annotation — skipped")
    return pd.DataFrame(results)


print("\nLoading labeled predictions...")
df_a = pd.read_csv(pred_csv_a)
df_b = pd.read_csv(pred_csv_b)

# Filter at operating threshold
df_a = df_a[df_a['conf'] >= CONF_THRESH].copy()
df_b = df_b[df_b['conf'] >= CONF_THRESH].copy()

print("Computing nearest-WBC distances...")
dist_a = compute_nearest_wbc_distances(df_a, df_wbc, 'Model_A')
dist_b = compute_nearest_wbc_distances(df_b, df_wbc, 'Model_B')

# =============================================================================
# MANN-WHITNEY U TEST
# =============================================================================
def run_mannwhitney(dist_df, model_label):
    fp_dists = dist_df[dist_df['is_fp']]['dist_to_wbc'].dropna().values
    tp_dists = dist_df[dist_df['is_tp']]['dist_to_wbc'].dropna().values

    if len(fp_dists) == 0 or len(tp_dists) == 0:
        print(f"  [{model_label}] Insufficient data for Mann-Whitney test.")
        return {}

    stat, p_val = mannwhitneyu(fp_dists, tp_dists, alternative='less')

    # Effect size: rank-biserial correlation
    n1, n2 = len(fp_dists), len(tp_dists)
    r_rb = 1 - (2 * stat) / (n1 * n2)

    result = {
        'model': model_label,
        'n_fp': n1, 'n_tp': n2,
        'fp_mean_dist': round(fp_dists.mean(), 2),
        'fp_median_dist': round(np.median(fp_dists), 2),
        'fp_std_dist': round(fp_dists.std(), 2),
        'tp_mean_dist': round(tp_dists.mean(), 2),
        'tp_median_dist': round(np.median(tp_dists), 2),
        'tp_std_dist': round(tp_dists.std(), 2),
        'U_statistic': round(stat, 2),
        'p_value': round(p_val, 6),
        'rank_biserial_r': round(r_rb, 4),
        'significant_p05': p_val < 0.05,
    }

    print(f"\n{'='*55}")
    print(f"{model_label} — Mann-Whitney U Test")
    print(f"{'='*55}")
    print(f"  FP dist to WBC: mean={fp_dists.mean():.1f}px  median={np.median(fp_dists):.1f}px")
    print(f"  TP dist to WBC: mean={tp_dists.mean():.1f}px  median={np.median(tp_dists):.1f}px")
    print(f"  U = {stat:.1f}  p = {p_val:.6f}  r_rb = {r_rb:.4f}")
    if p_val < 0.05:
        print(f"  *** SIGNIFICANT: FPs are closer to WBCs than TPs (p < 0.05) ***")
    else:
        print(f"  NOT significant at p=0.05")

    return result


results = [run_mannwhitney(dist_a, 'Model_A'), run_mannwhitney(dist_b, 'Model_B')]
df_results = pd.DataFrame([r for r in results if r])
df_results.to_csv(out_dir / "mannwhitney_results.csv", index=False)

# =============================================================================
# PLOT DISTANCE DISTRIBUTIONS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Distance to Nearest WBC: FP vs TP', fontsize=13)

for ax, dist_df, label in [(axes[0], dist_a, 'Model A'), (axes[1], dist_b, 'Model B')]:
    fp_dists = dist_df[dist_df['is_fp']]['dist_to_wbc'].dropna()
    tp_dists = dist_df[dist_df['is_tp']]['dist_to_wbc'].dropna()

    bins = np.linspace(0, min(fp_dists.max(), tp_dists.max(), 500), 50)
    ax.hist(fp_dists.clip(upper=500), bins=bins, alpha=0.6,
            color='red', label=f'FP (n={len(fp_dists)})', density=True)
    ax.hist(tp_dists.clip(upper=500), bins=bins, alpha=0.6,
            color='green', label=f'TP (n={len(tp_dists)})', density=True)
    ax.axvline(fp_dists.mean(), color='darkred',   linestyle='--', linewidth=1.5,
               label=f'FP mean={fp_dists.mean():.0f}px')
    ax.axvline(tp_dists.mean(), color='darkgreen', linestyle='--', linewidth=1.5,
               label=f'TP mean={tp_dists.mean():.0f}px')
    ax.set_xlabel('Distance to Nearest WBC (pixels)')
    ax.set_ylabel('Density')
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "distance_distribution_fp_vs_tp.png", dpi=150, bbox_inches='tight')
print(f"\nDistance distribution plot saved to: {out_dir}/distance_distribution_fp_vs_tp.png")

print("\nPhase 4 — Mann-Whitney distance analysis complete.")
