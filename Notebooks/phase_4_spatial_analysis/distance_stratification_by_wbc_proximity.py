# =============================================================================
# Phase 4 | Distance Stratification: FP Rate per WBC-Proximity Band
#
# Divides predictions into 5 distance bands from the nearest annotated WBC:
#   Band 0:  0 –  30 px  (inside or overlapping WBC)
#   Band 1: 30 –  75 px  (immediately adjacent to WBC)
#   Band 2: 75 – 150 px  (near-WBC neighborhood)
#   Band 3: 150 – 300 px (moderate distance)
#   Band 4: > 300 px     (far from any WBC — background / artifact region)
#
# For each band, computes: FP count, TP count, FP rate, and proportion of
# all FPs that fall in that band. A monotonic decrease in FP rate with
# increasing distance would confirm the WBC-proximity hypothesis.
#
# This analysis feeds directly into Phase 5's stratified McNemar test.
#
# Prerequisite: Phase 3 extract_predictions_and_label_tp_fp.py
#
# Outputs (saved to outputs/analysis_results/phase_4/):
#   distance_stratification_results.csv
#   fp_rate_by_distance_band.png
# =============================================================================

import sys
from pathlib import Path

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
out_dir       = project_root / "outputs/analysis_results/phase_4"
out_dir.mkdir(parents=True, exist_ok=True)

pred_csv_a    = phase3_dir / "predictions_tp_fp_labeled_model_a.csv"
pred_csv_b    = phase3_dir / "predictions_tp_fp_labeled_model_b.csv"
test_labels_b = project_root / "model_b/test/labels"

IMG_W, IMG_H = 1024, 1024
CONF_THRESH  = 0.25
BANDS = [(0, 30, '0–30 px'), (30, 75, '30–75 px'),
         (75, 150, '75–150 px'), (150, 300, '150–300 px'),
         (300, np.inf, '>300 px')]

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


def add_wbc_distance(df, wbc_df):
    dists = []
    for img_id, group in df.groupby('image_id'):
        wbc_img = wbc_df[wbc_df['image_id'] == img_id]
        if len(wbc_img) == 0:
            dists.extend([np.nan] * len(group))
        else:
            tree = cKDTree(wbc_img[['x', 'y']].values)
            d, _ = tree.query(group[['x_ctr', 'y_ctr']].values)
            dists.extend(d.tolist())
    df = df.copy()
    df['dist_to_wbc'] = dists
    return df


df_wbc = load_wbc_gt(test_labels_b)
print(f"WBC GT centers: {len(df_wbc)} total")

df_a = pd.read_csv(pred_csv_a)
df_b = pd.read_csv(pred_csv_b)
df_a = df_a[df_a['conf'] >= CONF_THRESH].copy()
df_b = df_b[df_b['conf'] >= CONF_THRESH].copy()

print("Adding WBC distances...")
df_a = add_wbc_distance(df_a, df_wbc)
df_b = add_wbc_distance(df_b, df_wbc)

# =============================================================================
# STRATIFY BY DISTANCE BAND
# =============================================================================
def stratify(df, model_label):
    rows = []
    valid = df[df['dist_to_wbc'].notna()].copy()
    total_fp = valid['is_fp'].sum()

    for lo, hi, band_label in BANDS:
        mask = (valid['dist_to_wbc'] >= lo) & (valid['dist_to_wbc'] < hi)
        band = valid[mask]
        n_fp = band['is_fp'].sum()
        n_tp = band['is_tp'].sum()
        n_total = len(band)
        fp_rate = n_fp / max(n_total, 1)
        pct_of_all_fp = n_fp / max(total_fp, 1) * 100
        rows.append({
            'model': model_label,
            'band': band_label,
            'lo_px': lo,
            'hi_px': hi if hi != np.inf else 9999,
            'n_total': int(n_total),
            'n_tp': int(n_tp),
            'n_fp': int(n_fp),
            'fp_rate': round(fp_rate, 4),
            'pct_of_all_fp': round(pct_of_all_fp, 2),
        })
    return pd.DataFrame(rows)


strat_a = stratify(df_a, 'Model_A')
strat_b = stratify(df_b, 'Model_B')
df_strat = pd.concat([strat_a, strat_b], ignore_index=True)
df_strat.to_csv(out_dir / "distance_stratification_results.csv", index=False)

print("\n--- Distance Stratification: Model A ---")
print(strat_a[['band', 'n_total', 'n_fp', 'fp_rate', 'pct_of_all_fp']].to_string(index=False))
print("\n--- Distance Stratification: Model B ---")
print(strat_b[['band', 'n_total', 'n_fp', 'fp_rate', 'pct_of_all_fp']].to_string(index=False))

# =============================================================================
# PLOT
# =============================================================================
band_labels = [b[2] for b in BANDS]
fp_rates_a  = strat_a['fp_rate'].tolist()
fp_rates_b  = strat_b['fp_rate'].tolist()

x = np.arange(len(band_labels))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: FP rate per band
ax = axes[0]
ax.bar(x - width/2, fp_rates_a, width, label='Model A', color='steelblue', alpha=0.85)
ax.bar(x + width/2, fp_rates_b, width, label='Model B', color='darkorange', alpha=0.85)
ax.set_title('FP Rate by WBC-Proximity Band')
ax.set_ylabel('FP Rate (FP / Total predictions in band)')
ax.set_xticks(x); ax.set_xticklabels(band_labels, fontsize=9)
ax.legend(); ax.grid(True, alpha=0.3, axis='y')

# Plot 2: % of all FPs that fall in each band
pct_a = strat_a['pct_of_all_fp'].tolist()
pct_b = strat_b['pct_of_all_fp'].tolist()
ax = axes[1]
ax.bar(x - width/2, pct_a, width, label='Model A', color='steelblue', alpha=0.85)
ax.bar(x + width/2, pct_b, width, label='Model B', color='darkorange', alpha=0.85)
ax.set_title('% of All FPs per WBC-Proximity Band')
ax.set_ylabel('% of All FPs')
ax.set_xticks(x); ax.set_xticklabels(band_labels, fontsize=9)
ax.legend(); ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(out_dir / "fp_rate_by_distance_band.png", dpi=150, bbox_inches='tight')
print(f"\nDistance stratification plot saved to: {out_dir}/fp_rate_by_distance_band.png")

print("\nPhase 4 — Distance stratification complete.")
