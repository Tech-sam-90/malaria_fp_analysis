# =============================================================================
# Phase 6 | Multivariate Position-Aware Calibration
#
# Extends temperature scaling by conditioning the calibration on the spatial
# position and size of each detection box. The intuition is that YOLO may
# be systematically less calibrated in specific image regions — particularly
# near WBC-dense areas where parasite-like blobs are common.
#
# Method: Küppers et al. (2020) CVPR Workshop (arXiv:2004.13546)
#   Train a logistic regression calibrator that takes:
#     input  = [raw_confidence, x_center/W, y_center/H, box_w/W, box_h/H]
#     output = P(TP)  (calibrated probability)
#
# This produces a Position-ECE (pECE) that can be stratified by image region,
# directly testing whether the model is less calibrated near WBC-dense zones.
#
# Prerequisite: Phase 3 extract_predictions_and_label_tp_fp.py
#
# Outputs (saved to outputs/analysis_results/phase_6/):
#   multivariate_calibration_results.csv
#   position_ece_heatmap.png   (shows where model is least calibrated)
#   calibration_comparison_global_vs_position.png
# =============================================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

# =============================================================================
# PATHS
# =============================================================================
project_root = Path("/home/sadeniji/scratch/Practicum_Main_Project")
phase3_dir   = project_root / "outputs/analysis_results/phase_3"
out_dir      = project_root / "outputs/analysis_results/phase_6"
out_dir.mkdir(parents=True, exist_ok=True)

pred_csv_b = phase3_dir / "predictions_tp_fp_labeled_model_b.csv"

if not pred_csv_b.exists():
    sys.exit("ERROR: Run Phase 3 first.")

IMG_W, IMG_H = 1024, 1024
SEED = 42
GRID_N = 8  # 8×8 = 64 tiles for position-ECE heatmap

# =============================================================================
# LOAD AND PREPARE FEATURES
# =============================================================================
df = pd.read_csv(pred_csv_b)
df = df[df['cls'] == 0].copy()   # parasite class only

# Feature matrix: [raw_conf, x_norm, y_norm, w_norm, h_norm]
df['x_norm'] = df['x_ctr'] / IMG_W
df['y_norm'] = df['y_ctr'] / IMG_H
df['w_norm'] = df['w'] / IMG_W
df['h_norm'] = df['h'] / IMG_H

X = df[['conf', 'x_norm', 'y_norm', 'w_norm', 'h_norm']].values
y = df['is_tp'].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                      random_state=SEED, stratify=y)
print(f"Train: {len(X_train)}  Test: {len(X_test)}")

# =============================================================================
# CALIBRATION METHODS
# =============================================================================
# 1. Temperature Scaling (global, position-independent baseline)
from scipy.optimize import minimize_scalar
from scipy.special import expit as sigmoid, logit

def temp_nll(T, confs, is_tp):
    cal = sigmoid(logit(np.clip(confs, 1e-7, 1-1e-7)) / T)
    return -(is_tp * np.log(cal + 1e-9) + (1 - is_tp) * np.log(1 - cal + 1e-9)).mean()

T_opt = minimize_scalar(temp_nll, args=(X_train[:, 0], y_train),
                         bounds=(0.1, 10), method='bounded').x
confs_ts = sigmoid(logit(np.clip(X_test[:, 0], 1e-7, 1-1e-7)) / T_opt)

# 2. Multivariate Logistic Regression (position-aware)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr_cal = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
lr_cal.fit(X_train_s, y_train)
confs_mv = lr_cal.predict_proba(X_test_s)[:, 1]

# 3. Isotonic Regression (on raw confidence alone)
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(X_train[:, 0], y_train)
confs_iso = iso.predict(X_test[:, 0])

# =============================================================================
# COMPUTE ECE FOR EACH METHOD
# =============================================================================
def ece(confs, labels, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        m = (confs >= lo) & (confs < (hi if i < n_bins-1 else hi+1e-9))
        if m.sum() == 0:
            continue
        e += abs(labels[m].mean() - confs[m].mean()) * m.sum() / len(confs)
    return e

ece_raw = ece(X_test[:, 0], y_test)
ece_ts  = ece(confs_ts,     y_test)
ece_mv  = ece(confs_mv,     y_test)
ece_iso = ece(confs_iso,    y_test)

print(f"\nECE Summary:")
print(f"  Raw YOLO conf     : {ece_raw:.4f}")
print(f"  Temperature (T={T_opt:.3f}) : {ece_ts:.4f}")
print(f"  Multivariate LR   : {ece_mv:.4f}")
print(f"  Isotonic Reg.     : {ece_iso:.4f}")

summary = pd.DataFrame([{
    'model': 'Model_B_parasite',
    'ece_raw': round(ece_raw, 4),
    'temperature_T': round(T_opt, 4),
    'ece_temperature_scaling': round(ece_ts, 4),
    'ece_multivariate_lr': round(ece_mv, 4),
    'ece_isotonic': round(ece_iso, 4),
    'lr_coef_x_norm': round(lr_cal.coef_[0][1], 4),
    'lr_coef_y_norm': round(lr_cal.coef_[0][2], 4),
    'lr_coef_w_norm': round(lr_cal.coef_[0][3], 4),
    'lr_coef_h_norm': round(lr_cal.coef_[0][4], 4),
}])
summary.to_csv(out_dir / "multivariate_calibration_results.csv", index=False)

# =============================================================================
# POSITION-ECE HEATMAP (shows where model is least calibrated)
# =============================================================================
x_test_df = pd.DataFrame({
    'x_norm': X_test[:, 1], 'y_norm': X_test[:, 2],
    'conf_raw': X_test[:, 0], 'conf_mv': confs_mv, 'is_tp': y_test
})

tile_ece_raw = np.zeros((GRID_N, GRID_N))
tile_ece_mv  = np.zeros((GRID_N, GRID_N))
tile_count   = np.zeros((GRID_N, GRID_N))

for i in range(GRID_N):
    for j in range(GRID_N):
        x_lo, x_hi = i/GRID_N, (i+1)/GRID_N
        y_lo, y_hi = j/GRID_N, (j+1)/GRID_N
        mask = ((x_test_df['x_norm'] >= x_lo) & (x_test_df['x_norm'] < x_hi) &
                (x_test_df['y_norm'] >= y_lo) & (x_test_df['y_norm'] < y_hi))
        if mask.sum() < 5:
            continue
        tile_ece_raw[j, i] = ece(x_test_df.loc[mask, 'conf_raw'].values,
                                   x_test_df.loc[mask, 'is_tp'].values, n_bins=5)
        tile_ece_mv[j, i]  = ece(x_test_df.loc[mask, 'conf_mv'].values,
                                   x_test_df.loc[mask, 'is_tp'].values, n_bins=5)
        tile_count[j, i]   = mask.sum()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Position-ECE Heatmap: Where Is the Model Least Calibrated?', fontsize=12)

for ax, data, title in [(axes[0], tile_ece_raw, 'Raw YOLO conf'),
                         (axes[1], tile_ece_mv,  'Multivariate LR calibrated'),
                         (axes[2], tile_ece_raw - tile_ece_mv, 'ECE improvement (raw - MV)')]:
    masked = np.ma.masked_where(tile_count == 0, data)
    cmap = 'RdYlGn_r' if 'improvement' not in title else 'RdYlGn'
    im = ax.imshow(masked, cmap=cmap, aspect='equal', origin='lower', vmin=0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('X tile'); ax.set_ylabel('Y tile')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(out_dir / "position_ece_heatmap.png", dpi=150, bbox_inches='tight')

print(f"\nPosition-ECE heatmap saved to: {out_dir}/position_ece_heatmap.png")
print("\nPhase 6 — Multivariate calibration complete.")
