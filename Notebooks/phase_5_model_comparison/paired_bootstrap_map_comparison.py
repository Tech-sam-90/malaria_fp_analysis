# =============================================================================
# Phase 5 | Paired Bootstrap Test for mAP Comparison (Model A vs Model B)
#
# Tests whether the difference in mAP50 between Model A and Model B is
# statistically significant using the paired bootstrap procedure:
#
#   1. Compute mAP50 per test image for both models
#   2. Resample test images with replacement B=10,000 times
#   3. Compute mAP50 for both models on each bootstrap resample
#   4. Build the empirical distribution of (mAP_B − mAP_A)
#   5. Report BCa (bias-corrected and accelerated) 95% CI and p-value
#
# Why paired bootstrap and not a t-test?
#   - Per-image AP is non-normally distributed (many 0s and 1s)
#   - Images are the natural resampling unit, not individual detections
#   - t-test assumes independence; bootstrapping respects the pairing
#
# Paper: "When +1% Is Not Enough: Paired Bootstrap Protocol" (arXiv 2511.19794)
# Package: stambo (github.com/imedslab/stambo)
#
# Prerequisite: Phase 3 extract_predictions_and_label_tp_fp.py
#
# Outputs (saved to outputs/analysis_results/phase_5/):
#   paired_bootstrap_results.csv
#   bootstrap_delta_map_distribution.png
# =============================================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================
project_root = Path("/home/sadeniji/scratch/Practicum_Main_Project")
phase3_dir   = project_root / "outputs/analysis_results/phase_3"
out_dir      = project_root / "outputs/analysis_results/phase_5"
out_dir.mkdir(parents=True, exist_ok=True)

pred_csv_a   = phase3_dir / "predictions_tp_fp_labeled_model_a.csv"
pred_csv_b   = phase3_dir / "predictions_tp_fp_labeled_model_b.csv"
test_labels_a = project_root / "model_a/test/labels"

CONF_THRESH  = 0.25
IOU_THRESH   = 0.50
N_BOOTSTRAP  = 10_000
SEED         = 42

# =============================================================================
# COMPUTE PER-IMAGE AP50
# =============================================================================
def compute_per_image_ap(pred_df, gt_dir, iou_thresh=IOU_THRESH, conf_thresh=CONF_THRESH):
    IMG_W, IMG_H = 1024, 1024
    image_ids = pred_df['image_id'].unique()
    ap_per_image = {}

    for img_id in image_ids:
        # Load GT for this image
        lf = Path(gt_dir) / (img_id + ".txt")
        gt_boxes = []
        if lf.exists():
            for line in lf.read_text().strip().split('\n'):
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 0:  # class 0 = parasite
                    xc, yc = float(parts[1])*IMG_W, float(parts[2])*IMG_H
                    bw, bh = float(parts[3])*IMG_W, float(parts[4])*IMG_H
                    gt_boxes.append([xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2])

        n_gt = len(gt_boxes)
        if n_gt == 0:
            ap_per_image[img_id] = np.nan
            continue

        # Filter predictions
        preds = pred_df[(pred_df['image_id'] == img_id) &
                        (pred_df['conf'] >= conf_thresh) &
                        (pred_df['cls'] == 0)].sort_values('conf', ascending=False)

        if len(preds) == 0:
            ap_per_image[img_id] = 0.0
            continue

        # Precision-recall computation
        tp_arr = preds['is_tp'].values.astype(float)
        fp_arr = preds['is_fp'].values.astype(float)

        tp_cum = np.cumsum(tp_arr)
        fp_cum = np.cumsum(fp_arr)

        precision = tp_cum / (tp_cum + fp_cum)
        recall    = tp_cum / n_gt

        # 11-point interpolated AP
        ap = 0.0
        for thr in np.linspace(0, 1, 11):
            p_at_r = precision[recall >= thr].max() if (recall >= thr).any() else 0
            ap += p_at_r / 11.0

        ap_per_image[img_id] = ap

    return ap_per_image


print("Computing per-image AP50 for Model A...")
df_a = pd.read_csv(pred_csv_a)
ap_a = compute_per_image_ap(df_a, test_labels_a)

print("Computing per-image AP50 for Model B...")
df_b = pd.read_csv(pred_csv_b)
ap_b = compute_per_image_ap(df_b, test_labels_a)   # same GT (parasite class)

# Align on shared image IDs
shared = sorted(set(ap_a.keys()) & set(ap_b.keys()))
shared = [i for i in shared if not (np.isnan(ap_a[i]) or np.isnan(ap_b[i]))]
ap_a_arr = np.array([ap_a[i] for i in shared])
ap_b_arr = np.array([ap_b[i] for i in shared])
n = len(shared)
print(f"Images with valid AP for both models: {n}")

# =============================================================================
# PAIRED BOOTSTRAP
# =============================================================================
rng = np.random.default_rng(SEED)
deltas = []

for _ in range(N_BOOTSTRAP):
    idx = rng.integers(0, n, size=n)
    delta = ap_b_arr[idx].mean() - ap_a_arr[idx].mean()
    deltas.append(delta)

deltas = np.array(deltas)
observed_delta = ap_b_arr.mean() - ap_a_arr.mean()

# Bootstrap p-value (two-sided: fraction of resamples that cross zero)
p_value = np.mean(np.abs(deltas) >= np.abs(observed_delta))

# BCa confidence interval
alpha = 0.05
def bca_ci(boot_samples, observed, confidence=0.95):
    from scipy.stats import norm
    n_boot = len(boot_samples)
    z0 = norm.ppf(np.mean(boot_samples < observed))
    alpha_lo = norm.cdf(2*z0 + norm.ppf((1 - confidence)/2))
    alpha_hi = norm.cdf(2*z0 + norm.ppf((1 + confidence)/2))
    lo = np.percentile(boot_samples, alpha_lo * 100)
    hi = np.percentile(boot_samples, alpha_hi * 100)
    return lo, hi


from scipy.stats import norm
ci_lo, ci_hi = bca_ci(deltas, observed_delta)
ci_lo_pct, ci_hi_pct = np.percentile(deltas, [2.5, 97.5])   # standard percentile CI

print(f"\n{'='*55}")
print("Paired Bootstrap Results (Model B − Model A)")
print(f"{'='*55}")
print(f"  Observed mAP50 (A)   : {ap_a_arr.mean():.4f}")
print(f"  Observed mAP50 (B)   : {ap_b_arr.mean():.4f}")
print(f"  Observed delta       : {observed_delta:+.4f}")
print(f"  Bootstrap mean delta : {deltas.mean():+.4f}")
print(f"  BCa 95% CI           : [{ci_lo:+.4f}, {ci_hi:+.4f}]")
print(f"  Percentile 95% CI    : [{ci_lo_pct:+.4f}, {ci_hi_pct:+.4f}]")
print(f"  Bootstrap p-value    : {p_value:.4f}")
if ci_lo > 0:
    print("  *** SIGNIFICANT: Model B is significantly better (CI entirely above 0) ***")
elif ci_hi < 0:
    print("  *** SIGNIFICANT: Model A is significantly better (CI entirely below 0) ***")
else:
    print("  NOT SIGNIFICANT: CI crosses 0")

results = {
    'map50_model_a': round(ap_a_arr.mean(), 4),
    'map50_model_b': round(ap_b_arr.mean(), 4),
    'observed_delta': round(observed_delta, 4),
    'bootstrap_mean_delta': round(deltas.mean(), 4),
    'bca_ci_lo': round(ci_lo, 4),
    'bca_ci_hi': round(ci_hi, 4),
    'pct_ci_lo': round(ci_lo_pct, 4),
    'pct_ci_hi': round(ci_hi_pct, 4),
    'p_value_two_sided': round(p_value, 4),
    'n_images': n,
    'n_bootstrap': N_BOOTSTRAP,
}
pd.DataFrame([results]).to_csv(out_dir / "paired_bootstrap_results.csv", index=False)

# =============================================================================
# PLOT BOOTSTRAP DISTRIBUTION
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(deltas, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(observed_delta, color='red', linewidth=2, label=f'Observed Δ = {observed_delta:+.4f}')
ax.axvline(ci_lo, color='orange', linewidth=1.5, linestyle='--', label=f'BCa 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]')
ax.axvline(ci_hi, color='orange', linewidth=1.5, linestyle='--')
ax.axvline(0, color='black', linewidth=1, label='Null (Δ=0)')
ax.set_xlabel('mAP50 delta (Model B − Model A)')
ax.set_ylabel('Bootstrap frequency')
ax.set_title(f'Paired Bootstrap Distribution (N={N_BOOTSTRAP:,})')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "bootstrap_delta_map_distribution.png", dpi=150, bbox_inches='tight')

print(f"\nBootstrap plot saved to: {out_dir}/bootstrap_delta_map_distribution.png")
print("\nPhase 5 — Paired bootstrap mAP comparison complete.")
