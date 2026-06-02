# =============================================================================
# Phase 6 | ECE, MCE and Reliability Diagrams (Pre-Calibration Diagnostics)
#
# Measures how well-calibrated the confidence scores of both YOLO models are.
# A perfectly calibrated model: if it says conf=0.7, then 70% of those
# detections should actually be true positives.
#
# Metrics:
#   ECE (Expected Calibration Error) = Σ |bin_accuracy − bin_confidence| × (n/N)
#   MCE (Maximum Calibration Error)  = max |bin_accuracy − bin_confidence|
#
# A reliability diagram plots bin_accuracy (y) vs bin_confidence (x).
# A perfectly calibrated model lies on the diagonal y = x.
# YOLO models tend to be overconfident: points fall below the diagonal.
#
# This analysis establishes the calibration baseline that Phase 6's
# temperature scaling script will then improve upon.
#
# Prerequisite: Phase 3 extract_predictions_and_label_tp_fp.py
#
# Outputs (saved to outputs/analysis_results/phase_6/):
#   calibration_metrics_pre_calibration.csv
#   reliability_diagrams_pre_calibration.png
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
out_dir      = project_root / "outputs/analysis_results/phase_6"
out_dir.mkdir(parents=True, exist_ok=True)

pred_csv_a = phase3_dir / "predictions_tp_fp_labeled_model_a.csv"
pred_csv_b = phase3_dir / "predictions_tp_fp_labeled_model_b.csv"

N_BINS = 15   # confidence bins for ECE/reliability diagram

# =============================================================================
# ECE COMPUTATION
# =============================================================================
def compute_ece_mce(confidences, is_tp, n_bins=N_BINS):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    bin_data = []
    n_total = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)

        n_bin = mask.sum()
        if n_bin == 0:
            bin_data.append({'bin_lo': lo, 'bin_hi': hi, 'n': 0,
                             'mean_conf': np.nan, 'accuracy': np.nan, 'gap': 0})
            continue

        mean_conf = confidences[mask].mean()
        accuracy  = is_tp[mask].mean()   # fraction of bin that are TPs
        gap = abs(accuracy - mean_conf)

        ece += gap * (n_bin / n_total)
        mce = max(mce, gap)
        bin_data.append({'bin_lo': lo, 'bin_hi': hi, 'n': n_bin,
                         'mean_conf': mean_conf, 'accuracy': accuracy, 'gap': gap})

    return ece, mce, pd.DataFrame(bin_data)


def run_calibration_diagnostics(df_preds, model_label, parasite_cls=0):
    # Filter to parasite predictions only
    preds = df_preds[df_preds['cls'] == parasite_cls].copy()
    confs = preds['conf'].values
    tps   = preds['is_tp'].values.astype(float)

    ece, mce, bins_df = compute_ece_mce(confs, tps)

    result = {
        'model': model_label,
        'n_predictions': len(preds),
        'n_tp': int(tps.sum()),
        'n_fp': int((~preds['is_tp']).sum()),
        'mean_confidence': round(confs.mean(), 4),
        'mean_accuracy': round(tps.mean(), 4),
        'ECE': round(ece, 4),
        'MCE': round(mce, 4),
        'overconfident': confs.mean() > tps.mean(),
    }

    print(f"\n{'='*50}")
    print(f"{model_label} — Calibration Diagnostics")
    print(f"{'='*50}")
    print(f"  N predictions : {result['n_predictions']}")
    print(f"  Mean conf     : {result['mean_confidence']:.4f}")
    print(f"  Mean accuracy : {result['mean_accuracy']:.4f}  "
          f"({'overconfident' if result['overconfident'] else 'underconfident'})")
    print(f"  ECE           : {result['ECE']:.4f}")
    print(f"  MCE           : {result['MCE']:.4f}")

    return result, bins_df


print("Loading predictions...")
df_a = pd.read_csv(pred_csv_a)
df_b = pd.read_csv(pred_csv_b)

res_a, bins_a = run_calibration_diagnostics(df_a, 'Model_A')
res_b, bins_b = run_calibration_diagnostics(df_b, 'Model_B')

pd.DataFrame([res_a, res_b]).to_csv(out_dir / "calibration_metrics_pre_calibration.csv", index=False)

# =============================================================================
# RELIABILITY DIAGRAMS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Reliability Diagrams (Pre-Calibration)', fontsize=13)

for ax, bins_df, res, label in [(axes[0], bins_a, res_a, 'Model A'),
                                  (axes[1], bins_b, res_b, 'Model B')]:
    valid = bins_df[bins_df['n'] > 0].copy()

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')

    # Gap shading (overconfidence = confidence above accuracy)
    ax.bar(valid['bin_lo'], valid['accuracy'], width=1/N_BINS,
           align='edge', alpha=0.7, color='steelblue', label='Accuracy')
    ax.bar(valid['bin_lo'], valid['mean_conf'] - valid['accuracy'],
           width=1/N_BINS, align='edge', bottom=valid['accuracy'],
           alpha=0.4, color='red', label='Gap (overconf)')

    # Line connecting accuracy midpoints
    mid_confs = (valid['bin_lo'] + valid['bin_hi']) / 2
    ax.plot(mid_confs, valid['accuracy'], 'bo-', markersize=5, linewidth=1.5)

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_xlabel('Mean Confidence'); ax.set_ylabel('Fraction of TPs (Accuracy)')
    ax.set_title(f"{label}\nECE={res['ECE']:.4f}  MCE={res['MCE']:.4f}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "reliability_diagrams_pre_calibration.png", dpi=150, bbox_inches='tight')
print(f"\nReliability diagrams saved to: {out_dir}/reliability_diagrams_pre_calibration.png")

print("\nPhase 6 — Pre-calibration diagnostics complete.")
