# =============================================================================
# Phase 6 | Temperature Scaling Calibration
#
# Applies temperature scaling to improve the calibration of both YOLO models.
# Temperature scaling is the simplest and most effective post-hoc calibration:
#
#   calibrated_conf = sigmoid(logit(raw_conf) / T)
#
# where T > 1 reduces overconfidence and T < 1 increases confidence.
# T is learned by minimising negative log-likelihood (NLL) on a calibration
# split (held-out subset of the validation set).
#
# Why temperature scaling works for YOLO:
#   YOLO confidence scores are known to be overconfident — this is documented
#   by Guo et al. (2017, ICML) and confirmed for YOLO-World with T > 1
#   reducing ECE from 6.78% to 2.31% (MDPI Applied Sciences 2025).
#
# This script:
#   1. Splits predictions into calibration (50%) and evaluation (50%) sets
#   2. Learns optimal T on the calibration split via scipy.optimize
#   3. Applies T to the evaluation split and reports pre/post ECE/MCE
#   4. Saves calibrated predictions with an added 'conf_calibrated' column
#
# Prerequisite: Phase 3 extract_predictions_and_label_tp_fp.py
# Prerequisite: Phase 6 ece_mce_reliability_diagrams.py (for baseline ECE)
#
# Outputs (saved to outputs/analysis_results/phase_6/):
#   temperature_scaling_results.csv
#   calibrated_predictions_model_a.csv
#   calibrated_predictions_model_b.csv
#   reliability_diagrams_post_temperature_scaling.png
# =============================================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit as sigmoid   # sigmoid(x) = 1/(1+exp(-x))
from scipy.special import logit

# =============================================================================
# PATHS
# =============================================================================
project_root = Path("/home/sadeniji/scratch/Practicum_Main_Project")
phase3_dir   = project_root / "outputs/analysis_results/phase_3"
out_dir      = project_root / "outputs/analysis_results/phase_6"
out_dir.mkdir(parents=True, exist_ok=True)

pred_csv_a = phase3_dir / "predictions_tp_fp_labeled_model_a.csv"
pred_csv_b = phase3_dir / "predictions_tp_fp_labeled_model_b.csv"

N_BINS = 15
SEED   = 42

# =============================================================================
# HELPERS
# =============================================================================
def nll_loss(T, confs, is_tp):
    calibrated = sigmoid(logit(np.clip(confs, 1e-6, 1-1e-6)) / T)
    eps = 1e-9
    nll = -(is_tp * np.log(calibrated + eps) +
            (1 - is_tp) * np.log(1 - calibrated + eps)).mean()
    return nll


def find_temperature(confs, is_tp):
    result = minimize_scalar(
        nll_loss, args=(confs, is_tp),
        bounds=(0.1, 10.0), method='bounded',
        options={'xatol': 1e-6}
    )
    return result.x


def apply_temperature(confs, T):
    return sigmoid(logit(np.clip(confs, 1e-6, 1-1e-6)) / T)


def compute_ece(confs, is_tp, n_bins=N_BINS):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (confs >= lo) & (confs < (hi if i < n_bins-1 else hi + 1e-9))
        if mask.sum() == 0:
            continue
        ece += abs(is_tp[mask].mean() - confs[mask].mean()) * mask.sum() / len(confs)
    return ece


def reliability_bins(confs, is_tp, n_bins=N_BINS):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mids, accs, confs_mean = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (confs >= lo) & (confs < (hi if i < n_bins-1 else hi + 1e-9))
        if mask.sum() == 0:
            continue
        mids.append((lo + hi) / 2)
        accs.append(is_tp[mask].mean())
        confs_mean.append(confs[mask].mean())
    return np.array(mids), np.array(accs), np.array(confs_mean)


# =============================================================================
# CALIBRATE BOTH MODELS
# =============================================================================
summary_rows = []
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('Reliability Diagrams: Before and After Temperature Scaling', fontsize=12)

for row_idx, (pred_csv, model_label, ax_before, ax_after) in enumerate([
    (pred_csv_a, 'Model_A', axes[0, 0], axes[0, 1]),
    (pred_csv_b, 'Model_B', axes[1, 0], axes[1, 1]),
]):
    df = pd.read_csv(pred_csv)
    df = df[df['cls'] == 0].copy()   # parasite class only

    confs  = df['conf'].values
    is_tp  = df['is_tp'].values.astype(float)

    # 50/50 split for calibration vs. evaluation
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    cal_idx  = idx[:len(idx)//2]
    eval_idx = idx[len(idx)//2:]

    confs_cal, is_tp_cal   = confs[cal_idx],  is_tp[cal_idx]
    confs_eval, is_tp_eval = confs[eval_idx], is_tp[eval_idx]

    # Learn T
    T = find_temperature(confs_cal, is_tp_cal)
    confs_eval_cal = apply_temperature(confs_eval, T)

    ece_before = compute_ece(confs_eval, is_tp_eval)
    ece_after  = compute_ece(confs_eval_cal, is_tp_eval)
    mce_before = max(abs(b - c) for b, c, _ in zip(*reliability_bins(confs_eval, is_tp_eval)[:2],
                                                     [None]*N_BINS) if b == b)
    mce_after  = max(abs(b - c) for b, c, _ in zip(*reliability_bins(confs_eval_cal, is_tp_eval)[:2],
                                                     [None]*N_BINS) if b == b)

    print(f"\n{model_label}: T = {T:.4f}")
    print(f"  ECE before: {ece_before:.4f}  → after: {ece_after:.4f}  "
          f"(reduction: {(ece_before - ece_after) / max(ece_before, 1e-9) * 100:.1f}%)")

    summary_rows.append({
        'model': model_label,
        'temperature_T': round(T, 4),
        'ece_before': round(ece_before, 4),
        'ece_after': round(ece_after, 4),
        'ece_reduction_pct': round((ece_before - ece_after) / max(ece_before, 1e-9) * 100, 1),
        'cal_set_size': len(cal_idx),
        'eval_set_size': len(eval_idx),
    })

    # Save calibrated full predictions
    df_out = df.copy()
    df_out['conf_calibrated'] = apply_temperature(confs, T)
    df_out['temperature_T']   = T
    save_name = f"calibrated_predictions_{model_label.lower()}.csv"
    df_out.to_csv(out_dir / save_name, index=False)

    # Reliability diagrams
    for ax, c, title in [(ax_before, confs_eval, f'{model_label} — Before (ECE={ece_before:.4f})'),
                          (ax_after,  confs_eval_cal, f'{model_label} — After T={T:.3f} (ECE={ece_after:.4f})')]:
        mids, accs, cmeans = reliability_bins(c, is_tp_eval)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect')
        ax.plot(mids, accs, 'bo-', markersize=5, linewidth=2, label='Observed')
        ax.fill_between(mids, mids, accs, where=accs < mids,
                        alpha=0.3, color='red', label='Overconfident gap')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
        ax.set_title(title, fontsize=9); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

pd.DataFrame(summary_rows).to_csv(out_dir / "temperature_scaling_results.csv", index=False)
plt.tight_layout()
plt.savefig(out_dir / "reliability_diagrams_post_temperature_scaling.png",
            dpi=150, bbox_inches='tight')

print(f"\nTemperature scaling results saved to: {out_dir}")
print("\nPhase 6 — Temperature scaling calibration complete.")
