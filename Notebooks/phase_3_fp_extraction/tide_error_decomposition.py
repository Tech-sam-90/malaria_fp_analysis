# =============================================================================
# Phase 3 | TIDE-Style Error Decomposition (no tidecv dependency)
#
# Implements TIDE error categories (Bolya et al., ECCV 2020) directly from
# the TP/FP labeled prediction CSV produced by extract_predictions_and_label_tp_fp.py
# and the GT label files — no external tidecv package required.
#
# TIDE error categories (mutually exclusive, applied to each FP/FN):
#   Bkg  — Background FP: best_iou < 0.10 with all GT (fires on empty background)
#   Loc  — Localisation FP: 0.10 <= best_iou < 0.50, correct class (too imprecise)
#   Cls  — Classification FP: best_iou >= 0.50, wrong class (WBC labelled as parasite)
#   Dup  — Duplicate FP: best_iou >= 0.50, correct class but GT already consumed
#   Both — Wrong class AND imprecise: 0.10 <= best_iou < 0.50, wrong class
#   Miss — False Negative: GT instance matched by no detection at any threshold
#
# Counts (not AP-delta) are reported since AP-delta requires the full TIDE harness.
#
# Outputs (saved to outputs/analysis_results/phase_3/):
#   tide_error_decomposition.csv
#   tide_error_decomposition_comparison.png
# =============================================================================

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
out_dir      = phase3_dir
out_dir.mkdir(parents=True, exist_ok=True)

IMG_W, IMG_H = 2048, 2048   # actual image dimensions
LOC_LO       = 0.10         # IoU lower bound for Loc/Both/Cls/Dup errors
IOU_THRESH   = 0.50         # TP threshold

# =============================================================================
# HELPER: count GT instances per class from label directory
# =============================================================================
def count_gt_per_class(labels_dir):
    counts = {}
    for lf in sorted(Path(labels_dir).glob("*.txt")):
        for line in lf.read_text().strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                counts[cls_id] = counts.get(cls_id, 0) + 1
    return counts


# =============================================================================
# HELPER: decompose FPs and compute Miss for one model
# =============================================================================
def tide_decompose(pred_csv, labels_dir, class_names, model_label):
    df = pd.read_csv(pred_csv)
    fp_df = df[df['is_fp']].copy()
    tp_df = df[df['is_tp']].copy()

    n_total  = len(df)
    n_tp     = len(tp_df)
    n_fp     = len(fp_df)

    # --- FP error categories ---
    bkg  = fp_df[fp_df['best_iou'] < LOC_LO]
    # FPs where 0.1 <= IoU < 0.5
    mid_iou = fp_df[(fp_df['best_iou'] >= LOC_LO) & (fp_df['best_iou'] < IOU_THRESH)]
    loc  = mid_iou[mid_iou['matched_gt_cls'] == mid_iou['cls']]
    both = mid_iou[mid_iou['matched_gt_cls'] != mid_iou['cls']]
    # FPs where IoU >= 0.5 (GT already consumed, or wrong class)
    hi_iou = fp_df[fp_df['best_iou'] >= IOU_THRESH]
    dup  = hi_iou[hi_iou['matched_gt_cls'] == hi_iou['cls']]
    cls_err = hi_iou[hi_iou['matched_gt_cls'] != hi_iou['cls']]

    # --- False Negatives (Miss) per class ---
    gt_counts = count_gt_per_class(labels_dir)
    # Count TP matches per class (using matched_gt_cls of TPs)
    tp_per_class = tp_df['matched_gt_cls'].value_counts().to_dict()
    miss_per_class = {}
    for cls_id, n_gt in gt_counts.items():
        matched = tp_per_class.get(cls_id, 0)
        miss_per_class[cls_id] = max(0, n_gt - matched)
    n_miss = sum(miss_per_class.values())

    row = {
        'model':   model_label,
        'n_total': n_total,
        'n_tp':    n_tp,
        'n_fp':    n_fp,
        'Bkg':     len(bkg),
        'Loc':     len(loc),
        'Cls':     len(cls_err),
        'Dup':     len(dup),
        'Both':    len(both),
        'Miss':    n_miss,
    }

    print(f"\n{'='*55}")
    print(f"TIDE Decomposition — {model_label}")
    print(f"{'='*55}")
    print(f"  Total predictions : {n_total}")
    print(f"  True Positives    : {n_tp}")
    print(f"  False Positives   : {n_fp}")
    print(f"\n  FP breakdown:")
    print(f"    Bkg  (IoU<0.10, no GT)          : {len(bkg):5d}  ({len(bkg)/max(n_fp,1)*100:.1f}%)")
    print(f"    Loc  (0.10<=IoU<0.50, right cls): {len(loc):5d}  ({len(loc)/max(n_fp,1)*100:.1f}%)")
    print(f"    Cls  (IoU>=0.50, wrong cls)     : {len(cls_err):5d}  ({len(cls_err)/max(n_fp,1)*100:.1f}%)")
    print(f"    Dup  (IoU>=0.50, right cls, dup): {len(dup):5d}  ({len(dup)/max(n_fp,1)*100:.1f}%)")
    print(f"    Both (0.10<=IoU<0.50, wrong cls): {len(both):5d}  ({len(both)/max(n_fp,1)*100:.1f}%)")
    print(f"\n  False Negatives (Miss):")
    for cls_id, n_gt in sorted(gt_counts.items()):
        name = class_names[cls_id] if cls_id < len(class_names) else f'cls_{cls_id}'
        print(f"    {name}: {n_gt} GT, {tp_per_class.get(cls_id,0)} detected, "
              f"{miss_per_class[cls_id]} missed ({miss_per_class[cls_id]/max(n_gt,1)*100:.1f}%)")
    print(f"    Total missed : {n_miss}")
    return row


# =============================================================================
# RUN FOR BOTH MODELS
# =============================================================================
row_a = tide_decompose(
    pred_csv    = phase3_dir / "predictions_tp_fp_labeled_model_a.csv",
    labels_dir  = project_root / "model_a/test/labels",
    class_names = ['parasite'],
    model_label = 'Model_A'
)

row_b = tide_decompose(
    pred_csv    = phase3_dir / "predictions_tp_fp_labeled_model_b.csv",
    labels_dir  = project_root / "model_b/test/labels",
    class_names = ['parasite', 'WBC'],
    model_label = 'Model_B'
)

# =============================================================================
# SAVE CSV
# =============================================================================
df_tide = pd.DataFrame([row_a, row_b])
df_tide.to_csv(out_dir / "tide_error_decomposition.csv", index=False)
print(f"\n\nTIDE results saved to: {out_dir}/tide_error_decomposition.csv")
print(df_tide[['model','Bkg','Loc','Cls','Dup','Both','Miss']].to_string(index=False))

# =============================================================================
# PLOT
# =============================================================================
error_cols = ['Bkg', 'Loc', 'Cls', 'Dup', 'Both', 'Miss']
colors_a   = ['#2980b9'] * 6
colors_b   = ['#e67e22'] * 6

vals_a = [row_a[c] for c in error_cols]
vals_b = [row_b[c] for c in error_cols]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
x = np.arange(len(error_cols))
width = 0.35
labels_x = ['Background\n(Bkg)', 'Localisation\n(Loc)', 'Class\nConfusion\n(Cls)',
             'Duplicate\n(Dup)', 'Both\nErrors\n(Both)', 'Missed GT\n(Miss)']

for ax, vals, model, color in [(axes[0], vals_a, 'Model A (parasite only)', '#2980b9'),
                                (axes[1], vals_b, 'Model B (parasite + WBC)', '#e67e22')]:
    bars = ax.bar(x, vals, color=color, alpha=0.85, edgecolor='white')
    ax.set_title(model, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=9)
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.annotate(str(v), xy=(bar.get_x() + bar.get_width()/2, v),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', fontsize=10, fontweight='bold')

fig.suptitle('TIDE-Style Error Decomposition: Error Counts by Category', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(out_dir / "tide_error_decomposition_comparison.png", dpi=150, bbox_inches='tight')
print(f"\nTIDE chart saved to: {out_dir}/tide_error_decomposition_comparison.png")

print("\n--- Key insight ---")
print(f"  Model A: {row_a['Bkg']} Bkg FPs ({row_a['Bkg']/max(row_a['n_fp'],1)*100:.1f}%) — "
      f"no class confusion possible (single class)")
print(f"  Model B: {row_b['Cls']} Cls FPs ({row_b['Cls']/max(row_b['n_fp'],1)*100:.1f}%) — "
      f"parasite predictions overlapping WBC GT (direct WBC confusion count)")
print(f"  Model B: {row_b['Bkg']} Bkg FPs ({row_b['Bkg']/max(row_b['n_fp'],1)*100:.1f}%) — "
      f"pure background fires")
print("\nPhase 3 — TIDE-style decomposition complete.")
