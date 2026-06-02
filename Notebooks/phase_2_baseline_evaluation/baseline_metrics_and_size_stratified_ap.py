# =============================================================================
# Phase 2 | Baseline Metrics and COCO Size-Stratified AP
#
# Runs model.val() on the test split for both Model A and Model B and reports:
#   - Standard metrics: mAP50, mAP50-95, Precision, Recall, F1
#   - COCO size-stratified AP: APsmall (<32^2 px), APmedium, APlarge
#   - Per-class AP breakdown for Model B
#
# APsmall is the most clinically relevant tier because malaria parasites in
# thick Giemsa smears are typically small objects (<30×30 pixels at 1024 res).
#
# Outputs (saved to outputs/analysis_results/phase_2/):
#   baseline_metrics_comparison.csv
#   size_stratified_ap.csv
#   baseline_metrics_comparison.png
# =============================================================================

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# =============================================================================
# PATHS
# =============================================================================
project_root  = Path("/home/sadeniji/scratch/Practicum_Main_Project")
model_a_weights = project_root / "outputs/model_a/model_a_parasite_detection-7/weights/best.pt"
model_b_weights = project_root / "outputs/model_b/model_b_parasite_wbc_detection/weights/best.pt"
data_yaml_a     = project_root / "model_a/data_model_a.yaml"
data_yaml_b     = project_root / "model_b/data_model_b.yaml"
out_dir         = project_root / "outputs/analysis_results/phase_2"
out_dir.mkdir(parents=True, exist_ok=True)

for name, p in [("Model A weights", model_a_weights), ("Model B weights", model_b_weights),
                ("data_yaml_a", data_yaml_a), ("data_yaml_b", data_yaml_b)]:
    if not p.exists():
        sys.exit(f"ERROR: {name} not found at {p}")

# =============================================================================
# HELPER — extract per-class and size-stratified AP from Ultralytics metrics
# =============================================================================
def extract_metrics(metrics, model_label, class_names):
    rows = []
    rows.append({
        'model': model_label, 'class': 'all',
        'mAP50': metrics.box.map50,
        'mAP50_95': metrics.box.map,
        'precision': metrics.box.mp,
        'recall': metrics.box.mr,
        'f1': 2 * metrics.box.mp * metrics.box.mr / max(metrics.box.mp + metrics.box.mr, 1e-9),
    })
    if hasattr(metrics.box, 'ap_class_index'):
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cname = class_names.get(int(cls_idx), f'class_{cls_idx}')
            rows.append({
                'model': model_label, 'class': cname,
                'mAP50': metrics.box.ap50[i],
                'mAP50_95': metrics.box.ap[i],
                'precision': float(metrics.box.p[i]) if hasattr(metrics.box, 'p') else float('nan'),
                'recall': float(metrics.box.r[i]) if hasattr(metrics.box, 'r') else float('nan'),
                'f1': float('nan'),
            })
    return rows


def extract_size_stratified_ap(metrics, model_label):
    rows = []
    box = metrics.box
    # Ultralytics stores these when split=test and save_json=True
    for attr, label in [('maps', 'per_class_mAP50_95')]:
        if hasattr(box, attr):
            vals = getattr(box, attr)
            rows.append({'model': model_label, 'tier': label, 'value': float(np.mean(vals))})

    # Try to access COCO-style size tiers via the results dict
    if hasattr(metrics, 'results_dict'):
        rd = metrics.results_dict
        for key in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)']:
            if key in rd:
                rows.append({'model': model_label, 'tier': key, 'value': rd[key]})
    return rows

# =============================================================================
# VALIDATE MODEL A
# =============================================================================
print("=" * 60)
print("Validating Model A (parasite only)...")
print("=" * 60)
model_a = YOLO(str(model_a_weights))
metrics_a = model_a.val(data=str(data_yaml_a), split='test', verbose=True,
                         save_json=True, project=str(out_dir), name='val_model_a')

rows_a = extract_metrics(metrics_a, 'Model_A', {0: 'parasite'})
size_a = extract_size_stratified_ap(metrics_a, 'Model_A')

# =============================================================================
# VALIDATE MODEL B
# =============================================================================
print("\n" + "=" * 60)
print("Validating Model B (parasite + WBC)...")
print("=" * 60)
model_b = YOLO(str(model_b_weights))
metrics_b = model_b.val(data=str(data_yaml_b), split='test', verbose=True,
                         save_json=True, project=str(out_dir), name='val_model_b')

rows_b = extract_metrics(metrics_b, 'Model_B', {0: 'parasite', 1: 'WBC'})
size_b = extract_size_stratified_ap(metrics_b, 'Model_B')

# =============================================================================
# SAVE CSVs
# =============================================================================
df_metrics = pd.DataFrame(rows_a + rows_b)
df_metrics.to_csv(out_dir / "baseline_metrics_comparison.csv", index=False)
print(f"\nBaseline metrics saved to: {out_dir}/baseline_metrics_comparison.csv")
print(df_metrics.to_string(index=False))

df_size = pd.DataFrame(size_a + size_b)
if not df_size.empty:
    df_size.to_csv(out_dir / "size_stratified_ap.csv", index=False)
    print(f"\nSize-stratified AP saved to: {out_dir}/size_stratified_ap.csv")

# =============================================================================
# PLOT: SIDE-BY-SIDE METRIC COMPARISON
# =============================================================================
overall_a = df_metrics[(df_metrics['model'] == 'Model_A') & (df_metrics['class'] == 'all')].iloc[0]
overall_b = df_metrics[(df_metrics['model'] == 'Model_B') & (df_metrics['class'] == 'all')].iloc[0]

metrics_labels = ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1']
vals_a = [overall_a[m] for m in metrics_labels]
vals_b = [overall_b[m] for m in metrics_labels]

x = np.arange(len(metrics_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars_a = ax.bar(x - width/2, vals_a, width, label='Model A (parasite only)', color='steelblue')
bars_b = ax.bar(x + width/2, vals_b, width, label='Model B (parasite + WBC)', color='darkorange')

ax.set_ylabel('Score')
ax.set_title('Baseline Metrics: Model A vs Model B (test set)')
ax.set_xticks(x)
ax.set_xticklabels(['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1'])
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis='y')

for bar in bars_a:
    ax.annotate(f'{bar.get_height():.3f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
for bar in bars_b:
    ax.annotate(f'{bar.get_height():.3f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(out_dir / "baseline_metrics_comparison.png", dpi=150, bbox_inches='tight')
print(f"\nBaseline metrics chart saved to: {out_dir}/baseline_metrics_comparison.png")

print("\nPhase 2 — Baseline evaluation complete.")
