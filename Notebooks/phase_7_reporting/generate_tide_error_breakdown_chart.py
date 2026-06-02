# =============================================================================
# Phase 7 | Generate TIDE Error Breakdown Chart
#
# Produces the final publication-quality TIDE error decomposition figure:
# a stacked bar chart showing the proportion of each error type
# (Bkg / Cls / Loc / Dup / Miss / Both) for Model A vs Model B.
#
# Prerequisite: Phase 3 tide_error_decomposition.py
#
# Outputs (saved to outputs/analysis_results/phase_7/figures/):
#   tide_error_breakdown_final.png
#   tide_error_breakdown_final.pdf
# =============================================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path("/home/sadeniji/scratch/Practicum_Main_Project")
phase3_dir   = project_root / "outputs/analysis_results/phase_3"
out_dir      = project_root / "outputs/analysis_results/phase_7/figures"
out_dir.mkdir(parents=True, exist_ok=True)

tide_csv = phase3_dir / "tide_error_decomposition.csv"
if not tide_csv.exists():
    sys.exit("ERROR: Run Phase 3 tide_error_decomposition.py first.")

df = pd.read_csv(tide_csv)

ERROR_COLS   = ['Bkg', 'Cls', 'Loc', 'Dup', 'Both', 'Miss']
ERROR_LABELS = ['Background\n(Bkg)', 'Class\nConfusion\n(Cls)',
                'Localization\n(Loc)', 'Duplicate\n(Dup)',
                'Both Errors\n(Both)', 'Missed GT\n(Miss)']
COLORS = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#9b59b6', '#3498db']

models = df['model'].tolist()
data   = df[ERROR_COLS].values  # shape (n_models, n_error_types)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('TIDE Error Decomposition: Model A vs Model B', fontsize=14, fontweight='bold')

# Panel 1: Grouped bar chart (absolute AP error values)
ax = axes[0]
x = np.arange(len(ERROR_COLS))
width = 0.35
for i, (row, model, hatch) in enumerate(zip(data, models, ['', '///'])):
    offset = (i - 0.5) * width
    bars = ax.bar(x + offset, row, width, label=model,
                  color=COLORS, alpha=0.85, hatch=hatch, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(ERROR_LABELS, fontsize=9)
ax.set_ylabel('AP Error Contribution')
ax.set_title('Absolute AP Error per Category')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Panel 2: Normalized stacked bar (% of total error per model)
ax = axes[1]
totals  = data.sum(axis=1, keepdims=True)
pct_data = data / np.where(totals > 0, totals, 1) * 100

bottoms = np.zeros(len(models))
for j, (col, color, label) in enumerate(zip(ERROR_COLS, COLORS, ERROR_LABELS)):
    vals = pct_data[:, j]
    bars = ax.bar(models, vals, bottom=bottoms, color=color, label=col.upper(),
                  edgecolor='white', linewidth=0.5)
    for k, (bar, val) in enumerate(zip(bars, vals)):
        if val >= 4:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bottoms[k] + val/2, f'{val:.1f}%',
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    bottoms += vals

ax.set_ylabel('% of Total Error')
ax.set_title('Error Category Breakdown (%)')
ax.legend(loc='upper right', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 105])

plt.tight_layout()
for fmt, fname in [('png', 'tide_error_breakdown_final.png'),
                    ('pdf', 'tide_error_breakdown_final.pdf')]:
    plt.savefig(out_dir / fname, dpi=150, bbox_inches='tight', format=fmt)
    print(f"Saved: {out_dir / fname}")

print("\nPhase 7 — TIDE chart generated.")
