# =============================================================================
# TP vs FP Confidence Score Distributions — Model A and Model B
# Shows histograms and KDE of confidence scores split by TP/FP label.
# Output: outputs/analysis_results/phase_3/confidence_score_distributions.png
# =============================================================================
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

project_root = Path("/home/sadeniji/scratch/Practicum_Main_Project")
phase3_dir   = project_root / "outputs/analysis_results/phase_3"
out_dir      = phase3_dir

df_a = pd.read_csv(phase3_dir / "predictions_tp_fp_labeled_model_a.csv")
df_b = pd.read_csv(phase3_dir / "predictions_tp_fp_labeled_model_b.csv")
# For Model B, use parasite class only for fair comparison
df_b_p = df_b[df_b['cls'] == 0].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Confidence Score Distributions: True Positives vs False Positives',
             fontsize=13, fontweight='bold')

def plot_conf_dist(ax, df, title):
    tp = df[df['is_tp']]['conf'].values
    fp = df[df['is_fp']]['conf'].values

    bins = np.linspace(0.10, 1.0, 40)
    ax.hist(tp, bins=bins, alpha=0.45, color='#27ae60', label=f'TP (n={len(tp):,})', density=True)
    ax.hist(fp, bins=bins, alpha=0.45, color='#e74c3c', label=f'FP (n={len(fp):,})', density=True)

    # KDE overlays
    for vals, color in [(tp, '#27ae60'), (fp, '#e74c3c')]:
        if len(vals) > 10:
            xs = np.linspace(0.10, 1.0, 300)
            kde = gaussian_kde(vals, bw_method=0.15)
            ax.plot(xs, kde(xs), color=color, linewidth=2.2)

    # Vertical lines for means
    ax.axvline(tp.mean(), color='#27ae60', linestyle='--', linewidth=1.5,
               label=f'TP mean={tp.mean():.3f}')
    ax.axvline(fp.mean(), color='#e74c3c', linestyle='--', linewidth=1.5,
               label=f'FP mean={fp.mean():.3f}')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Confidence Score', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.08, 1.02)

plot_conf_dist(axes[0], df_a, 'Model A — Parasite Only')
plot_conf_dist(axes[1], df_b_p, 'Model B — Parasite Class')

plt.tight_layout()
out_path = out_dir / "confidence_score_distributions.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")

# Print summary stats
for df, label in [(df_a, 'Model A'), (df_b_p, 'Model B (parasite)')]:
    tp = df[df['is_tp']]['conf']
    fp = df[df['is_fp']]['conf']
    print(f"\n{label}:")
    print(f"  TP: mean={tp.mean():.4f}  median={tp.median():.4f}  std={tp.std():.4f}")
    print(f"  FP: mean={fp.mean():.4f}  median={fp.median():.4f}  std={fp.std():.4f}")
