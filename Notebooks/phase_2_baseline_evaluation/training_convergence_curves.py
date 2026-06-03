# =============================================================================
# Training Convergence Curves — Model A vs Model B
# Plots loss (box, cls, dfl) and mAP50/mAP50-95 over epochs for both models
# on a shared figure for direct comparison.
# Output: outputs/analysis_results/phase_2/training_convergence_curves.png
# =============================================================================
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

project_root = Path("/home/sadeniji/scratch/Practicum_Main_Project")
out_dir = project_root / "outputs/analysis_results/phase_2"
out_dir.mkdir(parents=True, exist_ok=True)

csv_a = project_root / "outputs/model_a/model_a_parasite_detection-7/results.csv"
csv_b = project_root / "outputs/model_b/model_b_parasite_wbc_detection/results.csv"

df_a = pd.read_csv(csv_a).rename(columns=str.strip)
df_b = pd.read_csv(csv_b).rename(columns=str.strip)

# Trim whitespace from column names
df_a.columns = df_a.columns.str.strip()
df_b.columns = df_b.columns.str.strip()

COLOR_A = '#2980b9'
COLOR_B = '#e67e22'

fig = plt.figure(figsize=(16, 10))
fig.suptitle('Training Convergence: Model A (parasite only) vs Model B (parasite + WBC)',
             fontsize=14, fontweight='bold', y=1.01)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

panels = [
    ('train/box_loss',      'Box Loss (train)',       True),
    ('train/cls_loss',      'Class Loss (train)',     True),
    ('train/dfl_loss',      'DFL Loss (train)',       True),
    ('metrics/mAP50(B)',    'mAP50 (validation)',     False),
    ('metrics/mAP50-95(B)', 'mAP50-95 (validation)',  False),
    ('metrics/precision(B)','Precision (validation)', False),
]

for idx, (col, title, is_loss) in enumerate(panels):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    ax.plot(df_a['epoch'], df_a[col], color=COLOR_A, linewidth=1.8,
            label='Model A', alpha=0.9)
    ax.plot(df_b['epoch'], df_b[col], color=COLOR_B, linewidth=1.8,
            label='Model B', alpha=0.9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=8)
    if not is_loss:
        # Annotate final values
        final_a = df_a[col].iloc[-1]
        final_b = df_b[col].iloc[-1]
        ax.annotate(f'{final_a:.3f}', xy=(df_a['epoch'].iloc[-1], final_a),
                    xytext=(-30, 6), textcoords='offset points',
                    fontsize=8, color=COLOR_A, fontweight='bold')
        ax.annotate(f'{final_b:.3f}', xy=(df_b['epoch'].iloc[-1], final_b),
                    xytext=(-30, -14), textcoords='offset points',
                    fontsize=8, color=COLOR_B, fontweight='bold')

plt.tight_layout()
out_path = out_dir / "training_convergence_curves.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
