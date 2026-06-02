# =============================================================================
# Phase 1 | Model A: Single-Class Parasite-Only Detector (YOLOv12s)
#
# Trains YOLOv12s on the Lacuna Malaria Detection dataset with a single class:
# parasite. This is the baseline detection model for Objectives 1 and 2.
#
# Outputs:
#   outputs/model_a/model_a_parasite_detection/weights/best.pt
#   outputs/model_a/model_a_training_plots.png
#
# Run via SLURM: sbatch train_model_a.sh
# =============================================================================

import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
from ultralytics import YOLO

# =============================================================================
# 1. PATHS
# =============================================================================
project_root = Path("/home/sadeniji/scratch/Practicum_Main_Project")
dataset_root = project_root / "model_a"
data_yaml    = dataset_root / "data_model_a.yaml"
output_dir   = project_root / "outputs" / "model_a"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Project root : {project_root}")
print(f"Dataset root : {dataset_root}")
print(f"YAML         : {data_yaml}")
for label, p in [("project_root", project_root), ("dataset_root", dataset_root), ("data_yaml", data_yaml)]:
    print(f"  {label} exists: {p.exists()}")

if not data_yaml.exists():
    sys.exit("ERROR: data_model_a.yaml not found. Aborting.")

# =============================================================================
# 2. GPU CHECK
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print(f"\nCUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    sys.exit("ERROR: No GPU found. Aborting.")

# =============================================================================
# 3. WANDB SETUP (offline mode — no internet on compute nodes)
# =============================================================================
api_key = os.environ.get('WANDB_API_KEY')
if api_key:
    try:
        wandb.login(key=api_key, relogin=False)
        wandb.init(project='Spark_Practicum', name='model_a_training',
                   entity='brain_bytes_rw', reinit=True)
        print(f'WandB run: {wandb.run.name}')
    except Exception as e:
        print(f'WandB failed: {e}')
else:
    print('WANDB_API_KEY not set — skipping WandB.')

# =============================================================================
# 4. TRAIN
# =============================================================================
torch.cuda.empty_cache()
model = YOLO('/home/sadeniji/scratch/weights/yolov12s.pt')

print("\nStarting Model A training...")
results = model.train(
    data=str(data_yaml),
    epochs=75,
    imgsz=1024,
    batch=16,
    patience=25,
    device=0,
    project=str(output_dir),
    name='model_a_parasite_detection',
    save=True,
    verbose=True,
    plots=True,
    workers=4,
    amp=False,
)

print(f"\nTraining complete. Best weights: {model.trainer.best}")

# =============================================================================
# 5. COPY BEST WEIGHTS TO CHECKPOINTS
# =============================================================================
checkpoints_dir = project_root / "model_checkpoints"
checkpoints_dir.mkdir(parents=True, exist_ok=True)
best_dst = checkpoints_dir / "model_a_best.pt"
shutil.copy2(model.trainer.best, best_dst)
print(f"Best weights copied to: {best_dst}  ({best_dst.stat().st_size / 1e6:.1f} MB)")

# =============================================================================
# 6. VALIDATE ON VAL SET
# =============================================================================
print("\nValidating Model A on val split...")
metrics = model.val()
print(f"  mAP50     : {metrics.box.map50:.4f}")
print(f"  mAP50-95  : {metrics.box.map:.4f}")
print(f"  Precision : {metrics.box.mp:.4f}")
print(f"  Recall    : {metrics.box.mr:.4f}")

# =============================================================================
# 7. PLOT TRAINING CURVES
# =============================================================================
results_csv = output_dir / "model_a_parasite_detection" / "results.csv"
if results_csv.exists():
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Model A (Parasite Only) — Training Curves", fontsize=14)

    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train')
    axes[0, 0].plot(df['epoch'], df['val/box_loss'],   label='Val')
    axes[0, 0].set_title('Box Loss'); axes[0, 0].legend()

    axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train')
    axes[0, 1].plot(df['epoch'], df['val/cls_loss'],   label='Val')
    axes[0, 1].set_title('Class Loss'); axes[0, 1].legend()

    axes[0, 2].plot(df['epoch'], df['train/dfl_loss'], label='Train')
    axes[0, 2].plot(df['epoch'], df['val/dfl_loss'],   label='Val')
    axes[0, 2].set_title('DFL Loss'); axes[0, 2].legend()

    axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'])
    axes[1, 0].set_title('Precision')

    axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'])
    axes[1, 1].set_title('Recall')

    axes[1, 2].plot(df['epoch'], df['metrics/mAP50(B)'],    label='mAP50')
    axes[1, 2].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
    axes[1, 2].set_title('mAP'); axes[1, 2].legend()

    for ax in axes.flat:
        ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "model_a_training_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved to: {plot_path}")

# =============================================================================
# 8. SAMPLE INFERENCE ON TEST SET
# =============================================================================
test_images_dir = dataset_root / "test" / "images"
if test_images_dir.exists():
    preds = model.predict(source=str(test_images_dir), save=False, conf=0.25, verbose=False)
    total_dets = sum(len(r.boxes) for r in preds)
    print(f"\nTest set inference: {len(preds)} images, {total_dets} detections at conf>=0.25")

# =============================================================================
# 9. FINISH WANDB
# =============================================================================
if wandb.run is not None:
    wandb.finish()
    print("\nWandB run finished.")

print("\nDone — Model A training complete.")
