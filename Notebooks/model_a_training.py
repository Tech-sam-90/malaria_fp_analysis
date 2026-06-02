# =============================================================================
# Model A: Parasite-Only Detection (YOLOv12n)
# Converted from Colab notebook for Narval HPC
# =============================================================================

import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC (no display)
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
print(f"YAML file    : {data_yaml}")

print("\n--- Path Validation ---")
print(f"Project root exists : {project_root.exists()}")
print(f"Dataset root exists : {dataset_root.exists()}")
print(f"YAML file exists    : {data_yaml.exists()}")

if not data_yaml.exists():
    print("ERROR: YAML file not found. Aborting.")
    sys.exit(1)

print("\nProject root contents:")
for item in project_root.iterdir():
    print(f"  - {item.name}")

print("\nDataset root contents:")
for item in dataset_root.iterdir():
    print(f"  - {item.name}")

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
    print("ERROR: No GPU found. Aborting.")
    sys.exit(1)

# =============================================================================
# 3. WANDB SETUP (offline mode — no internet on compute nodes)
# =============================================================================
api_key = os.environ.get('WANDB_API_KEY')
if api_key:
    try:
        wandb.login(key=api_key, relogin=False)
        print('WandB login succeeded')
    except Exception as e:
        print(f'WandB login failed: {e}')
else:
    print('WANDB_API_KEY not set; continuing without WandB logging')

if wandb.run is None and api_key:
    try:
        wandb.init(
            project='Spark_Practicum',
            name='model_a_training',
            entity='brain_bytes_rw',
            reinit=True
        )
        print(f'WandB run initialized: {wandb.run.name}')
    except Exception as e:
        print(f'WandB init failed: {e}')

# =============================================================================
# 4. TRAIN
# =============================================================================
print(f"\nUsing YAML: {data_yaml}")
print(data_yaml.read_text())

torch.cuda.empty_cache()

model = YOLO('/home/sadeniji/scratch/weights/yolov12s.pt')

print("\nStarting training...")
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

print("\nTraining completed!")
print(f"Best model saved at: {model.trainer.best}")

# =============================================================================
# 5. COPY BEST WEIGHTS TO OUTPUT DIR
# =============================================================================
best_src = Path(model.trainer.best)
best_dst = output_dir / "model_a_best.pt"
shutil.copy2(best_src, best_dst)
print(f"\nBest model copied to: {best_dst}")
print(f"File size: {best_dst.stat().st_size / (1024*1024):.2f} MB")

# =============================================================================
# 6. VALIDATE
# =============================================================================
print("\nValidating Model A...")
metrics = model.val()
print(f"\nValidation Metrics:")
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

    print(f"\nAvailable columns: {df.columns.tolist()}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Model A - Training Results", fontsize=14)

    axes[0,0].plot(df['epoch'], df['train/box_loss'], label='Train')
    axes[0,0].plot(df['epoch'], df['val/box_loss'],   label='Val')
    axes[0,0].set_title('Box Loss'); axes[0,0].legend()

    axes[0,1].plot(df['epoch'], df['train/cls_loss'], label='Train')
    axes[0,1].plot(df['epoch'], df['val/cls_loss'],   label='Val')
    axes[0,1].set_title('Class Loss'); axes[0,1].legend()

    axes[0,2].plot(df['epoch'], df['train/dfl_loss'], label='Train')
    axes[0,2].plot(df['epoch'], df['val/dfl_loss'],   label='Val')
    axes[0,2].set_title('DFL Loss'); axes[0,2].legend()

    axes[1,0].plot(df['epoch'], df['metrics/precision(B)'])
    axes[1,0].set_title('Precision')

    axes[1,1].plot(df['epoch'], df['metrics/recall(B)'])
    axes[1,1].set_title('Recall')

    axes[1,2].plot(df['epoch'], df['metrics/mAP50(B)'],    label='mAP50')
    axes[1,2].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
    axes[1,2].set_title('mAP'); axes[1,2].legend()

    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "model_a_training_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
else:
    print(f"\nResults CSV not found at {results_csv}, skipping plots.")

# =============================================================================
# 8. INFERENCE ON TEST SET
# =============================================================================
test_images_dir = dataset_root / "test" / "images"
print(f"\nTest images directory : {test_images_dir}")
print(f"Exists                : {test_images_dir.exists()}")

if test_images_dir.exists():
    print("\nRunning inference on test set...")
    preds = model.predict(
        source=str(test_images_dir),
        save=False,
        conf=0.25,
        verbose=False
    )
    print(f"Predictions completed for {len(preds)} images")
    if preds:
        r = preds[0]
        print(f"\nSample result from : {r.path}")
        print(f"  Detections       : {len(r.boxes)}")
        if len(r.boxes) > 0:
            print(f"  First confidence : {r.boxes[0].conf[0]:.4f}")

# =============================================================================
# 9. SYNC WANDB (if offline run was active)
# =============================================================================
if wandb.run is not None:
    wandb.finish()
    print("\nWandB run finished. To sync to dashboard, run from login node:")
    print("  wandb sync ~/scratch/Practicum_Main_Project/outputs/model_a/wandb/offline-run-*")

print("\nDone!")
