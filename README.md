# Malaria Parasite Detection — False Positive Analysis Pipeline

**SPARK Academy 2026 | Team Rwanda**

A full machine learning pipeline for malaria parasite detection in thick Giemsa blood smear images, with a focus on characterising false positive (FP) predictions and testing whether FPs spatially cluster around white blood cell (WBC) locations.

---

## Project Overview

Automated malaria diagnosis from thick blood smear microscopy images requires object detection models that are not only accurate but also clinically trustworthy. This project trains two YOLOv12s detectors on the **Lacuna Malaria Detection Challenge** dataset and then applies a seven-phase analysis pipeline to:

1. Establish baseline detection performance
2. Extract and characterise all false positive predictions
3. Statistically test the hypothesis that FPs cluster near WBC locations
4. Compare the two models rigorously with paired statistical tests
5. Calibrate confidence scores for clinical reliability

**Dataset:** [Lacuna Malaria Detection Challenge](https://zindi.africa/competitions/lacuna-malaria-detection-challenge) (1,322 participants, Zindi 2024)
**Architecture:** YOLOv12s (Ultralytics), image size 1024×1024, 75 epochs
**Compute:** Narval HPC cluster (Compute Canada), A100 GPUs via SLURM

---

## Models

| Model | Classes | Val mAP50 | Val mAP50-95 | Description |
|-------|---------|-----------|--------------|-------------|
| **Model A** | parasite (×1) | **0.771** | **0.325** | Baseline single-class parasite detector |
| **Model B** | parasite + WBC (×2) | **0.871** | **0.451** | Two-class detector; WBC class provides spatial context |

Model B significantly outperforms Model A and its WBC detections provide the centroid coordinates used directly in the spatial FP analysis (Phase 4).

---

## Repository Structure

```
.
├── Notebooks/
│   ├── phase_1_training/
│   │   ├── train_model_a_parasite_only.py          # YOLOv12s single-class training
│   │   └── train_model_b_parasite_and_wbc.py       # YOLOv12s two-class training
│   │
│   ├── phase_2_baseline_evaluation/
│   │   ├── baseline_metrics_and_size_stratified_ap.py   # mAP50, mAP50-95, APsmall/medium/large
│   │   ├── per_confidence_fp_rate_curves.py             # FP rate vs confidence threshold sweep
│   │   └── confusion_matrix_multi_iou_thresholds.py     # Confusion matrices at IoU 0.1–0.9
│   │
│   ├── phase_3_fp_extraction/
│   │   ├── extract_predictions_and_label_tp_fp.py   # Central artifact: labels every prediction TP/FP
│   │   ├── tide_error_decomposition.py              # TIDE: Bkg/Cls/Loc/Dup/Miss breakdown
│   │   └── hard_negative_audit_and_visualization.py # Visual inspection of highest-confidence FPs
│   │
│   ├── phase_4_spatial_analysis/
│   │   ├── mannwhitney_distance_to_nearest_wbc.py           # Primary WBC-proximity hypothesis test
│   │   ├── distance_stratification_by_wbc_proximity.py      # FP rate per distance band (0–30, 30–75 … px)
│   │   ├── ripley_cross_k_function_fp_wbc_attraction.py     # K12(r): FP-WBC spatial attraction vs CSR
│   │   ├── kde_correlation_fp_density_vs_wbc_density.py     # Pixel-wise Pearson r of KDE surfaces
│   │   ├── morans_i_spatial_autocorrelation.py              # Global + Local Moran's I on FP grid
│   │   ├── overdispersion_negative_binomial_test.py         # Poisson vs NB fit for FP counts/image
│   │   └── dbscan_fp_cluster_analysis.py                    # DBSCAN cluster proximity to WBCs
│   │
│   ├── phase_5_model_comparison/
│   │   ├── paired_bootstrap_map_comparison.py               # Bootstrap mAP delta + BCa 95% CI
│   │   ├── wilcoxon_signedrank_per_image_ap.py              # Wilcoxon test on per-image AP50 vectors
│   │   └── mcnemar_test_and_stratified_wbc_proximity.py     # McNemar + BH-corrected stratified test
│   │
│   ├── phase_6_calibration/
│   │   ├── ece_mce_reliability_diagrams.py                  # ECE, MCE, reliability diagrams (baseline)
│   │   ├── temperature_scaling_calibration.py               # Temperature scaling; ECE before/after
│   │   └── multivariate_position_aware_calibration.py       # Position-ECE heatmap; LR calibration
│   │
│   ├── phase_7_reporting/
│   │   ├── generate_tide_error_breakdown_chart.py           # Publication TIDE stacked bar figure
│   │   ├── generate_spatial_heatmaps_and_kde_overlays.py   # KDE overlays on test images
│   │   └── generate_final_summary_tables.py                 # All summary tables + combined Excel
│   │
│   ├── false_positive_analysis.ipynb          # Original exploratory FP notebook
│   ├── spatial_analysis_wbc_clustering.ipynb  # Original spatial analysis notebook
│   ├── model_a_training.ipynb / .py           # Original training scripts (used by SLURM)
│   └── model_b_training.ipynb / .py
│
├── model_a/
│   └── data_model_a.yaml      # YOLO data config — single class (parasite)
├── model_b/
│   ├── data_model_b.yaml      # YOLO data config — two classes (parasite, WBC)
│   └── yolov8_results/        # Exploratory training run results and visualisations
│
├── Papers/
│   └── analysis_plan.txt      # Full written plan for all 7 phases with references
├── Other Resources/
│   ├── requirements.txt
│   └── SPARK2026_TeamRwanda_Hackathon_Resources.pdf
├── train_model_a.sh           # SLURM job script for Model A
├── train_model_b.sh           # SLURM job script for Model B
├── model_a_training.py        # Training entrypoint (called by SLURM)
└── .gitignore
```

---

## Analysis Pipeline: Phases 1–7

### Phase 1 — Training
Trains YOLOv12s on the Lacuna dataset using Ultralytics. Model A uses a single parasite class; Model B adds WBC as a second class. Both trained for 75 epochs at image size 1024 with early stopping (patience=25).

**Key design decision:** Model B's WBC class is not just for accuracy — the WBC centroid coordinates it produces at inference time feed directly into the Phase 4 spatial analysis without needing a separate WBC detector.

### Phase 2 — Baseline Evaluation
- mAP50, mAP50-95, Precision, Recall, F1 on the held-out test set
- COCO size-stratified AP (APsmall is the critical metric — parasites are small objects)
- Confidence threshold sweep (0.05 → 0.95): plots FP count per image vs threshold to identify the clinical operating point
- Confusion matrices at IoU ∈ {0.1, 0.3, 0.5, 0.7, 0.9}

### Phase 3 — False Positive Extraction
Produces `predictions_tp_fp_labeled_*.csv` — the central data artifact consumed by all downstream phases. Each prediction is matched to ground truth at IoU ≥ 0.5 and labeled TP or FP.

**TIDE decomposition** categorises all errors into:
- `Bkg` — background FP (no GT overlap at any IoU)
- `Cls` — class confusion (correct location, wrong class) — **in Model B, this = WBC-as-parasite count**
- `Loc` — localisation error (correct class, IoU just below threshold)
- `Dup` — duplicate detection of an already-matched GT
- `Miss` — GT object not detected at any confidence

### Phase 4 — Spatial Analysis of FPs
Tests the core hypothesis: *FPs cluster near annotated WBC locations.*

| Method | What it tests |
|--------|---------------|
| **Mann-Whitney U** | FP mean dist-to-WBC < TP mean dist-to-WBC (one-tailed) |
| **Distance stratification** | FP rate per proximity band (0–30, 30–75, 75–150, 150–300, >300 px) |
| **Ripley's Cross-K (K12)** | Spatial attraction between FP centroids and WBC centroids vs CSR null |
| **KDE correlation** | Pixel-wise Pearson r between FP density surface and WBC density surface |
| **Moran's I (LISA)** | Spatial autocorrelation of FP counts on a 50px tile grid; hot-spot mapping |
| **Overdispersion test** | Poisson vs Negative Binomial fit for FP counts per image |
| **DBSCAN clustering** | Whether FP clusters co-locate with annotated WBC positions |

### Phase 5 — Statistical Model Comparison
| Test | Purpose |
|------|---------|
| **Paired bootstrap** (N=10,000, BCa CI) | Is the mAP50 difference significant? |
| **Wilcoxon signed-rank** | Non-parametric paired test on per-image AP50 |
| **McNemar's test** | Per-GT-annotation recall comparison |
| **Stratified McNemar + BH correction** | Does Model B's advantage concentrate in WBC-proximal regions? |

### Phase 6 — Confidence Calibration
YOLO confidence scores are typically overconfident. For clinical deployment (parasite density counting), a confidence score of 0.7 should mean a 70% probability of a true parasite.

| Method | Description |
|--------|-------------|
| **ECE / MCE + reliability diagrams** | Quantifies and visualises miscalibration |
| **Temperature scaling** | Single scalar T; no retraining; reduces ECE significantly |
| **Position-aware calibration** | Logistic regression on (conf, x, y, w, h) — tests if WBC-dense regions are less calibrated |

### Phase 7 — Reporting
Assembles all results into publication-ready figures (TIDE bar chart, KDE overlays, Ripley's K curves, reliability diagrams) and summary tables (CSV + LaTeX + Excel).

---

## Execution Order

> **Critical:** Phase 3 must run before Phases 4–7. Its output CSVs are the shared data source for all downstream analysis.

```
Phase 1  →  Phase 2  →  Phase 3  →  Phase 4
                                  →  Phase 5
                                  →  Phase 6
                                           →  Phase 7
```

---

## Installation

```bash
pip install ultralytics tidecv mlxtend esda libpysal pointpats netcal stambo openpyxl
```

Core dependencies: `torch`, `ultralytics`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `opencv-python`

---

## Running on HPC (SLURM)

```bash
sbatch train_model_a.sh   # Model A — ~10h on A100
sbatch train_model_b.sh   # Model B — ~10h on A100
```

Analysis phases run on a login node or interactive session after training completes:

```bash
python Notebooks/phase_3_fp_extraction/extract_predictions_and_label_tp_fp.py
python Notebooks/phase_4_spatial_analysis/mannwhitney_distance_to_nearest_wbc.py
# ... etc.
```

---

## Key References

| Paper | Relevance |
|-------|-----------|
| YOLOv12 (arXiv 2502.12524, NeurIPS 2025) | Model architecture |
| TIDE: Bolya et al. (ECCV 2020, arXiv 2008.08115) | Error decomposition |
| Confidence Score: MDPI Sensors 2021 | Why precision > mAP for clinical use |
| Ripley (1976); Molina et al. PMC2889688 | Cross-K spatial analysis |
| Meissner et al. PMC3831262 (2013) | Overdispersion baseline for thick smears |
| Guo et al. ICML 2017 | Temperature scaling calibration |
| Küppers et al. CVPR Workshop 2020 | Position-aware calibration |
| "When +1% Is Not Enough" arXiv 2511.19794 | Bootstrap CI for mAP comparison |
| YOLO-PAM: MDPI J. Imaging 2023, PMC10744183 | WBC-parasite confusion mitigation |
| Frontiers in Malaria 2024 | Clinical metrics for malaria ML |

---

## Team

**SPARK Academy 2026 — Team Rwanda**
