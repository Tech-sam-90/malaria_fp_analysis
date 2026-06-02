# =============================================================================
# Phase 7 | Generate Final Summary Tables for the Report
#
# Collects all analysis results from Phases 2–6 and assembles them into
# clean, publication-ready summary tables:
#
#   Table 1: Baseline metrics (mAP50, mAP50-95, Precision, Recall, F1)
#   Table 2: COCO size-stratified AP (APsmall, APmedium, APlarge)
#   Table 3: TIDE error breakdown (% per category)
#   Table 4: Statistical model comparison (bootstrap CI, Wilcoxon, McNemar)
#   Table 5: Confidence calibration (ECE before/after temperature scaling)
#   Table 6: Spatial analysis summary (FP distance, Mann-Whitney, Moran's I)
#
# All tables are saved as CSV and as LaTeX-formatted text files.
#
# Outputs (saved to outputs/analysis_results/phase_7/tables/):
#   table_1_baseline_metrics.csv / .tex
#   table_2_size_stratified_ap.csv / .tex
#   table_3_tide_error_breakdown.csv / .tex
#   table_4_model_comparison_statistics.csv / .tex
#   table_5_calibration_metrics.csv / .tex
#   table_6_spatial_analysis_summary.csv / .tex
#   all_tables_combined.xlsx
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path("/home/sadeniji/scratch/Practicum_Main_Project")
results_root = project_root / "outputs/analysis_results"
out_dir      = results_root / "phase_7/tables"
out_dir.mkdir(parents=True, exist_ok=True)


def load_or_empty(path, fallback_cols):
    if Path(path).exists():
        return pd.read_csv(path)
    print(f"WARNING: {path} not found — using placeholder.")
    return pd.DataFrame(columns=fallback_cols)


def save_table(df, name, description):
    csv_path = out_dir / f"{name}.csv"
    tex_path = out_dir / f"{name}.tex"
    df.to_csv(csv_path, index=False)
    with open(tex_path, 'w') as f:
        f.write(f"% {description}\n")
        f.write(df.to_latex(index=False, float_format='%.4f', na_rep='—'))
    print(f"  {name}.csv + .tex")
    return df


print("Assembling final summary tables...")

# =============================================================================
# TABLE 1 — BASELINE METRICS
# =============================================================================
df_baseline = load_or_empty(
    results_root / "phase_2/baseline_metrics_comparison.csv",
    ['model', 'class', 'mAP50', 'mAP50_95', 'precision', 'recall', 'f1']
)
df_t1 = df_baseline[df_baseline['class'] == 'all'][
    ['model', 'mAP50', 'mAP50_95', 'precision', 'recall', 'f1']
].rename(columns={'mAP50': 'mAP50', 'mAP50_95': 'mAP50-95',
                   'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1'})
save_table(df_t1, "table_1_baseline_metrics", "Table 1: Baseline Detection Metrics (test set)")

# =============================================================================
# TABLE 2 — SIZE-STRATIFIED AP
# =============================================================================
df_size = load_or_empty(results_root / "phase_2/size_stratified_ap.csv",
                         ['model', 'tier', 'value'])
save_table(df_size, "table_2_size_stratified_ap", "Table 2: COCO Size-Stratified AP")

# =============================================================================
# TABLE 3 — TIDE ERROR BREAKDOWN
# =============================================================================
df_tide = load_or_empty(results_root / "phase_3/tide_error_decomposition.csv",
                         ['model', 'Bkg', 'Cls', 'Loc', 'Dup', 'Both', 'Miss'])
if not df_tide.empty:
    error_cols = [c for c in ['Bkg', 'Cls', 'Loc', 'Dup', 'Both', 'Miss'] if c in df_tide.columns]
    totals = df_tide[error_cols].sum(axis=1)
    df_tide_pct = df_tide.copy()
    for col in error_cols:
        df_tide_pct[col + '_pct'] = (df_tide[col] / totals.replace(0, 1) * 100).round(1)
    df_t3 = df_tide_pct[['model'] + [c + '_pct' for c in error_cols]]
    df_t3.columns = ['Model'] + [c for c in error_cols]
else:
    df_t3 = df_tide

save_table(df_t3, "table_3_tide_error_breakdown",
           "Table 3: TIDE Error Decomposition (% of total error)")

# =============================================================================
# TABLE 4 — STATISTICAL MODEL COMPARISON
# =============================================================================
df_boot = load_or_empty(results_root / "phase_5/paired_bootstrap_results.csv",
                         ['map50_model_a', 'map50_model_b', 'observed_delta',
                          'bca_ci_lo', 'bca_ci_hi', 'p_value_two_sided'])
df_wilc = load_or_empty(results_root / "phase_5/wilcoxon_per_image_ap_results.csv",
                         ['W_statistic', 'p_value_two_sided', 'rank_biserial_r'])
df_mcn  = load_or_empty(results_root / "phase_5/mcnemar_global_results.csv",
                         ['chi2', 'p_value', 'model_favored'])

rows = [{'test': 'Paired Bootstrap', 'statistic': 'Δ mAP50',
          'value': df_boot.get('observed_delta', pd.Series([np.nan]))[0] if not df_boot.empty else np.nan,
          '95_CI': f"[{df_boot.get('bca_ci_lo', pd.Series([np.nan]))[0]:.4f}, "
                   f"{df_boot.get('bca_ci_hi', pd.Series([np.nan]))[0]:.4f}]"
                   if not df_boot.empty else '—',
          'p_value': df_boot.get('p_value_two_sided', pd.Series([np.nan]))[0] if not df_boot.empty else np.nan},
         {'test': 'Wilcoxon Signed-Rank', 'statistic': 'W',
          'value': df_wilc.get('W_statistic', pd.Series([np.nan]))[0] if not df_wilc.empty else np.nan,
          '95_CI': '—',
          'p_value': df_wilc.get('p_value_two_sided', pd.Series([np.nan]))[0] if not df_wilc.empty else np.nan},
         {'test': 'McNemar (global)', 'statistic': 'χ²',
          'value': df_mcn.get('chi2', pd.Series([np.nan]))[0] if not df_mcn.empty else np.nan,
          '95_CI': '—',
          'p_value': df_mcn.get('p_value', pd.Series([np.nan]))[0] if not df_mcn.empty else np.nan}]

save_table(pd.DataFrame(rows), "table_4_model_comparison_statistics",
           "Table 4: Statistical Model Comparison")

# =============================================================================
# TABLE 5 — CALIBRATION METRICS
# =============================================================================
df_cal = load_or_empty(results_root / "phase_6/temperature_scaling_results.csv",
                        ['model', 'temperature_T', 'ece_before', 'ece_after', 'ece_reduction_pct'])
df_pre = load_or_empty(results_root / "phase_6/calibration_metrics_pre_calibration.csv",
                        ['model', 'ECE', 'MCE'])

if not df_cal.empty and not df_pre.empty:
    df_t5 = df_cal.merge(df_pre[['model', 'ECE', 'MCE']].rename(
        columns={'ECE': 'ECE_pre', 'MCE': 'MCE_pre'}), on='model', how='left')
else:
    df_t5 = df_cal

save_table(df_t5, "table_5_calibration_metrics",
           "Table 5: Confidence Calibration Before and After Temperature Scaling")

# =============================================================================
# TABLE 6 — SPATIAL ANALYSIS SUMMARY
# =============================================================================
df_mw   = load_or_empty(results_root / "phase_4/mannwhitney_results.csv",
                         ['model', 'fp_mean_dist', 'tp_mean_dist', 'p_value'])
df_moran = load_or_empty(results_root / "phase_4/morans_i_global_results.csv",
                          ['image_id', 'moran_I', 'p_value'])
df_over = load_or_empty(results_root / "phase_4/overdispersion_test_results.csv",
                         ['model', 'dispersion_index', 'p_chi2', 'nb_preferred'])

moran_summary = {'mean_moran_I': df_moran['moran_I'].mean(),
                  'pct_significant': df_moran['p_value'].lt(0.05).mean() * 100} \
                  if not df_moran.empty else {}

rows6 = []
for _, row in df_mw.iterrows() if not df_mw.empty else []:
    rows6.append({
        'model': row['model'],
        'fp_mean_dist_to_wbc_px': row.get('fp_mean_dist', np.nan),
        'tp_mean_dist_to_wbc_px': row.get('tp_mean_dist', np.nan),
        'mannwhitney_p': row.get('p_value', np.nan),
        'mean_moran_I': round(moran_summary.get('mean_moran_I', np.nan), 4),
        'pct_images_significant_moran': round(moran_summary.get('pct_significant', np.nan), 1),
    })

if not df_over.empty:
    for _, row in df_over.iterrows():
        matched = [r for r in rows6 if r.get('model') == row['model']]
        if matched:
            matched[0].update({
                'dispersion_index': row.get('dispersion_index', np.nan),
                'nb_preferred': row.get('nb_preferred', False),
            })

save_table(pd.DataFrame(rows6) if rows6 else pd.DataFrame(
    columns=['model', 'fp_mean_dist_to_wbc_px', 'tp_mean_dist_to_wbc_px',
             'mannwhitney_p', 'mean_moran_I', 'dispersion_index']),
    "table_6_spatial_analysis_summary",
    "Table 6: Spatial Analysis Summary")

# =============================================================================
# COMBINED EXCEL FILE
# =============================================================================
try:
    with pd.ExcelWriter(out_dir / "all_tables_combined.xlsx", engine='openpyxl') as writer:
        for name, df in [("T1 Baseline", df_t1), ("T2 Size-AP", df_size),
                          ("T3 TIDE", df_t3), ("T4 Stats", pd.DataFrame(rows)),
                          ("T5 Calib", df_t5), ("T6 Spatial", pd.DataFrame(rows6))]:
            df.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"\nCombined Excel saved: {out_dir}/all_tables_combined.xlsx")
except ImportError:
    print("Note: openpyxl not available — Excel file not created.")

print("\nPhase 7 — Final summary tables complete.")
