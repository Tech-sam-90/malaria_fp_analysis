# =============================================================================
# Phase 4 | Overdispersion Test: Poisson vs Negative Binomial for FP Counts
#
# Tests whether FP counts per image follow a Poisson distribution (random,
# variance = mean) or a Negative Binomial distribution (overdispersed,
# variance > mean — consistent with clustering).
#
# This matters because:
#   - Poisson null → use standard Poisson-based tests for spatial clustering
#   - Negative Binomial null → use NB-adjusted tests (more conservative)
#
# Meissner et al. (2013, PMC3831262) showed that BOTH parasite and leukocyte
# spatial distributions in thick smears are inherently overdispersed.
# If FP counts are also overdispersed, the WBC-proximity effect must be
# tested against a NB baseline (not Poisson) to avoid false positives.
#
# Tests performed:
#   1. Index of Dispersion (variance / mean, chi-squared test)
#   2. Likelihood Ratio Test: Poisson vs Negative Binomial MLE fits
#   3. AIC comparison: Poisson vs NB
#   4. Cameron & Trivedi's overdispersion regression test
#
# Outputs (saved to outputs/analysis_results/phase_4/):
#   overdispersion_test_results.csv
#   fp_count_distribution_fit.png
# =============================================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln

# =============================================================================
# PATHS
# =============================================================================
project_root = Path("/home/sadeniji/scratch/Practicum_Main_Project")
phase3_dir   = project_root / "outputs/analysis_results/phase_3"
out_dir      = project_root / "outputs/analysis_results/phase_4"
out_dir.mkdir(parents=True, exist_ok=True)

pred_csv_a = phase3_dir / "predictions_tp_fp_labeled_model_a.csv"
pred_csv_b = phase3_dir / "predictions_tp_fp_labeled_model_b.csv"

CONF_THRESH = 0.25

# =============================================================================
# NEGATIVE BINOMIAL LOG-LIKELIHOOD
# =============================================================================
def nb_loglik(params, counts):
    mu, r = params
    if mu <= 0 or r <= 0:
        return 1e10
    p = r / (r + mu)
    ll = np.sum(
        gammaln(counts + r) - gammaln(r) - gammaln(counts + 1)
        + r * np.log(p) + counts * np.log(1 - p)
    )
    return -ll   # minimise negative log-likelihood


def fit_poisson(counts):
    mu_hat = counts.mean()
    ll = np.sum(stats.poisson.logpmf(counts, mu_hat))
    k = 1
    aic = 2*k - 2*ll
    return mu_hat, ll, aic


def fit_negative_binomial(counts):
    mu0 = counts.mean()
    var0 = counts.var()
    r0 = mu0**2 / max(var0 - mu0, 0.1)
    try:
        res = minimize(nb_loglik, x0=[mu0, r0], args=(counts,),
                       method='Nelder-Mead',
                       options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
        mu_hat, r_hat = res.x
        ll = -res.fun
        k = 2
        aic = 2*k - 2*ll
        return mu_hat, r_hat, ll, aic
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


# =============================================================================
# OVERDISPERSION TESTS
# =============================================================================
def run_overdispersion_tests(fp_counts, model_label):
    counts = fp_counts.astype(int)
    n = len(counts)
    mu = counts.mean()
    var = counts.var()
    disp_index = var / max(mu, 1e-9)

    # Test 1: Index of Dispersion chi-squared test
    chi2_stat = (n - 1) * disp_index
    p_chi2 = 1 - stats.chi2.cdf(chi2_stat, df=n - 1)

    # Test 2: Poisson vs NB likelihood ratio test
    mu_pois, ll_pois, aic_pois = fit_poisson(counts)
    mu_nb, r_nb, ll_nb, aic_nb = fit_negative_binomial(counts)

    lrt_stat = 2 * (ll_nb - ll_pois) if not np.isnan(ll_nb) else np.nan
    p_lrt    = 1 - stats.chi2.cdf(lrt_stat, df=1) if not np.isnan(lrt_stat) else np.nan

    result = {
        'model': model_label,
        'n_images': n,
        'mean_fp_per_image': round(mu, 4),
        'var_fp_per_image': round(var, 4),
        'dispersion_index': round(disp_index, 4),
        'chi2_stat': round(chi2_stat, 4),
        'p_chi2': round(p_chi2, 6),
        'overdispersed_chi2': p_chi2 < 0.05,
        'poisson_ll': round(ll_pois, 4),
        'poisson_aic': round(aic_pois, 4),
        'nb_mu': round(mu_nb, 4) if not np.isnan(mu_nb) else np.nan,
        'nb_r': round(r_nb, 4) if not np.isnan(r_nb) else np.nan,
        'nb_ll': round(ll_nb, 4) if not np.isnan(ll_nb) else np.nan,
        'nb_aic': round(aic_nb, 4) if not np.isnan(aic_nb) else np.nan,
        'lrt_stat': round(lrt_stat, 4) if not np.isnan(lrt_stat) else np.nan,
        'p_lrt': round(p_lrt, 6) if not np.isnan(p_lrt) else np.nan,
        'nb_preferred': aic_nb < aic_pois if not np.isnan(aic_nb) else False,
    }

    print(f"\n{'='*55}")
    print(f"{model_label} — Overdispersion Analysis")
    print(f"{'='*55}")
    print(f"  Mean FPs/image     : {mu:.2f}")
    print(f"  Variance           : {var:.2f}")
    print(f"  Dispersion index   : {disp_index:.4f}  (Poisson = 1.0)")
    print(f"  Chi2 test p-value  : {p_chi2:.6f}  {'(OVERDISPERSED)' if p_chi2 < 0.05 else '(not significant)'}")
    print(f"  Poisson AIC        : {aic_pois:.2f}")
    print(f"  NB AIC             : {aic_nb:.2f}  {'(NB preferred)' if aic_nb < aic_pois else '(Poisson preferred)'}")
    if not np.isnan(p_lrt):
        print(f"  LRT p-value (NB>P) : {p_lrt:.6f}")

    return result, counts


print("Loading predictions...")
df_a = pd.read_csv(pred_csv_a)
df_b = pd.read_csv(pred_csv_b)
df_a = df_a[df_a['conf'] >= CONF_THRESH]
df_b = df_b[df_b['conf'] >= CONF_THRESH]

fp_counts_a = df_a.groupby('image_id')['is_fp'].sum().values
fp_counts_b = df_b.groupby('image_id')['is_fp'].sum().values

res_a, counts_a = run_overdispersion_tests(fp_counts_a, 'Model_A')
res_b, counts_b = run_overdispersion_tests(fp_counts_b, 'Model_B')

pd.DataFrame([res_a, res_b]).to_csv(out_dir / "overdispersion_test_results.csv", index=False)

# =============================================================================
# PLOT — OBSERVED VS FITTED DISTRIBUTIONS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('FP Count per Image: Observed vs Poisson vs Negative Binomial', fontsize=12)

for ax, counts, res, label in [
    (axes[0], counts_a, res_a, 'Model A'),
    (axes[1], counts_b, res_b, 'Model B'),
]:
    max_count = min(int(counts.max()), 50)
    k_vals = np.arange(0, max_count + 1)
    observed_freq = np.array([(counts == k).sum() for k in k_vals])

    mu_p = res['mean_fp_per_image']
    pois_freq = stats.poisson.pmf(k_vals, mu_p) * len(counts)

    mu_nb = res.get('nb_mu', np.nan)
    r_nb  = res.get('nb_r', np.nan)

    ax.bar(k_vals, observed_freq, alpha=0.5, color='steelblue', label='Observed', width=0.8)
    ax.plot(k_vals, pois_freq, 'r-o', markersize=4, linewidth=1.5, label=f'Poisson (μ={mu_p:.1f})')

    if not (np.isnan(mu_nb) or np.isnan(r_nb)):
        p_nb = r_nb / (r_nb + mu_nb)
        nb_freq = stats.nbinom.pmf(k_vals, r_nb, p_nb) * len(counts)
        ax.plot(k_vals, nb_freq, 'g-s', markersize=4, linewidth=1.5,
                label=f'NegBin (μ={mu_nb:.1f}, r={r_nb:.2f})')

    ax.set_title(f'{label}  |  Dispersion index = {res["dispersion_index"]:.3f}')
    ax.set_xlabel('FP count per image')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "fp_count_distribution_fit.png", dpi=150, bbox_inches='tight')
print(f"\nDistribution fit plot saved to: {out_dir}/fp_count_distribution_fit.png")

print("\nPhase 4 — Overdispersion analysis complete.")
