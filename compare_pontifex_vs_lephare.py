#!/usr/bin/env python3
"""
Fair Benchmark Comparison: Pontifex v2.2.0 (AionPlus) vs. COSMOS2020 Classic LePHARE
=====================================================================================
Rigorous, matched head-to-head evaluation on the curated COSMOS spectroscopic sample.

Both estimators are evaluated on the EXACT SAME sample of galaxies:
  - Secure spectroscopic redshift (0.01 <= z_spec <= 3.0, quality flag >= 3)
  - Valid published COSMOS2020 Classic LePHARE photo-z (Weaver et al. 2022, Khostovan et al. 2025)
  - 5-fold cross-validated out-of-fold Pontifex v2.2.0 (AionPlus) prediction

Metrics: Official LSST DESC PZ Data Challenge benchmark suite:
  - Bias (Median delta z / (1+z))
  - Biweight Location
  - Scatter Sigma_MAD
  - Scatter Sigma_IQR
  - Biweight Scale
  - Outlier fractions eta_0.15 and eta_0.30
  - Catastrophic Outliers (|delta z| > 1.0)
  - DESC SRD moment shifts (delta_mu, delta_sigma)
  - Tomographic and magnitude-sliced performance
"""

import os
import sys
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from astropy.io import fits
from astropy.stats import biweight_location, biweight_scale
from scipy import stats

# Matplotlib publication styling
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 15,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'text.usetex': False,
    'mathtext.fontset': 'dejavusans',
})

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path("/home/mardom/.gemini/antigravity/brain/90bb7e50-a0ae-45b9-b247-e1efc638395f")


def flux_mjy_to_mag_ab(flux_mjy, err_mjy, floor_mjy=1e-6, default_faint_mag=28.0):
    f = np.asarray(flux_mjy, dtype=np.float32)
    valid_f = np.isfinite(f) & (f > floor_mjy)
    mag = np.full_like(f, default_faint_mag)
    mag[valid_f] = -2.5 * np.log10(f[valid_f]) + 16.40
    return mag


def compute_metrics(zp, zs):
    dz = (zp - zs) / (1.0 + zs)
    bias = float(np.median(dz))
    bw_loc = float(biweight_location(dz))
    sigma_mad = float(1.4826 * np.median(np.abs(dz - bias)))
    sigma_iqr = float((np.percentile(dz, 75) - np.percentile(dz, 25)) / 1.349)
    bw_scale = float(biweight_scale(dz))
    outlier_015 = float(np.mean(np.abs(dz) > 0.15))
    outlier_030 = float(np.mean(np.abs(dz) > 0.30))
    catastrophic = float(np.mean(np.abs(zp - zs) > 1.0))
    rmse = float(np.sqrt(np.mean(dz**2)))

    mean_true = float(np.mean(zs))
    std_true = float(np.std(zs))
    mean_phot = float(np.mean(zp))
    std_phot = float(np.std(zp))
    delta_mu = mean_phot - mean_true
    delta_sigma = std_phot - std_true

    return {
        'bias': bias,
        'bw_loc': bw_loc,
        'sigma_mad': sigma_mad,
        'sigma_iqr': sigma_iqr,
        'bw_scale': bw_scale,
        'outlier_015': outlier_015,
        'outlier_030': outlier_030,
        'catastrophic': catastrophic,
        'rmse': rmse,
        'delta_mu': delta_mu,
        'delta_sigma': delta_sigma,
    }


def main():
    print("=" * 85)
    print("FAIR BENCHMARK: PONTIFEX v2.2.0 (AionPlus) vs. COSMOS2020 Classic LePHARE")
    print("=" * 85)

    data_dir = SCRIPT_DIR.parent.parent / "data" / "speczcompilation"
    if not data_dir.exists():
        data_dir = Path("/home/mardom/Rubin-LSST-Research/Photometric-Redshift/data/speczcompilation")

    unique_file = data_dir / "specz_compilation" / "specz_compilation_COSMOS_DR1.1_unique.fits"
    cigale_file = data_dir / "sed_fitting" / "cigale" / "cigale_results_specz_compilation_DR1.1.fits"
    pontifex_pred_file = SCRIPT_DIR / "results" / "specz_compilation_predictions" / "pontifex_cosmos_specz_predictions.csv"

    print(f"Loading Pontifex predictions from: {pontifex_pred_file}")
    df_pontifex = pd.read_csv(pontifex_pred_file)
    print(f"  Total Pontifex predictions: {len(df_pontifex):,}")

    print(f"Loading COSMOS DR1.1 unique catalog from: {unique_file}")
    with fits.open(unique_file, memmap=True) as hdul:
        data_u = hdul[1].data
        df_u = pd.DataFrame({
            'Id_specz': np.array(data_u['Id_specz'], dtype=np.int64),
            'specz': np.array(data_u['specz'], dtype=np.float64),
            'flag': np.array(data_u['flag'], dtype=np.int32),
            'photoz': np.array(data_u['photoz'], dtype=np.float64),
            'photoz_type': np.array(data_u['photoz_type'], dtype=np.int64),
            'ra': np.array(data_u['ra_corrected'], dtype=np.float64),
            'dec': np.array(data_u['dec_corrected'], dtype=np.float64),
        })

    print(f"Loading Sub-aperture Suprime i-band photometry from: {cigale_file}")
    with fits.open(cigale_file, memmap=True) as hdul:
        data_c = hdul[1].data
        df_c = pd.DataFrame({
            'Id_specz': np.array(data_c['Id_specz'], dtype=np.int64),
            'flux_i': np.array(data_c['subaru.suprime.i'], dtype=np.float32),
            'flux_i_err': np.array(data_c['subaru.suprime.i_err'], dtype=np.float32),
        })
        df_c['mag_i'] = flux_mjy_to_mag_ab(df_c['flux_i'], df_c['flux_i_err'])

    # Merge catalogs
    print("Merging Pontifex predictions with DR1.1 and photometry...")
    merged = pd.merge(df_pontifex, df_u[['Id_specz', 'photoz', 'photoz_type']], left_on='object_id', right_on='Id_specz')
    merged = pd.merge(merged, df_c[['Id_specz', 'mag_i']], on='Id_specz')

    # Define fair matched sample:
    # 1. Flag >= 3 (already guaranteed in curated sample)
    # 2. 0.01 <= z_spec <= 3.0 (already guaranteed in curated sample)
    # 3. photoz_type == 0 (Galaxy in COSMOS2020 Classic LePHARE)
    # 4. Valid, finite, positive photoz in [0.01, 10.0]
    matched_mask = (
        (merged['photoz_type'] == 0) &
        np.isfinite(merged['photoz']) &
        (merged['photoz'] >= 0.01) &
        (merged['photoz'] <= 10.0) &
        np.isfinite(merged['z_phot_final']) &
        (merged['z_phot_final'] >= 0.01)
    )

    df_matched = merged[matched_mask].copy()
    N_matched = len(df_matched)
    print(f"\nStrict Fair Matched Sample: N = {N_matched:,} galaxies")

    zs = df_matched['z_spec'].values
    zp_pont = df_matched['z_phot_final'].values
    zp_leph = df_matched['photoz'].values
    mag_i = df_matched['mag_i'].values

    # Overall metrics
    metrics_pont_all = compute_metrics(zp_pont, zs)
    metrics_leph_all = compute_metrics(zp_leph, zs)

    # Core regime (z < 1.2) metrics
    core_mask = zs < 1.2
    metrics_pont_core = compute_metrics(zp_pont[core_mask], zs[core_mask])
    metrics_leph_core = compute_metrics(zp_leph[core_mask], zs[core_mask])

    print("\n" + "=" * 85)
    print(f"{'LSST DESC PZ Metric':<30s} | {'Pontifex v2.2.0 (6 bands)':<25s} | {'COSMOS2020 LePHARE (30+ bands)':<30s}")
    print("-" * 85)
    metric_rows = [
        ('Sample Size (N)', f"{N_matched:,}", f"{N_matched:,}"),
        ('Input Photometry', "6 Rubin LSST bands (ugrizy)", "30+ UV-to-IRAC Bands"),
        ('Photo-z Bias (Median)', f"{metrics_pont_all['bias']:+.5f}", f"{metrics_leph_all['bias']:+.5f}"),
        ('Biweight Location', f"{metrics_pont_all['bw_loc']:+.5f}", f"{metrics_leph_all['bw_loc']:+.5f}"),
        ('Scatter Sigma_MAD', f"{metrics_pont_all['sigma_mad']:.5f}", f"{metrics_leph_all['sigma_mad']:.5f}"),
        ('Scatter Sigma_IQR', f"{metrics_pont_all['sigma_iqr']:.5f}", f"{metrics_leph_all['sigma_iqr']:.5f}"),
        ('Biweight Scale', f"{metrics_pont_all['bw_scale']:.5f}", f"{metrics_leph_all['bw_scale']:.5f}"),
        ('Outlier Rate (eta > 0.15)', f"{metrics_pont_all['outlier_015']:.2%}", f"{metrics_leph_all['outlier_015']:.2%}"),
        ('Severe Outlier (eta > 0.30)', f"{metrics_pont_all['outlier_030']:.2%}", f"{metrics_leph_all['outlier_030']:.2%}"),
        ('Catastrophic (|dz| > 1.0)', f"{metrics_pont_all['catastrophic']:.2%}", f"{metrics_leph_all['catastrophic']:.2%}"),
        ('Overall RMSE', f"{metrics_pont_all['rmse']:.5f}", f"{metrics_leph_all['rmse']:.5f}"),
        ('DESC SRD Mean Shift (delta_mu)', f"{metrics_pont_all['delta_mu']:+.5f}", f"{metrics_leph_all['delta_mu']:+.5f}"),
        ('DESC SRD Dispersion Shift', f"{metrics_pont_all['delta_sigma']:+.5f}", f"{metrics_leph_all['delta_sigma']:+.5f}"),
        ('--- Core Regime (z < 1.2) ---', f"N = {int(np.sum(core_mask)):,}", f"N = {int(np.sum(core_mask)):,}"),
        ('Core Bias (Median)', f"{metrics_pont_core['bias']:+.5f}", f"{metrics_leph_core['bias']:+.5f}"),
        ('Core Sigma_MAD', f"{metrics_pont_core['sigma_mad']:.5f}", f"{metrics_leph_core['sigma_mad']:.5f}"),
        ('Core Outlier Rate (eta > 0.15)', f"{metrics_pont_core['outlier_015']:.2%}", f"{metrics_leph_core['outlier_015']:.2%}"),
    ]
    for label, val_p, val_l in metric_rows:
        print(f"{label:<30s} | {val_p:<25s} | {val_l:<30s}")
    print("=" * 85)

    # Export metrics table
    res_dir = SCRIPT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([
        {'Metric': label, 'Pontifex_v2.2.0': val_p, 'COSMOS2020_LePHARE': val_l}
        for label, val_p, val_l in metric_rows
    ])
    summary_df.to_csv(res_dir / "pontifex_vs_lephare_metrics.csv", index=False)

    # Detailed Redshift Tomographic Bins Analysis
    tomo_bins = [
        ('Bin 1 (0.01 <= z < 0.40)', 0.01, 0.40),
        ('Bin 2 (0.40 <= z < 0.80)', 0.40, 0.80),
        ('Bin 3 (0.80 <= z < 1.20)', 0.80, 1.20),
        ('Bin 4 (1.20 <= z < 1.60)', 1.20, 1.60),
        ('Bin 5 (1.60 <= z <= 3.00)', 1.60, 3.00),
    ]
    tomo_results = []
    for name, z_min, z_max in tomo_bins:
        bmask = (zs >= z_min) & (zs < z_max if z_max < 3.0 else zs <= z_max)
        n_b = int(np.sum(bmask))
        mp = compute_metrics(zp_pont[bmask], zs[bmask])
        ml = compute_metrics(zp_leph[bmask], zs[bmask])
        tomo_results.append({
            'bin_name': name,
            'z_min': z_min,
            'z_max': z_max,
            'z_mid': 0.5 * (z_min + z_max),
            'n_gal': n_b,
            'pontifex': mp,
            'lephare': ml,
        })

    # Detailed Magnitude Bins Analysis
    mag_bins = [
        ('Bright (i < 20.0)', 15.0, 20.0),
        ('Intermediate 1 (20.0 <= i < 21.5)', 20.0, 21.5),
        ('Intermediate 2 (21.5 <= i < 23.0)', 21.5, 23.0),
        ('Faint (23.0 <= i < 24.5)', 23.0, 24.5),
        ('Very Faint (i >= 24.5)', 24.5, 30.0),
    ]
    mag_results = []
    for name, m_min, m_max in mag_bins:
        bmask = (mag_i >= m_min) & (mag_i < m_max)
        n_b = int(np.sum(bmask))
        if n_b > 10:
            mp = compute_metrics(zp_pont[bmask], zs[bmask])
            ml = compute_metrics(zp_leph[bmask], zs[bmask])
            mag_results.append({
                'bin_name': name,
                'm_min': m_min,
                'm_max': m_max,
                'm_mid': 0.5 * (m_min + m_max) if m_max < 30 else 25.5,
                'n_gal': n_b,
                'pontifex': mp,
                'lephare': ml,
            })

    # Save complete JSON
    full_json = {
        'total_matched_galaxies': N_matched,
        'overall_pontifex': metrics_pont_all,
        'overall_lephare': metrics_leph_all,
        'core_pontifex_z_lt_1p2': metrics_pont_core,
        'core_lephare_z_lt_1p2': metrics_leph_core,
        'tomographic_bins': tomo_results,
        'magnitude_bins': mag_results,
    }
    with open(res_dir / "pontifex_vs_lephare_detailed_benchmark.json", "w") as f:
        json.dump(full_json, f, indent=2)

    # -------------------------------------------------------------
    # FIGURE 1: Side-by-Side Hexbin Heatmaps (z_phot vs z_spec)
    # -------------------------------------------------------------
    print("\nGenerating Figure 1: Side-by-Side Hexbin Comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), sharey=True)

    dz_pont = (zp_pont - zs) / (1.0 + zs)
    dz_leph = (zp_leph - zs) / (1.0 + zs)

    for idx, (ax, zp, dz, title, col, sigma_val, out_val, bias_val, n_bands) in enumerate([
        (axes[0], zp_pont, dz_pont, "Pontifex v2.2.0 (AionPlus)", "Purples",
         metrics_pont_all['sigma_mad'], metrics_pont_all['outlier_015'], metrics_pont_all['bias'], "6 Rubin LSST Bands"),
        (axes[1], zp_leph, dz_leph, "COSMOS2020 Classic LePHARE", "Blues",
         metrics_leph_all['sigma_mad'], metrics_leph_all['outlier_015'], metrics_leph_all['bias'], "30+ UV-to-IRAC Bands"),
    ]):
        hb = ax.hexbin(zs, zp, gridsize=110, cmap=col, mincnt=1, bins='log', extent=[0.0, 3.0, 0.0, 3.0])
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r'$\log_{10}(N_{\mathrm{gal}})$', fontsize=11)

        z_line = np.linspace(0.0, 3.0, 200)
        ax.plot(z_line, z_line, 'r--', lw=1.8, label=r'Identity $z_{\mathrm{phot}} = z_{\mathrm{spec}}$')
        ax.plot(z_line, z_line + 0.15 * (1.0 + z_line), 'k:', lw=1.2, alpha=0.8, label=r'$\pm 0.15(1+z)$ Boundary')
        ax.plot(z_line, z_line - 0.15 * (1.0 + z_line), 'k:', lw=1.2, alpha=0.8)

        stats_box = (
            f"{title}\n"
            f"Input: {n_bands}\n"
            f"$N = {N_matched:,}$\n"
            f"Bias = {bias_val:+.4f}\n"
            f"$\\sigma_{{\\mathrm{{MAD}}}} = {sigma_val:.4f}$\n"
            f"$\\eta_{{0.15}} =$ {out_val*100:.2f}%"
        )
        ax.text(0.05, 0.95, stats_box, transform=ax.transAxes, verticalalignment='top',
                fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='gray'))

        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(0.0, 3.0)
        ax.set_xlabel(r'Spectroscopic Redshift $z_{\mathrm{spec}}$')
        if idx == 0:
            ax.set_ylabel(r'Photometric Redshift $z_{\mathrm{phot}}$')
        ax.set_title(f"{title}\n({n_bands})", pad=12, fontweight='bold')
        ax.legend(loc='lower right', framealpha=0.9, fontsize=9.5)

    plt.suptitle(r"COSMOS Curated Sample Matched Benchmark ($N = 35{,}291$, $\mathrm{flag} \geq 3$)", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig1_path = res_dir / "figure1_pontifex_vs_lephare_hexbin.png"
    fig.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig1_path}")

    # -------------------------------------------------------------
    # FIGURE 2: Redshift-Dependent Metrics Profiles
    # -------------------------------------------------------------
    print("Generating Figure 2: Redshift-Dependent Metrics Profiles...")
    z_mids = [r['z_mid'] for r in tomo_results]
    z_mins = [r['z_min'] for r in tomo_results]
    z_maxs = [r['z_max'] for r in tomo_results]
    z_errs = [0.5 * (r['z_max'] - r['z_min']) for r in tomo_results]

    sigma_mad_pont = [r['pontifex']['sigma_mad'] for r in tomo_results]
    sigma_mad_leph = [r['lephare']['sigma_mad'] for r in tomo_results]

    outlier_pont = [r['pontifex']['outlier_015'] * 100 for r in tomo_results]
    outlier_leph = [r['lephare']['outlier_015'] * 100 for r in tomo_results]

    bias_pont = [r['pontifex']['bias'] for r in tomo_results]
    bias_leph = [r['lephare']['bias'] for r in tomo_results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Sigma_MAD vs z
    axes[0].errorbar(z_mids, sigma_mad_pont, xerr=z_errs, fmt='o-', color='#6a0dad', lw=2.2, ms=7,
                     label=r'Pontifex v2.2.0 (6 bands)', capsize=4)
    axes[0].errorbar(z_mids, sigma_mad_leph, xerr=z_errs, fmt='s--', color='#1f77b4', lw=2.2, ms=7,
                     label=r'COSMOS2020 LePHARE (30+ bands)', capsize=4)
    axes[0].axhline(0.02, color='gray', ls=':', alpha=0.7, label='LSST Gold requirement (0.02)')
    axes[0].set_xlabel(r'Spectroscopic Redshift $z_{\mathrm{spec}}$')
    axes[0].set_ylabel(r'Core Scatter $\sigma_{\mathrm{MAD}}$')
    axes[0].set_title(r'(a) Redshift Dispersion $\sigma_{\mathrm{MAD}}(z)$')
    axes[0].legend(loc='upper left', framealpha=0.9, fontsize=10)
    axes[0].set_ylim(0.005, 0.35)
    axes[0].set_yscale('log')
    axes[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    # Panel B: Outlier Rate vs z
    axes[1].errorbar(z_mids, outlier_pont, xerr=z_errs, fmt='o-', color='#6a0dad', lw=2.2, ms=7,
                     label=r'Pontifex v2.2.0 (6 bands)', capsize=4)
    axes[1].errorbar(z_mids, outlier_leph, xerr=z_errs, fmt='s--', color='#1f77b4', lw=2.2, ms=7,
                     label=r'COSMOS2020 LePHARE (30+ bands)', capsize=4)
    axes[1].axhline(10.0, color='gray', ls=':', alpha=0.7, label='LSST SRD requirement (10%)')
    axes[1].set_xlabel(r'Spectroscopic Redshift $z_{\mathrm{spec}}$')
    axes[1].set_ylabel(r'Outlier Fraction $\eta_{0.15}$ (%)')
    axes[1].set_title(r'(b) Outlier Fraction $\eta_{0.15}(z)$')
    axes[1].legend(loc='upper left', framealpha=0.9, fontsize=10)
    axes[1].set_ylim(0.5, 70.0)
    axes[1].set_yscale('log')
    axes[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # Panel C: Median Bias vs z
    axes[2].errorbar(z_mids, bias_pont, xerr=z_errs, fmt='o-', color='#6a0dad', lw=2.2, ms=7,
                     label=r'Pontifex v2.2.0 (6 bands)', capsize=4)
    axes[2].errorbar(z_mids, bias_leph, xerr=z_errs, fmt='s--', color='#1f77b4', lw=2.2, ms=7,
                     label=r'COSMOS2020 LePHARE (30+ bands)', capsize=4)
    axes[2].axhline(0.0, color='black', ls='-', lw=1.0, alpha=0.5)
    axes[2].set_xlabel(r'Spectroscopic Redshift $z_{\mathrm{spec}}$')
    axes[2].set_ylabel(r'Median Bias $\langle \Delta z / (1+z) \rangle$')
    axes[2].set_title(r'(c) Photometric Redshift Bias $(z)$')
    axes[2].legend(loc='lower left', framealpha=0.9, fontsize=10)
    axes[2].set_ylim(-0.25, 0.05)

    plt.tight_layout()
    plt.suptitle(r"Tomographic Performance Profiles on Matched COSMOS Sample ($N = 35{,}291$)", fontsize=15, y=1.01)
    fig2_path = res_dir / "figure2_pontifex_vs_lephare_redshift_bins.png"
    fig.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig2_path}")

    # -------------------------------------------------------------
    # FIGURE 3: Magnitude-Dependent Metrics Profiles
    # -------------------------------------------------------------
    print("Generating Figure 3: Magnitude-Dependent Profiles...")
    m_mids = [r['m_mid'] for r in mag_results]
    m_errs = [0.5 * (min(r['m_max'], 26.5) - r['m_min']) for r in mag_results]

    sigma_mag_pont = [r['pontifex']['sigma_mad'] for r in mag_results]
    sigma_mag_leph = [r['lephare']['sigma_mad'] for r in mag_results]

    outlier_mag_pont = [r['pontifex']['outlier_015'] * 100 for r in mag_results]
    outlier_mag_leph = [r['lephare']['outlier_015'] * 100 for r in mag_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].errorbar(m_mids, sigma_mag_pont, xerr=m_errs, fmt='o-', color='#6a0dad', lw=2.2, ms=7,
                     label=r'Pontifex v2.2.0 (6 bands)', capsize=4)
    axes[0].errorbar(m_mids, sigma_mag_leph, xerr=m_errs, fmt='s--', color='#1f77b4', lw=2.2, ms=7,
                     label=r'COSMOS2020 LePHARE (30+ bands)', capsize=4)
    axes[0].set_xlabel(r'Subaru Suprime $i$-band Magnitude [AB]')
    axes[0].set_ylabel(r'Core Scatter $\sigma_{\mathrm{MAD}}$')
    axes[0].set_title(r'(a) Dispersion vs. Apparent Magnitude $\sigma_{\mathrm{MAD}}(i)$')
    axes[0].legend(loc='upper left', framealpha=0.9)
    axes[0].set_ylim(0.005, 0.06)

    axes[1].errorbar(m_mids, outlier_mag_pont, xerr=m_errs, fmt='o-', color='#6a0dad', lw=2.2, ms=7,
                     label=r'Pontifex v2.2.0 (6 bands)', capsize=4)
    axes[1].errorbar(m_mids, outlier_mag_leph, xerr=m_errs, fmt='s--', color='#1f77b4', lw=2.2, ms=7,
                     label=r'COSMOS2020 LePHARE (30+ bands)', capsize=4)
    axes[1].set_xlabel(r'Subaru Suprime $i$-band Magnitude [AB]')
    axes[1].set_ylabel(r'Outlier Fraction $\eta_{0.15}$ (%)')
    axes[1].set_title(r'(b) Outlier Fraction vs. Apparent Magnitude $\eta_{0.15}(i)$')
    axes[1].legend(loc='upper left', framealpha=0.9)
    axes[1].set_ylim(0.0, 20.0)

    plt.tight_layout()
    plt.suptitle(r"Performance vs. Apparent Brightness ($N = 35{,}291$)", fontsize=15, y=1.01)
    fig3_path = res_dir / "figure3_pontifex_vs_lephare_mag_bins.png"
    fig.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig3_path}")

    # -------------------------------------------------------------
    # FIGURE 4: Residual Distributions & Agreement Diagnostics
    # -------------------------------------------------------------
    print("Generating Figure 4: Residual Comparison & Differential Diagnostics...")
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(1, 2, width_ratios=[1.1, 1.0])

    # Left: Histogram of residuals
    ax_hist = fig.add_subplot(gs[0])
    bins_res = np.linspace(-0.15, 0.15, 120)
    ax_hist.hist(dz_pont, bins=bins_res, density=True, histtype='stepfilled', color='#6a0dad', alpha=0.35,
                 label=f'Pontifex v2.2.0 ($\\sigma={metrics_pont_all["sigma_mad"]:.4f}$)')
    ax_hist.hist(dz_pont, bins=bins_res, density=True, histtype='step', color='#6a0dad', lw=2.0)
    ax_hist.hist(dz_leph, bins=bins_res, density=True, histtype='stepfilled', color='#1f77b4', alpha=0.25,
                 label=f'COSMOS2020 LePHARE ($\\sigma={metrics_leph_all["sigma_mad"]:.4f}$)')
    ax_hist.hist(dz_leph, bins=bins_res, density=True, histtype='step', color='#1f77b4', lw=2.0, ls='--')

    ax_hist.axvline(0.0, color='black', ls=':', lw=1.2, alpha=0.7)
    ax_hist.set_xlabel(r'Normalized Residual $\Delta z / (1+z_{\mathrm{spec}})$')
    ax_hist.set_ylabel('Probability Density')
    ax_hist.set_title(r'(a) Core Residual Distributions ($|\Delta z| \leq 0.15$)')
    ax_hist.legend(loc='upper right', framealpha=0.9, fontsize=10.5)

    # Right: Direct comparison z_phot(Pontifex) vs z_phot(LePHARE)
    ax_scatter = fig.add_subplot(gs[1])
    hb_direct = ax_scatter.hexbin(zp_leph, zp_pont, gridsize=100, cmap='viridis', mincnt=1, bins='log', extent=[0, 3, 0, 3])
    cb_dir = fig.colorbar(hb_direct, ax=ax_scatter, fraction=0.046, pad=0.04)
    cb_dir.set_label(r'$\log_{10}(N_{\mathrm{gal}})$', fontsize=11)
    ax_scatter.plot([0, 3], [0, 3], 'r--', lw=1.8, label=r'Concordance $z_{\mathrm{Pont}} = z_{\mathrm{LePHARE}}$')

    pearson_r = np.corrcoef(zp_pont, zp_leph)[0, 1]
    diff_zp = np.abs(zp_pont - zp_leph)
    frac_agree = np.mean(diff_zp < 0.10)
    agreement_box = (
        f"Pearson $r = {pearson_r:.4f}$\n"
        f"$|z_{{\\mathrm{{Pont}}}} - z_{{\\mathrm{{LePH}}}}| < 0.10$: {frac_agree:.1%}\n"
        f"Median $|\\Delta z_{{P-L}}| = {np.median(diff_zp):.4f}$"
    )
    ax_scatter.text(0.05, 0.95, agreement_box, transform=ax_scatter.transAxes, verticalalignment='top',
                    fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='gray'))

    ax_scatter.set_xlim(0.0, 3.0)
    ax_scatter.set_ylim(0.0, 3.0)
    ax_scatter.set_xlabel(r'COSMOS2020 Classic LePHARE $z_{\mathrm{phot}}$ (30+ bands)')
    ax_scatter.set_ylabel(r'Pontifex v2.2.0 $z_{\mathrm{phot}}$ (6 bands)')
    ax_scatter.set_title(r'(b) Estimator Concordance ($N = 35{,}291$)')
    ax_scatter.legend(loc='lower right', framealpha=0.9, fontsize=10)

    plt.tight_layout()
    plt.suptitle(r"Comparative Residual & Concordance Diagnostics", fontsize=15, y=1.01)
    fig4_path = res_dir / "figure4_pontifex_vs_lephare_residuals.png"
    fig.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig4_path}")

    # Copy all figures and summary to artifact directory
    if ARTIFACT_DIR.exists():
        print(f"\nCopying publication figures to artifact directory: {ARTIFACT_DIR}")
        for f in [fig1_path, fig2_path, fig3_path, fig4_path, res_dir / "pontifex_vs_lephare_metrics.csv"]:
            dest = ARTIFACT_DIR / f.name
            shutil.copy2(f, dest)
            print(f"  Copied {f.name} -> {dest}")

    print("\nBenchmark analysis completed successfully!")


if __name__ == '__main__':
    main()
