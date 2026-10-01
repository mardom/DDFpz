#!/usr/bin/env python3
"""
Pontifex Photometric Redshift PDF Pipeline - COSMOS Spectroscopic Compilation
=============================================================================
Application of the unified Pontifex (v2.2.0 / AionPlus) architecture:
- Data ingestion from COSMOS spec-z compilation DR1.1 & CIGALE photometry
- Curated secure spectroscopic sample (flag >= 3, 0.01 <= z <= 3.0)
- 44-D Pontifex feature space engineering with input feature guard
- Unified Committee of Experts (RAIL BPZ, FlexZBoost, GPz, kNN, NN1, NN2 + AION-PZ + SOM)
- Dynamic Mixture of Experts (MoE) weighting
- 5-Fold Stratified Cross-Validation Out-Of-Fold (OOF) evaluation
- Spatial Clustering Expectation-Maximization (EM) loop for blend candidates
- Complete LSST DESC PZ Data Challenge benchmark metrics
- Publication-quality diagnostic figures (hexbin zphot vs zspec, PIT, N(z), blend deconvolution)
"""

import os
import sys
import time
import argparse
import tempfile
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from astropy.table import Table
from astropy.stats import biweight_location, biweight_scale
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

# Setup repository paths
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir / 'src'))

import pontifex
from pontifex.core import (
    sanitize_input_catalog,
    extract_features,
    compute_moments_bias,
    compute_distribution_moments,
    LSST_BANDS,
)
from pontifex.pz.estimators import CommitteeOfExperts, Z_CENTERS, Z_GRID

_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)


def find_data_paths():
    candidates = [
        script_dir.parent.parent / "data" / "speczcompilation",
        script_dir.parent / "data" / "speczcompilation",
        Path("../../data/speczcompilation").resolve(),
        Path("/home/mardom/Rubin-LSST-Research/Photometric-Redshift/data/speczcompilation"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not locate speczcompilation directory in {candidates}")


def flux_mjy_to_mag_ab(flux_mjy, err_mjy, floor_mjy=1e-6, default_faint_mag=28.0):
    f = np.asarray(flux_mjy, dtype=np.float32)
    err = np.asarray(err_mjy, dtype=np.float32)

    valid_f = np.isfinite(f) & (f > floor_mjy)
    mag = np.full_like(f, default_faint_mag)
    mag[valid_f] = -2.5 * np.log10(f[valid_f]) + 16.40

    mag_err = np.full_like(err, 1.0)
    valid_err = valid_f & np.isfinite(err) & (err > 0.0)
    mag_err[valid_err] = 1.08574 * (err[valid_err] / f[valid_err])
    mag_err = np.clip(mag_err, 0.01, 2.50)

    return mag, mag_err


def radec_to_unit_cartesian(ra_deg, dec_deg):
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    return np.column_stack([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad)
    ])


def compute_desc_challenge_metrics(z_phot, z_spec, pdfs, z_eval=Z_CENTERS):
    valid = np.isfinite(z_phot) & np.isfinite(z_spec)
    zp, zs, p_sub = z_phot[valid], z_spec[valid], pdfs[valid]
    N = len(zp)

    dz = (zp - zs) / (1.0 + zs)
    bias = float(np.median(dz))
    bw_loc = float(biweight_location(dz))
    sigma_mad = float(1.4826 * np.median(np.abs(dz - bias)))
    sigma_iqr = float((np.percentile(dz, 75) - np.percentile(dz, 25)) / 1.349)
    bw_scale = float(biweight_scale(dz))
    outlier_015 = float(np.mean(np.abs(dz) > 0.15))
    outlier_030 = float(np.mean(np.abs(dz) > 0.30))

    # Distribution / PIT metrics
    dz_eval = z_eval[1] - z_eval[0]
    cdf_mat = np.cumsum(p_sub, axis=1) * dz_eval
    cdf_mat /= np.maximum(cdf_mat[:, -1:], 1e-12)

    pit_vals = np.array([np.interp(zs[i], z_eval, cdf_mat[i]) for i in range(N)])
    pit_vals = np.clip(pit_vals, 0.0, 1.0)

    mean_pit = float(np.mean(pit_vals))
    var_pit = float(np.var(pit_vals))
    ks_stat, _ = stats.kstest(pit_vals, 'uniform')
    sorted_pit = np.sort(pit_vals)
    i_vec = np.arange(1, N + 1)
    cvm_stat = float(1.0 / (12.0 * N) + np.sum((sorted_pit - (2.0 * i_vec - 1.0) / (2.0 * N)) ** 2))
    pit_outliers = float(np.mean((pit_vals < 1e-4) | (pit_vals > 1.0 - 1e-4)))

    # CRPS
    theta_mat = (z_eval[None, :] >= zs[:, None]).astype(float)
    crps_array = np.sum((cdf_mat - theta_mat) ** 2, axis=1) * dz_eval
    mean_crps = float(np.mean(crps_array))

    # Stacked N(z) & Wasserstein distance
    stacked_pz = np.mean(p_sub, axis=0)
    stacked_norm = stacked_pz / (np.sum(stacked_pz) * dz_eval)
    hist_true, _ = np.histogram(zs, bins=np.linspace(z_eval[0] - 0.5*dz_eval, z_eval[-1] + 0.5*dz_eval, len(z_eval) + 1), density=True)
    w1_dist = float(stats.wasserstein_distance(z_eval, z_eval, u_weights=stacked_norm, v_weights=hist_true))

    # DESC SRD moment shifts
    mean_true = float(np.mean(zs))
    std_true = float(np.std(zs))
    mean_phot = float(np.sum(z_eval * stacked_norm * dz_eval))
    std_phot = float(np.sqrt(np.sum((z_eval - mean_phot)**2 * stacked_norm * dz_eval)))
    delta_mu = mean_phot - mean_true
    delta_sigma = std_phot - std_true

    return {
        'Bias (Median)': bias,
        'Biweight Location': bw_loc,
        'Sigma_MAD': sigma_mad,
        'Sigma_IQR': sigma_iqr,
        'Biweight Scale': bw_scale,
        'Outlier Rate (>0.15)': outlier_015,
        'Severe Outlier (>0.30)': outlier_030,
        'Mean PIT': mean_pit,
        'Var(PIT)': var_pit,
        'PIT KS Statistic': ks_stat,
        'Cramer-von Mises (CvM)': cvm_stat,
        'PIT Outlier Rate': pit_outliers,
        'Mean CRPS': mean_crps,
        'Wasserstein W1': w1_dist,
        'Delta_mu (DESC SRD)': delta_mu,
        'Delta_sigma (DESC SRD)': delta_sigma,
        'pit_values': pit_vals,
        'residuals': dz,
    }


def main():
    parser = argparse.ArgumentParser(description="Pontifex COSMOS Spec-Z Compilation 5-Fold Cross Validation")
    parser.add_argument('--folds', type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument('--max-samples', type=int, default=None, help="Maximum number of curated galaxies to process (default: all)")
    parser.add_argument('--em-iterations', type=int, default=4, help="Number of spatial EM iterations for blends (default: 4)")
    args = parser.parse_args()

    print("=" * 85)
    print(f"PONTIFEX PHOTO-Z PIPELINE (v{pontifex.__version__} / AionPlus) - COSMOS SPEC-Z 5-FOLD CV")
    print("=" * 85)

    t_start = time.time()

    # 1. Load Data
    data_dir = find_data_paths()
    print(f"Data directory: {data_dir}")
    unique_file = data_dir / "specz_compilation" / "specz_compilation_COSMOS_DR1.1_unique.fits"
    cigale_file = data_dir / "sed_fitting" / "cigale" / "cigale_results_specz_compilation_DR1.1.fits"

    print("Loading FITS catalogs...")
    unique_table = Table.read(unique_file)
    cigale_table = Table.read(cigale_file)

    # Secure spectroscopic filter: flag >= 3 and 0.01 <= specz <= 3.0
    secure_mask = (unique_table['flag'] >= 3) & (unique_table['specz'] >= 0.01) & (unique_table['specz'] <= 3.0)
    unique_secure = unique_table[secure_mask]
    print(f"Secure spectroscopic records (flag >= 3, 0.01 <= z <= 3.0): {len(unique_secure):,}")

    unique_lookup = {
        row['Id_specz']: (float(row['ra_corrected']), float(row['dec_corrected']), int(row['flag']))
        for row in unique_secure
    }

    matched_indices = [i for i, row in enumerate(cigale_table) if row['Id_specz'] in unique_lookup]
    cigale_matched = cigale_table[matched_indices]
    print(f"Successfully matched secure galaxies with CIGALE photometry: {len(cigale_matched):,}")

    # 2. Photometric Calibration & Conversion to AB Magnitudes
    filter_map = {
        'u': ('cfht.megacam.u', 'cfht.megacam.u_err'),
        'g': ('subaru.suprime.g', 'subaru.suprime.g_err'),
        'r': ('subaru.suprime.r', 'subaru.suprime.r_err'),
        'i': ('subaru.suprime.i', 'subaru.suprime.i_err'),
        'z': ('subaru.suprime.z', 'subaru.suprime.z_err'),
        'y': ('subaru.suprime.Y', 'subaru.suprime.Y_err'),
    }

    catalog_raw = {
        'object_id': np.array(cigale_matched['Id_specz'], dtype=np.int64),
        'redshift': np.array(cigale_matched['specz'], dtype=np.float64),
        'ra': np.array([unique_lookup[id_][0] for id_ in cigale_matched['Id_specz']], dtype=np.float64),
        'dec': np.array([unique_lookup[id_][1] for id_ in cigale_matched['Id_specz']], dtype=np.float64),
        'specz_flag': np.array([unique_lookup[id_][2] for id_ in cigale_matched['Id_specz']], dtype=np.int32),
    }

    for band, (f_col, err_col) in filter_map.items():
        m, me = flux_mjy_to_mag_ab(cigale_matched[f_col], cigale_matched[err_col])
        catalog_raw[f"mag_{band}_lsst"] = m
        catalog_raw[f"mag_{band}_lsst_err"] = me

    # Feature Guard & Input Protection
    sanitized_catalog, guard_report = sanitize_input_catalog(catalog_raw, raise_warnings=False)
    print("\n--- PONTIFEX FEATURE GUARD REPORT ---")
    for k, v in guard_report.items():
        print(f"  {k:30s}: {v}")

    # Subsample if requested
    n_total = len(sanitized_catalog['redshift'])
    if args.max_samples is not None and args.max_samples < n_total:
        print(f"\nSubsampling catalog from {n_total:,} to {args.max_samples:,} records (random_state=42)...")
        np.random.seed(42)
        idx_sub = np.random.choice(n_total, size=args.max_samples, replace=False)
        catalog = {k: v[idx_sub] for k, v in sanitized_catalog.items()}
    else:
        catalog = sanitized_catalog

    N = len(catalog['redshift'])
    print(f"\nProceeding with complete curated sample of {N:,} galaxies.")
    print(f"Redshift: min={catalog['redshift'].min():.3f}, max={catalog['redshift'].max():.3f}, median={np.median(catalog['redshift']):.3f}")

    bands = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    ref_band = 'mag_i_lsst'
    n_grid = len(Z_CENTERS)
    dz = Z_CENTERS[1] - Z_CENTERS[0]

    # Pre-extract physical feature representation for blend detection
    print("\nExtracting physical feature matrix for 3D PCA outlier detection...")
    X_features = extract_features(catalog)
    scaler_global = StandardScaler()
    X_features_scaled = scaler_global.fit_transform(X_features)

    # 3. 5-Fold Stratified Cross-Validation
    n_splits = args.folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_pdfs_moe = np.zeros((N, n_grid), dtype=np.float32)
    oof_pdfs_calib = np.zeros((N, n_grid), dtype=np.float32)
    oof_is_blend = np.zeros(N, dtype=bool)

    print("\n" + "=" * 85)
    print(f"EXECUTING {n_splits}-FOLD CROSS-VALIDATION WITH RAIL EXPERTS + AION-PZ + EM BLEND LOOP")
    print("=" * 85)

    for fold, (tr_idx, te_idx) in enumerate(kf.split(catalog['redshift'])):
        t_fold = time.time()
        print(f"\n>>> [FOLD {fold + 1}/{n_splits}] Train: {len(tr_idx):,} | Test: {len(te_idx):,} galaxies")

        tr_dict = {k: catalog[k][tr_idx] for k in catalog}
        te_dict = {k: catalog[k][te_idx] for k in catalog}

        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                # Fit CommitteeOfExperts (RAIL BPZ, FlexZBoost, GPz, kNN, NN1, NN2 + AION-PZ + SOM)
                committee = CommitteeOfExperts(is_ci=True)
                committee.fit(tr_dict, bands=bands, ref_band=ref_band, is_roman=False)

                # Predict MoE holdout PDFs
                preds_fold = committee.predict(te_dict)
                oof_pdfs_moe[te_idx] = preds_fold

                # Blend candidate detection via 3D PCA Mahalanobis distance + PDF variance
                pca_3d = PCA(n_components=3, random_state=42).fit(X_features_scaled[tr_idx])
                X_te_pca = pca_3d.transform(X_features_scaled[te_idx])
                pca_cov_diag = np.var(X_te_pca, axis=0) + 1e-10
                mahalanobis_sq = np.sum((X_te_pca - np.mean(X_te_pca, axis=0)) ** 2 / pca_cov_diag, axis=1)
                is_pca_outlier = mahalanobis_sq > 7.81

                z_mode_fold = Z_CENTERS[np.argmax(preds_fold, axis=1)]
                diff_grid = Z_CENTERS[None, :] - z_mode_fold[:, None]
                pdf_stds_fold = np.sqrt(np.sum((diff_grid ** 2) * preds_fold, axis=1) * dz)
                is_broad_pdf = pdf_stds_fold > 0.15

                is_blend_fold = is_pca_outlier | is_broad_pdf
                oof_is_blend[te_idx] = is_blend_fold
                n_blends_fold = int(np.sum(is_blend_fold))
                print(f"  Flagged {n_blends_fold:,} / {len(te_idx):,} blend candidates ({n_blends_fold / len(te_idx):.1%})")

                # Spatial EM Clustering Loop for Blend Candidates
                pdf_calib_fold = preds_fold.copy()
                if n_blends_fold > 0:
                    ref_coords = radec_to_unit_cartesian(tr_dict['ra'], tr_dict['dec'])
                    unk_coords = radec_to_unit_cartesian(te_dict['ra'], te_dict['dec'])
                    spatial_tree = cKDTree(ref_coords)

                    theta_max_deg = 2.5 / 60.0
                    r_chord_max = 2.0 * np.sin(np.radians(theta_max_deg) / 2.0)
                    sigma_theta_chord = r_chord_max / 2.0

                    blend_indices = np.where(is_blend_fold)[0]
                    neighbor_indices_list = spatial_tree.query_ball_point(unk_coords[blend_indices], r=r_chord_max)

                    em_iterations = args.em_iterations
                    learning_rate = 0.20
                    for iteration in range(em_iterations):
                        rms_diffs = []
                        for idx_local, gal_idx in enumerate(blend_indices):
                            neighbors = neighbor_indices_list[idx_local]
                            if len(neighbors) < 4:
                                continue

                            z_neighbors = tr_dict['redshift'][neighbors]
                            coords_neighbors = ref_coords[neighbors]
                            dists_chord = np.linalg.norm(coords_neighbors - unk_coords[gal_idx], axis=1)
                            spatial_weights = np.exp(-0.5 * (dists_chord / sigma_theta_chord) ** 2)

                            diff_neigh = Z_CENTERS[None, :] - z_neighbors[:, None]
                            spatial_L = np.sum(spatial_weights[:, None] * np.exp(-0.5 * (diff_neigh / 0.05) ** 2), axis=0)
                            spatial_L = np.maximum(spatial_L, 1e-4)
                            spatial_L /= np.sum(spatial_L)

                            old_pdf = pdf_calib_fold[gal_idx]
                            new_pdf = old_pdf * (spatial_L ** learning_rate)
                            new_pdf = gaussian_filter1d(new_pdf, sigma=0.02 / dz)
                            integ = np.sum(new_pdf) * dz
                            if integ > 0:
                                new_pdf /= integ

                            rms_diff = np.sqrt(np.mean((new_pdf - old_pdf) ** 2))
                            rms_diffs.append(rms_diff)
                            pdf_calib_fold[gal_idx] = new_pdf

                        avg_rms = np.mean(rms_diffs) if rms_diffs else 0.0
                        if iteration == em_iterations - 1:
                            print(f"  [EM Final Convergence] Mean PDF delta RMS across blends: {avg_rms:.2e}")

                oof_pdfs_calib[te_idx] = pdf_calib_fold

            finally:
                os.chdir(orig_cwd)

        duration_fold = time.time() - t_fold
        print(f"✓ Fold {fold + 1} completed in {duration_fold:.1f}s")

    # 4. Global Out-Of-Fold Evaluation
    print("\n" + "=" * 85)
    print("COMPUTING GLOBAL OUT-OF-FOLD (OOF) DESC PZ CHALLENGE METRICS")
    print("=" * 85)

    z_true_all = catalog['redshift']
    z_mode_moe = Z_CENTERS[np.argmax(oof_pdfs_moe, axis=1)]
    z_mode_calib = Z_CENTERS[np.argmax(oof_pdfs_calib, axis=1)]

    metrics_moe = compute_desc_challenge_metrics(z_mode_moe, z_true_all, oof_pdfs_moe)
    metrics_calib = compute_desc_challenge_metrics(z_mode_calib, z_true_all, oof_pdfs_calib)

    comparison_df = pd.DataFrame({
        'DESC Metric': [
            'Photo-z Bias (Median)',
            'Biweight Location',
            'Scatter Sigma_MAD',
            'Scatter Sigma_IQR',
            'Biweight Scale',
            'Outlier Rate (eta > 0.15)',
            'Severe Outlier (eta > 0.30)',
            'Mean PIT',
            'Var(PIT)',
            'PIT KS Distance (D_KS)',
            'Cramer-von Mises (CvM)',
            'PIT Outlier Rate',
            'Mean CRPS',
            'Wasserstein Distance (W1)',
            'Mean Shift (delta_mu)',
            'Dispersion Shift (delta_sigma)',
        ],
        'Tier 2: 5-Fold OOF MoE': [
            f"{metrics_moe['Bias (Median)']:+.5f}",
            f"{metrics_moe['Biweight Location']:+.5f}",
            f"{metrics_moe['Sigma_MAD']:.5f}",
            f"{metrics_moe['Sigma_IQR']:.5f}",
            f"{metrics_moe['Biweight Scale']:.5f}",
            f"{metrics_moe['Outlier Rate (>0.15)']:.2%}",
            f"{metrics_moe['Severe Outlier (>0.30)']:.2%}",
            f"{metrics_moe['Mean PIT']:.4f}",
            f"{metrics_moe['Var(PIT)']:.4f}",
            f"{metrics_moe['PIT KS Statistic']:.5f}",
            f"{metrics_moe['Cramer-von Mises (CvM)']:.2f}",
            f"{metrics_moe['PIT Outlier Rate']:.2%}",
            f"{metrics_moe['Mean CRPS']:.5f}",
            f"{metrics_moe['Wasserstein W1']:.5f}",
            f"{metrics_moe['Delta_mu (DESC SRD)']:+.5f}",
            f"{metrics_moe['Delta_sigma (DESC SRD)']:+.5f}",
        ],
        'Tier 3: 5-Fold OOF MoE + EM Blends': [
            f"{metrics_calib['Bias (Median)']:+.5f}",
            f"{metrics_calib['Biweight Location']:+.5f}",
            f"{metrics_calib['Sigma_MAD']:.5f}",
            f"{metrics_calib['Sigma_IQR']:.5f}",
            f"{metrics_calib['Biweight Scale']:.5f}",
            f"{metrics_calib['Outlier Rate (>0.15)']:.2%}",
            f"{metrics_calib['Severe Outlier (>0.30)']:.2%}",
            f"{metrics_calib['Mean PIT']:.4f}",
            f"{metrics_calib['Var(PIT)']:.4f}",
            f"{metrics_calib['PIT KS Statistic']:.5f}",
            f"{metrics_calib['Cramer-von Mises (CvM)']:.2f}",
            f"{metrics_calib['PIT Outlier Rate']:.2%}",
            f"{metrics_calib['Mean CRPS']:.5f}",
            f"{metrics_calib['Wasserstein W1']:.5f}",
            f"{metrics_calib['Delta_mu (DESC SRD)']:+.5f}",
            f"{metrics_calib['Delta_sigma (DESC SRD)']:+.5f}",
        ],
    })

    print(comparison_df.to_string(index=False))

    # Save metrics table
    res_dir = script_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(res_dir / "specz_compilation_metrics.csv", index=False)

    metrics_json = {
        'total_galaxies': N,
        'cv_folds': n_splits,
        'blend_candidates': int(np.sum(oof_is_blend)),
        'moe': {k: float(v) for k, v in metrics_moe.items() if not isinstance(v, np.ndarray)},
        'moe_em': {k: float(v) for k, v in metrics_calib.items() if not isinstance(v, np.ndarray)},
    }
    with open(res_dir / "specz_compilation_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    # 5. Save Out-Of-Fold Predictions
    pred_dir = res_dir / "specz_compilation_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving predictions and PDFs to {pred_dir}...")
    pred_df = pd.DataFrame({
        'object_id': catalog['object_id'],
        'ra': catalog['ra'],
        'dec': catalog['dec'],
        'z_spec': z_true_all,
        'z_phot_moe': z_mode_moe,
        'z_phot_final': z_mode_calib,
        'dz_final': (z_mode_calib - z_true_all) / (1.0 + z_true_all),
        'is_blend_candidate': oof_is_blend,
    })
    pred_df.to_csv(pred_dir / "pontifex_cosmos_specz_predictions.csv", index=False)

    np.savez_compressed(
        pred_dir / "pontifex_cosmos_specz_pdfs.npz",
        z_grid=Z_CENTERS,
        pdfs_moe=oof_pdfs_moe,
        pdfs_calib=oof_pdfs_calib,
        object_id=catalog['object_id'],
        z_spec=z_true_all
    )
    print("✓ Predictions and compressed PDFs archived.")

    # 6. Generate Publication-Quality Visualizations
    plot_dir = res_dir / "specz_compilation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating publication diagnostic figures into {plot_dir}...")

    # Figure 1: 2D Density Heatmap / Hexbin
    fig, ax = plt.subplots(figsize=(8.8, 7.8), dpi=300)
    hb = ax.hexbin(
        z_true_all, z_mode_calib,
        gridsize=110,
        cmap='inferno',
        bins='log',
        mincnt=1,
        extent=[0, 3, 0, 3]
    )
    z_line = np.linspace(0, 3, 300)
    ax.plot(z_line, z_line, 'w--', lw=1.8, label=r'$z_{\rm phot} = z_{\rm spec}$')
    ax.plot(z_line, z_line + 0.15 * (1 + z_line), color='#00e5ff', ls=':', lw=1.5, label=r'$\pm 0.15(1 + z_{\rm spec})$ envelopes')
    ax.plot(z_line, z_line - 0.15 * (1 + z_line), color='#00e5ff', ls=':', lw=1.5)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xlabel(r'Spectroscopic Redshift $z_{\rm spec}$', fontsize=13)
    ax.set_ylabel(r'Photometric Redshift $z_{\rm phot}$ (Mode)', fontsize=13)
    ax.set_title(r'$\mathbf{Pontifex\ v2.2.0\ (AionPlus):}$ 5-Fold OOF COSMOS Sample', fontsize=14, pad=12)

    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label(r'$\log_{10}(N_{\rm galaxies})$ per hexbin', fontsize=12)

    info_lines = [
        f"Bias (Median) = {metrics_calib['Bias (Median)']:+.4f}",
        rf"$\sigma_{{\rm MAD}} = {metrics_calib['Sigma_MAD']:.4f}$",
        rf"$\sigma_{{\rm BW}} = {metrics_calib['Biweight Scale']:.4f}$",
        rf"$\eta_{{0.15}} = {metrics_calib['Outlier Rate (>0.15)'] * 100:.2f}\%$",
        rf"$\eta_{{0.30}} = {metrics_calib['Severe Outlier (>0.30)'] * 100:.2f}\%$",
        rf"$W_1 = {metrics_calib['Wasserstein W1']:.4f}$",
        f"$N = {N:,}$ (5-Fold OOF)",
    ]
    info_text = "\n".join(info_lines)
    ax.text(
        0.05, 0.95, info_text,
        transform=ax.transAxes,
        fontsize=11.5,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#cccccc')
    )
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#cccccc', fontsize=11)
    fig.tight_layout()
    fig.savefig(plot_dir / "figure1_zphot_vs_zspec_hexbin.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  -> Saved figure1_zphot_vs_zspec_hexbin.png")

    # Figure 2: PIT Diagnostics
    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    pit = metrics_calib['pit_values']

    ax_hist.hist(pit, bins=40, density=True, color='#1f77b4', edgecolor='white', alpha=0.85, label='Empirical PIT')
    ax_hist.axhline(1.0, color='crimson', ls='--', lw=2.0, label='Uniform (Ideal Calibration)')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel('Probability Integral Transform (PIT)', fontsize=12)
    ax_hist.set_ylabel('Normalized Density', fontsize=12)
    ax_hist.set_title('PIT Coverage Distribution', fontsize=13, pad=10)
    ax_hist.legend(loc='upper center', frameon=True, facecolor='white', framealpha=0.9, fontsize=10.5)

    sorted_p = np.sort(pit)
    cum_p = np.linspace(0, 1, len(sorted_p))
    ax_cdf.plot(sorted_p, cum_p, color='#1f77b4', lw=2.2, label='Empirical Cumulative PIT')
    ax_cdf.plot([0, 1], [0, 1], color='crimson', ls='--', lw=2.0, label='Ideal Diagonal')
    ax_cdf.set_xlim(0, 1)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.set_xlabel('PIT Value', fontsize=12)
    ax_cdf.set_ylabel('Empirical CDF', fontsize=12)
    ax_cdf.set_title('Cumulative PIT vs. Uniform CDF', fontsize=13, pad=10)

    pit_text = (
        f"Mean(PIT) = {metrics_calib['Mean PIT']:.4f}\n"
        f"Var(PIT)  = {metrics_calib['Var(PIT)']:.4f}\n"
        f"$D_{{\\rm KS}}$     = {metrics_calib['PIT KS Statistic']:.4f}\n"
        f"CvM       = {metrics_calib['Cramer-von Mises (CvM)']:.2f}"
    )
    ax_cdf.text(
        0.05, 0.95, pit_text,
        transform=ax_cdf.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.92, edgecolor='#cccccc')
    )
    ax_cdf.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(plot_dir / "figure2_pit_diagnostics.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  -> Saved figure2_pit_diagnostics.png")

    # Figure 3: Stacked N(z) Ensemble
    fig, (ax_nz, ax_res) = plt.subplots(2, 1, figsize=(9.5, 7.5), dpi=300, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.15})
    stacked_nz = np.mean(oof_pdfs_calib, axis=0) / dz
    hist_true, bin_edges = np.histogram(z_true_all, bins=len(Z_CENTERS), range=(Z_CENTERS[0]-0.5*dz, Z_CENTERS[-1]+0.5*dz), density=True)

    ax_nz.plot(Z_CENTERS, stacked_nz, color='#1f77b4', lw=2.2, label=r'Stacked Ensemble $\hat{N}(z)$ (Pontifex)')
    ax_nz.step(Z_CENTERS, hist_true, where='mid', color='black', lw=1.8, ls='--', label=r'True Spectroscopic $N_{\rm spec}(z)$')
    ax_nz.set_xlim(0, 3)
    ax_nz.set_ylabel(r'Normalized Redshift Density $\mathrm{d}N/\mathrm{d}z$', fontsize=12)
    ax_nz.set_title(r'Stacked Ensemble Redshift Distribution vs. Spectroscopic Ground Truth', fontsize=13, pad=10)
    ax_nz.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)

    srd_box = (
        rf"$\delta\mu = {metrics_calib['Delta_mu (DESC SRD)']:+.4f}$" + "\n"
        rf"$\delta\sigma = {metrics_calib['Delta_sigma (DESC SRD)']:+.4f}$" + "\n"
        rf"$W_1 = {metrics_calib['Wasserstein W1']:.4f}$"
    )
    ax_nz.text(
        0.05, 0.95, srd_box,
        transform=ax_nz.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.92, edgecolor='#cccccc')
    )

    residual = stacked_nz - hist_true
    ax_res.plot(Z_CENTERS, residual, color='#1f77b4', lw=1.5)
    ax_res.axhline(0.0, color='gray', ls='--', lw=1.0)
    ax_res.set_xlim(0, 3)
    ax_res.set_xlabel(r'Redshift $z$', fontsize=12)
    ax_res.set_ylabel(r'$\Delta N(z)$', fontsize=11)

    fig.savefig(plot_dir / "figure3_stacked_nz_ensemble.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  -> Saved figure3_stacked_nz_ensemble.png")

    # Figure 4: Error Evolution vs Redshift
    fig, (ax_mad, ax_out) = plt.subplots(2, 1, figsize=(9.5, 7.5), dpi=300, sharex=True, gridspec_kw={'hspace': 0.12})
    z_bins_eval = np.linspace(0.0, 3.0, 16)
    z_mids = 0.5 * (z_bins_eval[:-1] + z_bins_eval[1:])
    sigma_mad_binned = []
    outlier_binned = []

    res_all = metrics_calib['residuals']
    for i in range(len(z_bins_eval) - 1):
        mask_b = (z_true_all >= z_bins_eval[i]) & (z_true_all < z_bins_eval[i+1])
        if np.sum(mask_b) > 20:
            dz_b = res_all[mask_b]
            sm = float(1.4826 * np.median(np.abs(dz_b - np.median(dz_b))))
            ot = float(np.mean(np.abs(dz_b) > 0.15) * 100.0)
        else:
            sm, ot = np.nan, np.nan
        sigma_mad_binned.append(sm)
        outlier_binned.append(ot)

    ax_mad.plot(z_mids, sigma_mad_binned, marker='o', color='#1f77b4', lw=2.0, label=r'$\sigma_{\rm MAD}(z)$')
    ax_mad.axhline(0.03, color='forestgreen', ls='--', lw=1.5, label='LSD DESC Y10 Target (0.03)')
    ax_mad.axhline(0.05, color='orange', ls=':', lw=1.5, label='LSST DESC Y1 Target (0.05)')
    ax_mad.set_ylabel(r'$\sigma_{\rm MAD}$', fontsize=12)
    ax_mad.set_title('Photo-z Error Dispersion and Outlier Fraction vs. Redshift', fontsize=13, pad=10)
    ax_mad.set_ylim(0.0, 0.12)
    ax_mad.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    ax_out.plot(z_mids, outlier_binned, marker='s', color='#d62728', lw=2.0, label=r'Outlier Fraction $\eta_{0.15}(z)$ (%)')
    ax_out.axhline(10.0, color='gray', ls='--', lw=1.5, label='LSST DESC Y10 Outlier Limit (10%)')
    ax_out.set_xlabel(r'Spectroscopic Redshift $z_{\rm spec}$', fontsize=12)
    ax_out.set_ylabel(r'Outlier Rate $\eta_{0.15}$ (%)', fontsize=12)
    ax_out.set_xlim(0, 3)
    ax_out.set_ylim(0, 35)
    ax_out.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    fig.savefig(plot_dir / "figure4_error_vs_redshift.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  -> Saved figure4_error_vs_redshift.png")

    total_time = time.time() - t_start
    print("\n" + "=" * 85)
    print(f"PIPELINE EXECUTION COMPLETED IN {total_time / 60.0:.2f} MINUTES")
    print(f"Results archived in: {res_dir}")
    print("=" * 85)


if __name__ == '__main__':
    main()
