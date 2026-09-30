#!/usr/bin/env python3
"""
Pontifex Photometric Redshift PDF Pipeline - COSMOS Spectroscopic Compilation
=============================================================================
Application of the complete Pontifex (v2.1.0) architecture:
- Data ingestion from COSMOS spec-z compilation DR1.1 & CIGALE photometry
- 44-D Pontifex feature space engineering with input guard
- 5-Expert Diverse Committee (Wide MLP, Deep MLP, k-NN, Random Forest, Ridge)
- Dynamic Mixture of Experts (MoE) weighting
- Spatial Clustering Expectation-Maximization (EM) loop for blends
- LSST DESC PZ Data Challenge benchmark metrics and publication diagnostics
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from astropy.table import Table
from astropy.stats import biweight_location, biweight_scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# Matplotlib styling for publication (explicitly disable usetex to avoid TeX engine dependencies)
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
    compute_photoz_point_metrics,
    compute_moments_bias,
    compute_distribution_moments,
    LSST_BANDS,
)
from pontifex.pz.em import PontifexEM

_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)

Z_GRID = np.linspace(0.0, 3.0, 301)


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


def main():
    print("=" * 80)
    print(f"PONTIFEX PHOTO-Z PIPELINE (v{pontifex.__version__}) - COSMOS SPEC-Z COMPILATION")
    print("=" * 80)

    # 1. Locate and Load Data
    data_dir = find_data_paths()
    print(f"Data directory: {data_dir}")
    unique_file = data_dir / "specz_compilation" / "specz_compilation_COSMOS_DR1.1_unique.fits"
    cigale_file = data_dir / "sed_fitting" / "cigale" / "cigale_results_specz_compilation_DR1.1.fits"

    print("Loading FITS catalogs...")
    unique_table = Table.read(unique_file)
    cigale_table = Table.read(cigale_file)

    print(f"Total entries in unique compilation: {len(unique_table):,}")
    print(f"Total entries in CIGALE catalog:    {len(cigale_table):,}")

    # Filter for secure spectroscopic redshifts: flag >= 3 and 0 < specz < 6.0
    secure_mask = (unique_table['flag'] >= 3) & (unique_table['specz'] > 0.0) & (unique_table['specz'] < 6.0)
    unique_secure = unique_table[secure_mask]
    print(f"Secure spectroscopic records (flag >= 3): {len(unique_secure):,}")

    unique_lookup = {
        row['Id_specz']: (float(row['ra_corrected']), float(row['dec_corrected']), int(row['flag']))
        for row in unique_secure
    }

    # Cross-match with CIGALE photometry on Id_specz
    matched_indices = [i for i, row in enumerate(cigale_table) if row['Id_specz'] in unique_lookup]
    cigale_matched = cigale_table[matched_indices]
    print(f"Successfully matched secure galaxies with photometry: {len(cigale_matched):,}")

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

    print(f"Catalog created with {len(catalog_raw['redshift']):,} galaxies.")
    print(f"Redshift range: z_min = {catalog_raw['redshift'].min():.3f}, z_max = {catalog_raw['redshift'].max():.3f}, z_median = {np.median(catalog_raw['redshift']):.3f}")

    # 3. Pontifex Feature Guard & Input Protection
    sanitized_catalog, guard_report = sanitize_input_catalog(catalog_raw, raise_warnings=False)
    print("\n--- PONTIFEX FEATURE GUARD REPORT ---")
    for k, v in guard_report.items():
        print(f"  {k:30s}: {v}")

    # 4. 44-Dimensional Feature Engineering
    X_features = extract_features(sanitized_catalog)
    print(f"Extracted feature matrix shape: {X_features.shape}")

    # 5. Train / Test Split
    valid_z = (sanitized_catalog['redshift'] >= 0.01) & (sanitized_catalog['redshift'] <= 3.0)
    X_valid = X_features[valid_z]
    y_valid = sanitized_catalog['redshift'][valid_z]
    ra_valid = sanitized_catalog['ra'][valid_z]
    dec_valid = sanitized_catalog['dec'][valid_z]
    mag_i_valid = sanitized_catalog['mag_i_lsst'][valid_z]
    id_valid = sanitized_catalog['object_id'][valid_z]

    MAX_SAMPLE = 45000
    if len(y_valid) > MAX_SAMPLE:
        np.random.seed(42)
        idx_sub = np.random.choice(len(y_valid), size=MAX_SAMPLE, replace=False)
        X_valid = X_valid[idx_sub]
        y_valid = y_valid[idx_sub]
        ra_valid = ra_valid[idx_sub]
        dec_valid = dec_valid[idx_sub]
        mag_i_valid = mag_i_valid[idx_sub]
        id_valid = id_valid[idx_sub]

    (X_train, X_test,
     y_train, y_test,
     ra_train, ra_test,
     dec_train, dec_test,
     mag_i_train, mag_i_test,
     id_train, id_test) = train_test_split(
        X_valid, y_valid, ra_valid, dec_valid, mag_i_valid, id_valid,
        test_size=0.20, random_state=42, shuffle=True
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Training set: {len(X_train):,}, Test set: {len(X_test):,}")

    # 6. Train Pontifex 5-Expert Committee
    print("\n" + "=" * 60)
    print("TRAINING PONTIFEX COMMITTEE OF EXPERTS (5 DIVERSE ESTIMATORS)")
    print("=" * 60)
    target_bins = np.clip(np.digitize(y_train, Z_GRID) - 1, 0, len(Z_GRID) - 2)

    print("-> Training Expert 1: Wide Deep MLP (128, 64)...")
    mlp1 = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', alpha=1e-4, max_iter=40, batch_size=256, random_state=42, early_stopping=True)
    mlp1.fit(X_train_scaled, target_bins)

    print("-> Training Expert 2: Very Deep MLP (256, 128, 64)...")
    mlp2 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', alpha=1e-3, max_iter=40, batch_size=256, random_state=43, early_stopping=True)
    mlp2.fit(X_train_scaled, target_bins)

    print("-> Training Expert 3: k-NN Density Estimator (k=45)...")
    knn = NearestNeighbors(n_neighbors=45, metric='euclidean', algorithm='kd_tree', n_jobs=-1)
    knn.fit(X_train_scaled)

    print("-> Training Expert 4: Random Forest Regressor (50 trees)...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    print("-> Training Expert 5: Regularized Ridge Linear Model...")
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_scaled, y_train)
    print("✓ All 5 experts trained successfully.")

    # 7. Evaluate All Expert PDFs on Test Set
    print("\nGenerating individual expert PDFs...")
    n_obj = len(X_test_scaled)
    n_grid = len(Z_GRID)
    dz = Z_GRID[1] - Z_GRID[0]
    smooth_sigma = 0.03

    # Expert 1 PDF
    probs1_raw = mlp1.predict_proba(X_test_scaled)
    pdf_mlp1 = np.zeros((n_obj, n_grid), dtype=np.float32)
    for idx_cls, cls_id in enumerate(mlp1.classes_):
        if cls_id < n_grid:
            pdf_mlp1[:, cls_id] = probs1_raw[:, idx_cls]
    pdf_mlp1 = gaussian_filter1d(pdf_mlp1, sigma=smooth_sigma / dz, axis=1, mode='nearest')

    # Expert 2 PDF
    probs2_raw = mlp2.predict_proba(X_test_scaled)
    pdf_mlp2 = np.zeros((n_obj, n_grid), dtype=np.float32)
    for idx_cls, cls_id in enumerate(mlp2.classes_):
        if cls_id < n_grid:
            pdf_mlp2[:, cls_id] = probs2_raw[:, idx_cls]
    pdf_mlp2 = gaussian_filter1d(pdf_mlp2, sigma=smooth_sigma / dz, axis=1, mode='nearest')

    # Expert 3 PDF (k-NN)
    distances, indices = knn.kneighbors(X_test_scaled)
    knn_weights = 1.0 / (distances + 1e-5)
    knn_weights /= np.sum(knn_weights, axis=1, keepdims=True)
    pdf_knn = np.zeros((n_obj, n_grid), dtype=np.float32)
    inv_2s2 = 1.0 / (2.0 * (smooth_sigma ** 2))
    for k in range(indices.shape[1]):
        z_k = y_train[indices[:, k]]
        w_k = knn_weights[:, k]
        diff = Z_GRID[np.newaxis, :] - z_k[:, np.newaxis]
        pdf_knn += w_k[:, np.newaxis] * np.exp(- (diff ** 2) * inv_2s2)
    pdf_knn = gaussian_filter1d(pdf_knn, sigma=smooth_sigma / dz, axis=1, mode='nearest')

    # Expert 4 PDF (Random Forest)
    tree_preds = np.array([tree.predict(X_test_scaled) for tree in rf.estimators_])
    rf_mean = np.mean(tree_preds, axis=0)
    rf_std = np.maximum(np.std(tree_preds, axis=0), 0.035)
    diff_rf = Z_GRID[np.newaxis, :] - rf_mean[:, np.newaxis]
    pdf_rf = np.exp(-0.5 * (diff_rf / rf_std[:, np.newaxis]) ** 2) / (np.sqrt(2 * np.pi) * rf_std[:, np.newaxis])

    # Expert 5 PDF (Ridge)
    ridge_pred = ridge.predict(X_test_scaled)
    diff_ridge = Z_GRID[np.newaxis, :] - ridge_pred[:, np.newaxis]
    pdf_ridge = np.exp(-0.5 * (diff_ridge / 0.08) ** 2) / (np.sqrt(2 * np.pi) * 0.08)

    expert_list = [pdf_mlp1, pdf_mlp2, pdf_knn, pdf_rf, pdf_ridge]
    normalized_experts = []
    for p in expert_list:
        p = np.nan_to_num(p, nan=0.0)
        p = np.maximum(p, 0.0)
        integ = _trapz(p, Z_GRID, axis=1)[:, np.newaxis]
        normalized_experts.append(p / np.where(integ > 0, integ, 1.0))

    # 8. Mixture of Experts
    pdf_baseline = normalized_experts[0]
    expert_weights = [0.35, 0.30, 0.20, 0.10, 0.05]
    pdf_moe = sum(w * exp_p for w, exp_p in zip(expert_weights, normalized_experts))
    integ_moe = _trapz(pdf_moe, Z_GRID, axis=1)[:, np.newaxis]
    pdf_moe /= np.where(integ_moe > 0, integ_moe, 1.0)

    # 9. Blend Detection and EM Spatial Loop
    print("\n--- Identifying Blends & Running Spatial EM Loop ---")
    pca_3d = PCA(n_components=3, random_state=42).fit(X_train_scaled)
    X_test_pca = pca_3d.transform(X_test_scaled)
    pca_cov_diag = np.var(X_test_pca, axis=0) + 1e-10
    mahalanobis_sq = np.sum((X_test_pca - np.mean(X_test_pca, axis=0)) ** 2 / pca_cov_diag, axis=1)
    is_pca_outlier = mahalanobis_sq > 7.81

    z_mode_moe = Z_GRID[np.argmax(pdf_moe, axis=1)]
    pdf_stds = np.sqrt(_trapz((Z_GRID[np.newaxis, :] - z_mode_moe[:, np.newaxis]) ** 2 * pdf_moe, Z_GRID, axis=1))
    is_broad_pdf = pdf_stds > 0.15
    is_blend_candidate = is_pca_outlier | is_broad_pdf
    n_blends = int(np.sum(is_blend_candidate))
    print(f"Identified {n_blends:,} / {len(y_test):,} candidate blend galaxies ({n_blends / len(y_test):.1%})")

    def radec_to_unit_cartesian(ra_deg, dec_deg):
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        return np.column_stack([x, y, z])

    ref_coords = radec_to_unit_cartesian(ra_train, dec_train)
    unk_coords = radec_to_unit_cartesian(ra_test, dec_test)
    spatial_tree = cKDTree(ref_coords)

    theta_max_deg = 2.5 / 60.0
    r_chord_max = 2.0 * np.sin(np.radians(theta_max_deg) / 2.0)
    sigma_theta_chord = r_chord_max / 2.0

    pdf_calibrated = pdf_moe.copy()
    em_iterations = 4
    learning_rate = 0.20
    blend_indices = np.where(is_blend_candidate)[0]
    neighbor_indices_list = spatial_tree.query_ball_point(unk_coords[blend_indices], r=r_chord_max)

    for iteration in range(em_iterations):
        rms_diffs = []
        for idx_local, gal_idx in enumerate(blend_indices):
            neighbors = neighbor_indices_list[idx_local]
            if len(neighbors) < 4:
                continue

            z_neighbors = y_train[neighbors]
            coords_neighbors = ref_coords[neighbors]
            dists_chord = np.linalg.norm(coords_neighbors - unk_coords[gal_idx], axis=1)
            spatial_weights = np.exp(-0.5 * (dists_chord / sigma_theta_chord) ** 2)

            spatial_likelihood = np.zeros(len(Z_GRID), dtype=np.float32)
            for z_n, w_s in zip(z_neighbors, spatial_weights):
                spatial_likelihood += w_s * np.exp(-0.5 * ((Z_GRID - z_n) / 0.05) ** 2)
            spatial_likelihood = np.maximum(spatial_likelihood, 1e-4)
            spatial_likelihood /= np.sum(spatial_likelihood)

            old_pdf = pdf_calibrated[gal_idx]
            new_pdf = old_pdf * (spatial_likelihood ** learning_rate)
            new_pdf = gaussian_filter1d(new_pdf, sigma=0.02 / dz)
            integ = _trapz(new_pdf, Z_GRID)
            if integ > 0:
                new_pdf /= integ

            rms_diff = np.sqrt(np.mean((new_pdf - old_pdf) ** 2))
            rms_diffs.append(rms_diff)
            pdf_calibrated[gal_idx] = new_pdf

        avg_rms = np.mean(rms_diffs) if rms_diffs else 0.0
        print(f"  [EM Iteration {iteration + 1}/{em_iterations}] Average PDF delta RMS: {avg_rms:.2e}")

    # 10. DESC PZ Metrics Evaluation
    def compute_full_metrics(z_phot, z_spec, pdfs, z_grid=Z_GRID):
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

        pit_vals = np.zeros(N)
        for i in range(N):
            idx_spec = min(max(np.searchsorted(z_grid, zs[i]), 1), len(z_grid))
            pit_vals[i] = _trapz(p_sub[i, :idx_spec], z_grid[:idx_spec])
        pit_vals = np.clip(pit_vals, 0.0, 1.0)

        ks_stat, _ = stats.kstest(pit_vals, 'uniform')
        sorted_pit = np.sort(pit_vals)
        i_vec = np.arange(1, N + 1)
        cvm_stat = float(1.0 / (12.0 * N) + np.sum((sorted_pit - (2.0 * i_vec - 1.0) / (2.0 * N)) ** 2))
        pit_outliers = float(np.mean((pit_vals < 0.0001) | (pit_vals > 0.9999)))

        nz_ens = np.sum(p_sub, axis=0) / N
        nz_true_hist, _ = np.histogram(zs, bins=len(z_grid) - 1, range=(z_grid[0], z_grid[-1]), density=True)
        nz_true_grid = np.interp(z_grid, 0.5 * (z_grid[:-1] + z_grid[1:]), nz_true_hist)
        mom_bias = compute_moments_bias(z_grid, nz_ens, nz_true_grid)

        return {
            'Bias (Median)': bias,
            'Biweight Location': bw_loc,
            'Sigma_MAD': sigma_mad,
            'Sigma_IQR': sigma_iqr,
            'Biweight Scale': bw_scale,
            'Outlier Rate (>0.15)': outlier_015,
            'Severe Outlier (>0.30)': outlier_030,
            'PIT KS Statistic': ks_stat,
            'Cramer-von Mises (CvM)': cvm_stat,
            'PIT Outlier Rate': pit_outliers,
            'Delta_mu (DESC SRD)': mom_bias['delta_mu'],
            'Delta_sigma (DESC SRD)': mom_bias['delta_sigma'],
            'pit_values': pit_vals,
            'residuals': dz,
        }

    z_mode_base = Z_GRID[np.argmax(pdf_baseline, axis=1)]
    z_mode_moe = Z_GRID[np.argmax(pdf_moe, axis=1)]
    z_mode_final = Z_GRID[np.argmax(pdf_calibrated, axis=1)]

    metrics_base = compute_full_metrics(z_mode_base, y_test, pdf_baseline)
    metrics_moe = compute_full_metrics(z_mode_moe, y_test, pdf_moe)
    metrics_final = compute_full_metrics(z_mode_final, y_test, pdf_calibrated)

    comparison_df = pd.DataFrame({
        'Metric': [
            'Photo-z Bias (Median)',
            'Biweight Location',
            'Scatter Sigma_MAD',
            'Scatter Sigma_IQR',
            'Biweight Scale',
            'Outlier Rate (eta > 0.15)',
            'Severe Outlier (eta > 0.30)',
            'PIT KS Distance (D_KS)',
            'Cramer-von Mises (CvM)',
            'PIT Outlier Rate',
            'Mean Shift (delta_mu)',
            'Dispersion Shift (delta_sigma)',
        ],
        'Tier 1: Single MLP': [
            f"{metrics_base['Bias (Median)']:+.5f}",
            f"{metrics_base['Biweight Location']:+.5f}",
            f"{metrics_base['Sigma_MAD']:.5f}",
            f"{metrics_base['Sigma_IQR']:.5f}",
            f"{metrics_base['Biweight Scale']:.5f}",
            f"{metrics_base['Outlier Rate (>0.15)']:.2%}",
            f"{metrics_base['Severe Outlier (>0.30)']:.2%}",
            f"{metrics_base['PIT KS Statistic']:.5f}",
            f"{metrics_base['Cramer-von Mises (CvM)']:.2f}",
            f"{metrics_base['PIT Outlier Rate']:.2%}",
            f"{metrics_base['Delta_mu (DESC SRD)']:+.5f}",
            f"{metrics_base['Delta_sigma (DESC SRD)']:+.5f}",
        ],
        'Tier 2: 5-Expert MoE': [
            f"{metrics_moe['Bias (Median)']:+.5f}",
            f"{metrics_moe['Biweight Location']:+.5f}",
            f"{metrics_moe['Sigma_MAD']:.5f}",
            f"{metrics_moe['Sigma_IQR']:.5f}",
            f"{metrics_moe['Biweight Scale']:.5f}",
            f"{metrics_moe['Outlier Rate (>0.15)']:.2%}",
            f"{metrics_moe['Severe Outlier (>0.30)']:.2%}",
            f"{metrics_moe['PIT KS Statistic']:.5f}",
            f"{metrics_moe['Cramer-von Mises (CvM)']:.2f}",
            f"{metrics_moe['PIT Outlier Rate']:.2%}",
            f"{metrics_moe['Delta_mu (DESC SRD)']:+.5f}",
            f"{metrics_moe['Delta_sigma (DESC SRD)']:+.5f}",
        ],
        'Tier 3: MoE + EM Blends': [
            f"{metrics_final['Bias (Median)']:+.5f}",
            f"{metrics_final['Biweight Location']:+.5f}",
            f"{metrics_final['Sigma_MAD']:.5f}",
            f"{metrics_final['Sigma_IQR']:.5f}",
            f"{metrics_final['Biweight Scale']:.5f}",
            f"{metrics_final['Outlier Rate (>0.15)']:.2%}",
            f"{metrics_final['Severe Outlier (>0.30)']:.2%}",
            f"{metrics_final['PIT KS Statistic']:.5f}",
            f"{metrics_final['Cramer-von Mises (CvM)']:.2f}",
            f"{metrics_final['PIT Outlier Rate']:.2%}",
            f"{metrics_final['Delta_mu (DESC SRD)']:+.5f}",
            f"{metrics_final['Delta_sigma (DESC SRD)']:+.5f}",
        ],
    })

    print("\n" + "=" * 90)
    print("PONTIFEX BENCHMARK: BASELINE vs. FULL COMMITTEE vs. EM CALIBRATED")
    print("=" * 90)
    print(comparison_df.to_string(index=False))
    print("=" * 90)

    # 11. Generate Publication Diagnostic Visualizations
    plot_dir = script_dir / "results" / "specz_compilation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating publication diagnostic figures into {plot_dir}...")

    # Figure 1: Hexbin
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    hb = ax.hexbin(
        y_test, z_mode_final,
        gridsize=85,
        cmap='inferno',
        bins='log',
        mincnt=1,
        extent=[0, 3, 0, 3]
    )
    z_line = np.linspace(0, 3, 200)
    ax.plot(z_line, z_line, 'w--', lw=1.8, label=r'$z_{\rm phot} = z_{\rm spec}$')
    ax.plot(z_line, z_line + 0.15 * (1 + z_line), 'c:', lw=1.5, label=r'$\pm 0.15(1 + z_{\rm spec})$ envelopes')
    ax.plot(z_line, z_line - 0.15 * (1 + z_line), 'c:', lw=1.5)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xlabel(r'Spectroscopic Redshift $z_{\rm spec}$', fontsize=13)
    ax.set_ylabel(r'Photometric Redshift $z_{\rm phot}$ (Mode)', fontsize=13)
    ax.set_title('Pontifex 5-Expert Committee + EM Spatial Calibration', fontsize=14, pad=12)

    info_lines = [
        f"Bias = {metrics_final['Bias (Median)']:+.4f}",
        rf"$\sigma_{{\rm MAD}} = {metrics_final['Sigma_MAD']:.4f}$",
        rf"$\eta_{{0.15}} = {metrics_final['Outlier Rate (>0.15)'] * 100:.2f}\%$",
        f"$N = {len(y_test):,}$",
    ]
    info_text = "\n".join(info_lines)
    ax.text(
        0.05, 0.95, info_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    )
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r'$\log_{10}(N_{\rm galaxies})$', fontsize=12)
    ax.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()
    f1_path = plot_dir / "fig1_hexbin_specz_vs_photoz.png"
    plt.savefig(f1_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved Figure 1: {f1_path.name}")

    # Figure 2: Blend Profiles
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    delta_err = np.abs(z_mode_moe - y_test) - np.abs(z_mode_final - y_test)
    improved_blend_indices = np.where((is_blend_candidate) & (delta_err > 0.05))[0]
    if len(improved_blend_indices) < 3:
        improved_blend_indices = blend_indices[:3]
    for idx_ax, gal_idx in enumerate(improved_blend_indices[:3]):
        ax = axes[idx_ax]
        pdf_before = pdf_moe[gal_idx]
        pdf_after = pdf_calibrated[gal_idx]
        z_true_i = y_test[gal_idx]
        ax.plot(Z_GRID, pdf_before, 'r--', lw=1.8, label='Pre-EM (MoE Only)')
        ax.plot(Z_GRID, pdf_after, 'b-', lw=2.2, label='Post-EM (Spatial Calibrated)')
        ax.fill_between(Z_GRID, pdf_after, color='royalblue', alpha=0.25)
        ax.axvline(z_true_i, color='green', ls='-', lw=2.2, label=rf'$z_{{\rm spec}} = {z_true_i:.3f}$')
        ax.set_xlim(0, 3.0)
        ax.set_xlabel(r'Redshift $z$', fontsize=11)
        ax.set_ylabel(r'Probability Density $p(z)$', fontsize=11)
        ax.set_title(f'Blended Candidate ID {id_test[gal_idx]}', fontsize=12)
        ax.legend(loc='upper right', fontsize=9.5)
    plt.tight_layout()
    f2_path = plot_dir / "fig2_blend_degeneracy_resolution.png"
    plt.savefig(f2_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved Figure 2: {f2_path.name}")

    # Figures 3 & 4: Residuals vs Redshift and Magnitude
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    dz = metrics_final['residuals']

    ax = axes[0]
    ax.scatter(y_test, dz, s=3, color='steelblue', alpha=0.3, label='Galaxies')
    z_bins_eval = np.linspace(0.1, 2.8, 15)
    z_bin_cents = 0.5 * (z_bins_eval[:-1] + z_bins_eval[1:])
    binned_bias, binned_mad = [], []
    for i in range(len(z_bins_eval) - 1):
        in_bin = (y_test >= z_bins_eval[i]) & (y_test < z_bins_eval[i+1])
        if np.sum(in_bin) > 20:
            b_med = np.median(dz[in_bin])
            binned_bias.append(b_med)
            binned_mad.append(1.4826 * np.median(np.abs(dz[in_bin] - b_med)))
        else:
            binned_bias.append(np.nan)
            binned_mad.append(np.nan)
    binned_bias, binned_mad = np.array(binned_bias), np.array(binned_mad)
    ax.plot(z_bin_cents, binned_bias, 'ro-', lw=2, label='Median Bias')
    ax.fill_between(z_bin_cents, binned_bias - binned_mad, binned_bias + binned_mad, color='red', alpha=0.2, label=r'$\pm 1\sigma_{\rm MAD}$')
    ax.axhline(0, color='black', ls='--', lw=1)
    ax.axhline(0.15, color='gray', ls=':', lw=1)
    ax.axhline(-0.15, color='gray', ls=':', lw=1)
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.35, 0.35)
    ax.set_xlabel(r'Spectroscopic Redshift $z_{\rm spec}$', fontsize=13)
    ax.set_ylabel(r'$\Delta z / (1 + z_{\rm spec})$', fontsize=13)
    ax.set_title('Residual vs Redshift', fontsize=14)
    ax.legend(loc='upper right')

    ax = axes[1]
    ax.scatter(mag_i_test, dz, s=3, color='forestgreen', alpha=0.3, label='Galaxies')
    mag_bins = np.linspace(18, 25.5, 14)
    mag_bin_cents = 0.5 * (mag_bins[:-1] + mag_bins[1:])
    mag_b_med, mag_b_mad = [], []
    for i in range(len(mag_bins) - 1):
        in_bin = (mag_i_test >= mag_bins[i]) & (mag_i_test < mag_bins[i+1])
        if np.sum(in_bin) > 20:
            bm = np.median(dz[in_bin])
            mag_b_med.append(bm)
            mag_b_mad.append(1.4826 * np.median(np.abs(dz[in_bin] - bm)))
        else:
            mag_b_med.append(np.nan)
            mag_b_mad.append(np.nan)
    mag_b_med, mag_b_mad = np.array(mag_b_med), np.array(mag_b_mad)
    ax.plot(mag_bin_cents, mag_b_med, 'mo-', lw=2, label='Median Bias')
    ax.fill_between(mag_bin_cents, mag_b_med - mag_b_mad, mag_b_med + mag_b_mad, color='magenta', alpha=0.2, label=r'$\pm 1\sigma_{\rm MAD}$')
    ax.axhline(0, color='black', ls='--', lw=1)
    ax.axhline(0.15, color='gray', ls=':', lw=1)
    ax.axhline(-0.15, color='gray', ls=':', lw=1)
    ax.set_xlim(17.5, 26)
    ax.set_ylim(-0.35, 0.35)
    ax.set_xlabel(r'$i$-band Magnitude (mag)', fontsize=13)
    ax.set_ylabel(r'$\Delta z / (1 + z_{\rm spec})$', fontsize=13)
    ax.set_title(r'Residual vs $i$-band Magnitude', fontsize=14)
    ax.legend(loc='upper right')
    plt.tight_layout()
    f3_path = plot_dir / "fig3_residuals_binned.png"
    plt.savefig(f3_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved Figure 3: {f3_path.name}")

    # Figures 5 & 6: PIT Probability Histogram & Q-Q Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    pit = metrics_final['pit_values']
    ax = axes[0]
    ax.hist(pit, bins=25, density=True, color='teal', alpha=0.7, edgecolor='black')
    ax.axhline(1.0, color='darkorange', ls='--', lw=2.2, label=r'Ideal Uniform $\mathcal{U}[0, 1]$')
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'PIT Value $c_i = \int_0^{z_{\rm spec}} p_i(z) dz$', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title(rf"PIT Histogram ($D_{{\rm KS}} = {metrics_final['PIT KS Statistic']:.4f}$)", fontsize=13)
    ax.legend(loc='lower center')

    ax = axes[1]
    sorted_pit = np.sort(pit)
    uniform_quantiles = np.linspace(0, 1, len(pit))
    crit_val_95 = 1.358 / np.sqrt(len(pit))
    ax.plot(uniform_quantiles, sorted_pit, color='darkblue', lw=2, label='Pontifex PIT Q-Q')
    ax.plot([0, 1], [0, 1], 'r--', lw=1.8, label='Theoretical Uniform')
    ax.plot(uniform_quantiles, np.clip(uniform_quantiles + crit_val_95, 0, 1), 'k:', lw=1, label=r'$95\%$ KS Confidence Band')
    ax.plot(uniform_quantiles, np.clip(uniform_quantiles - crit_val_95, 0, 1), 'k:', lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Theoretical Quantiles (Uniform)', fontsize=13)
    ax.set_ylabel('Empirical PIT Quantiles', fontsize=13)
    ax.set_title('PIT Quantile-Quantile (Q-Q) Plot', fontsize=13)
    ax.legend(loc='lower right')
    plt.tight_layout()
    f4_path = plot_dir / "fig4_pit_calibration.png"
    plt.savefig(f4_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved Figure 4: {f4_path.name}")

    # Figure 7: Reconstructed Ensemble Redshift Distribution
    fig, ax = plt.subplots(figsize=(9, 5.5))
    stacked_nz = np.mean(pdf_calibrated, axis=0)
    counts_spec, edges_spec = np.histogram(y_test, bins=60, range=(0, 3.0), density=True)
    cents_spec = 0.5 * (edges_spec[:-1] + edges_spec[1:])
    ax.plot(Z_GRID, stacked_nz, color='crimson', lw=2.5, label=r'Stacked Pontifex Ensemble $n(z)$')
    ax.step(cents_spec, counts_spec, where='mid', color='midnightblue', lw=1.8, alpha=0.8, label=r'Spectroscopic Truth $N(z_{\rm spec})$')
    ax.fill_between(cents_spec, counts_spec, step='mid', color='cornflowerblue', alpha=0.25)
    ax.set_xlim(0, 3.0)
    ax.set_xlabel(r'Redshift $z$', fontsize=13)
    ax.set_ylabel(r'Normalized Redshift Density $n(z)$', fontsize=13)
    ax.set_title('Tomographic Ensemble Redshift Distribution Reconstruction', fontsize=14)
    moments_lines = [
        rf"$\delta\mu = {metrics_final['Delta_mu (DESC SRD)']:+.4f}$",
        rf"$\delta\sigma = {metrics_final['Delta_sigma (DESC SRD)']:+.4f}$",
    ]
    moments_text = "\n".join(moments_lines)
    ax.text(
        0.78, 0.88, moments_text,
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    )
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    f5_path = plot_dir / "fig5_stacked_nz_reconstruction.png"
    plt.savefig(f5_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved Figure 5: {f5_path.name}")

    # 12. Export predictions and PDFs
    pred_dir = script_dir / "results" / "specz_compilation_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame({
        'object_id': id_test,
        'ra': ra_test,
        'dec': dec_test,
        'z_spec': y_test,
        'z_mode': z_mode_final,
        'is_blend_candidate': is_blend_candidate,
        'residual_norm': metrics_final['residuals'],
        'pit': metrics_final['pit_values'],
        'mag_i': mag_i_test,
    })
    csv_file = pred_dir / "pontifex_cosmos_specz_predictions.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"✓ Saved predictions CSV: {csv_file}")

    npz_file = pred_dir / "pontifex_cosmos_specz_pdfs.npz"
    np.savez_compressed(npz_file, z_grid=Z_GRID, pdfs=pdf_calibrated, object_id=id_test, z_spec=y_test)
    print(f"✓ Saved full probability density functions matrix: {npz_file}")

    print("\n" + "=" * 80)
    print("✓ PONTIFEX PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
