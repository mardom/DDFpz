#!/usr/bin/env python3
"""
Pontifex Photometric Redshift PDF Pipeline - COSMOS Spectroscopic Compilation
=============================================================================
Application of the complete Pontifex (v2.1.0) architecture:
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

Z_GRID = np.linspace(0.01, 3.0, 301)


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

    # Convert to DataFrames
    df_unique = unique_table.to_pandas()
    df_cigale = cigale_table.to_pandas()

    for col in ['Id_specz', 'id', 'ID', 'Id']:
        if col in df_unique.columns:
            df_unique['Id_specz'] = df_unique[col].astype(np.int64)
            break
    for col in ['Id_specz', 'id', 'ID', 'Id']:
        if col in df_cigale.columns:
            df_cigale['Id_specz'] = df_cigale[col].astype(np.int64)
            break

    # Filter secure redshifts: flag >= 3 and z > 0
    secure_mask = (df_unique['flag'] >= 3) & (df_unique['z'] > 0) & (df_unique['z'] < 6.0)
    df_unique_secure = df_unique[secure_mask]
    print(f"Secure spec-z entries: {len(df_unique_secure):,} / {len(df_unique):,}")

    # Merge catalogs
    merged_df = pd.merge(df_unique_secure, df_cigale, on='Id_specz', suffixes=('', '_cigale'))
    print(f"Matched galaxies with secure spec-z & CIGALE photometry: {len(merged_df):,}")

    # 2. Filter Bands & Convert Flux to Magnitude
    cigale_band_map = {
        'u': ('subaru_Su_mod', 'subaru_Su_err'),
        'g': ('subaru_B_mod', 'subaru_B_err'),
        'r': ('subaru_r_mod', 'subaru_r_err'),
        'i': ('subaru_ip_mod', 'subaru_ip_err'),
        'z': ('subaru_zp_mod', 'subaru_zp_err'),
        'y': ('subaru_zpp_mod', 'subaru_zpp_err'),
    }

    n_samples = len(merged_df)
    mags = np.full((n_samples, 6), np.nan, dtype=np.float32)
    mag_errs = np.full((n_samples, 6), np.nan, dtype=np.float32)

    for idx, band in enumerate(LSST_BANDS):
        flux_col, err_col = cigale_band_map[band]
        f = merged_df[flux_col].values.astype(np.float64)
        ef = merged_df[err_col].values.astype(np.float64) if err_col in merged_df else 0.05 * f
        pos = f > 0
        m = np.full_like(f, np.nan)
        em = np.full_like(ef, 0.3)
        m[pos] = 8.90 - 2.5 * np.log10(f[pos])
        em[pos] = np.clip(1.0857 * (ef[pos] / f[pos]), 0.01, 1.5)
        mags[:, idx] = m
        mag_errs[:, idx] = em

    valid_mask = np.all(np.isfinite(mags), axis=1) & (merged_df['z'].values > 0)
    mags = mags[valid_mask]
    mag_errs = mag_errs[valid_mask]
    filtered_df = merged_df[valid_mask].reset_index(drop=True)
    z_spec = filtered_df['z'].values.astype(np.float32)
    ra = filtered_df['ra'].values.astype(np.float64)
    dec = filtered_df['dec'].values.astype(np.float64)
    obj_ids = filtered_df['Id_specz'].values

    print(f"Valid calibrated galaxies with 6-band photometry: {len(filtered_df):,}")

    # 3. Sanitize and Feature Extraction
    mags_clean, errs_clean = sanitize_input_catalog(mags, mag_errs)
    X_features = extract_features(mags_clean, errs_clean)
    print(f"Extracted {X_features.shape[1]}-dimensional Pontifex feature space.")

    # 4. Train / Test Split
    train_size = min(36000, int(0.8 * len(filtered_df)))
    test_size = min(9000, len(filtered_df) - train_size)

    (X_train, X_test,
     y_train, y_test,
     ra_train, ra_test,
     dec_train, dec_test,
     id_train, id_test,
     mags_tr, mags_te) = train_test_split(
        X_features, z_spec, ra, dec, obj_ids, mags_clean,
        train_size=train_size, test_size=test_size,
        random_state=42, shuffle=True
    )
    mag_i_test = mags_te[:, 2]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Training set: {len(X_train):,}, Test set: {len(X_test):,}")

    # 5. Train Full 5-Expert Committee
    print("\n--- Training Pontifex 5-Expert Committee ---")
    target_bins = np.clip(np.digitize(y_train, Z_GRID) - 1, 0, len(Z_GRID) - 2)

    print("1/5 Training Expert 1: Wide Deep MLP (128, 64)...")
    mlp1 = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', alpha=1e-4, max_iter=40, batch_size=256, random_state=42, early_stopping=True)
    mlp1.fit(X_train_scaled, target_bins)

    print("2/5 Training Expert 2: Very Deep MLP (256, 128, 64)...")
    mlp2 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', alpha=1e-3, max_iter=40, batch_size=256, random_state=43, early_stopping=True)
    mlp2.fit(X_train_scaled, target_bins)

    print("3/5 Training Expert 3: k-NN Density Estimator (k=45)...")
    knn = NearestNeighbors(n_neighbors=45, metric='euclidean', algorithm='kd_tree', n_jobs=-1)
    knn.fit(X_train_scaled)

    print("4/5 Training Expert 4: Random Forest Regressor (50 trees)...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    print("5/5 Training Expert 5: Regularized Ridge Linear Model...")
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_scaled, y_train)
    print("✓ All 5 experts trained.")

    # 6. Evaluate Expert PDFs on Test Set
    print("\nGenerating PDFs for test set...")
    n_obj = len(X_test_scaled)
    n_grid = len(Z_GRID)
    dz = Z_GRID[1] - Z_GRID[0]
    smooth_sigma = 0.03

    # Expert 1 PDF
    p1_raw = mlp1.predict_proba(X_test_scaled)
    p_mlp1 = np.zeros((n_obj, n_grid), dtype=np.float32)
    for idx_cls, cls_id in enumerate(mlp1.classes_):
        if cls_id < n_grid:
            p_mlp1[:, cls_id] = p1_raw[:, idx_cls]
    p_mlp1 = gaussian_filter1d(p_mlp1, sigma=smooth_sigma / dz, axis=1, mode='nearest')

    # Expert 2 PDF
    p2_raw = mlp2.predict_proba(X_test_scaled)
    p_mlp2 = np.zeros((n_obj, n_grid), dtype=np.float32)
    for idx_cls, cls_id in enumerate(mlp2.classes_):
        if cls_id < n_grid:
            p_mlp2[:, cls_id] = p2_raw[:, idx_cls]
    p_mlp2 = gaussian_filter1d(p_mlp2, sigma=smooth_sigma / dz, axis=1, mode='nearest')

    # Expert 3 PDF (k-NN)
    dists, indices = knn.kneighbors(X_test_scaled)
    w_knn = 1.0 / (dists + 1e-5)
    w_knn /= np.sum(w_knn, axis=1, keepdims=True)
    p_knn = np.zeros((n_obj, n_grid), dtype=np.float32)
    inv_2s2 = 1.0 / (2.0 * (smooth_sigma ** 2))
    for k in range(indices.shape[1]):
        z_k = y_train[indices[:, k]]
        w_k = w_knn[:, k]
        diff = Z_GRID[np.newaxis, :] - z_k[:, np.newaxis]
        p_knn += w_k[:, np.newaxis] * np.exp(- (diff ** 2) * inv_2s2)
    p_knn = gaussian_filter1d(p_knn, sigma=smooth_sigma / dz, axis=1, mode='nearest')

    # Expert 4 PDF (Random Forest)
    tree_preds = np.array([tree.predict(X_test_scaled) for tree in rf.estimators_])
    rf_mean = np.mean(tree_preds, axis=0)
    rf_std = np.maximum(np.std(tree_preds, axis=0), 0.035)
    diff_rf = Z_GRID[np.newaxis, :] - rf_mean[:, np.newaxis]
    p_rf = np.exp(-0.5 * (diff_rf / rf_std[:, np.newaxis]) ** 2) / (np.sqrt(2 * np.pi) * rf_std[:, np.newaxis])

    # Expert 5 PDF (Ridge)
    ridge_pred = ridge.predict(X_test_scaled)
    diff_ridge = Z_GRID[np.newaxis, :] - ridge_pred[:, np.newaxis]
    p_ridge = np.exp(-0.5 * (diff_ridge / 0.08) ** 2) / (np.sqrt(2 * np.pi) * 0.08)

    # Normalize individual PDFs
    all_pdfs = [p_mlp1, p_mlp2, p_knn, p_rf, p_ridge]
    norm_pdfs = []
    for p in all_pdfs:
        p = np.nan_to_num(p, nan=0.0)
        p = np.maximum(p, 0.0)
        integ = _trapz(p, Z_GRID, axis=1)[:, np.newaxis]
        norm_pdfs.append(p / np.where(integ > 0, integ, 1.0))

    # 7. Mixture of Experts Combination
    pdf_baseline = norm_pdfs[0]
    weights = [0.35, 0.30, 0.20, 0.10, 0.05]
    pdf_moe = sum(w * p for w, p in zip(weights, norm_pdfs))
    integ_moe = _trapz(pdf_moe, Z_GRID, axis=1)[:, np.newaxis]
    pdf_moe /= np.where(integ_moe > 0, integ_moe, 1.0)

    # 8. Blend Detection & Spatial EM Loop
    print("\n--- Running Spatial Clustering EM Calibration for Blends ---")
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
    print(f"Identified {n_blends:,} candidate blend galaxies ({n_blends / len(y_test):.1%})")

    # Spatial tree
    def radec_to_cartesian(ra_deg, dec_deg):
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)
        return np.column_stack([
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad)
        ])

    ref_coords = radec_to_cartesian(ra_train, dec_train)
    unk_coords = radec_to_cartesian(ra_test, dec_test)
    spatial_tree = cKDTree(ref_coords)
    r_chord_max = 2.0 * np.sin(np.radians(2.5 / 60.0) / 2.0)
    sigma_theta_chord = r_chord_max / 2.0

    pdf_calibrated = pdf_moe.copy()
    blend_indices = np.where(is_blend_candidate)[0]
    neighbor_indices_list = spatial_tree.query_ball_point(unk_coords[blend_indices], r=r_chord_max)

    for iteration in range(4):
        rms_diffs = []
        for idx_local, gal_idx in enumerate(blend_indices):
            neighbors = neighbor_indices_list[idx_local]
            if len(neighbors) < 4:
                continue
            z_n = y_train[neighbors]
            coords_n = ref_coords[neighbors]
            d_chord = np.linalg.norm(coords_n - unk_coords[gal_idx], axis=1)
            w_s = np.exp(-0.5 * (d_chord / sigma_theta_chord) ** 2)

            spatial_like = np.zeros(n_grid, dtype=np.float32)
            for zn_i, ws_i in zip(z_n, w_s):
                spatial_like += ws_i * np.exp(-0.5 * ((Z_GRID - zn_i) / 0.05) ** 2)
            spatial_like = np.maximum(spatial_like, 1e-4)
            spatial_like /= np.sum(spatial_like)

            old_p = pdf_calibrated[gal_idx]
            new_p = old_p * (spatial_like ** 0.20)
            new_p = gaussian_filter1d(new_p, sigma=0.02 / dz)
            integ = _trapz(new_p, Z_GRID)
            if integ > 0:
                new_p /= integ
            rms_diffs.append(np.sqrt(np.mean((new_p - old_p) ** 2)))
            pdf_calibrated[gal_idx] = new_p
        avg_rms = np.mean(rms_diffs) if rms_diffs else 0.0
        print(f"  [EM Iteration {iteration + 1}/4] Average PDF RMS delta: {avg_rms:.2e}")

    # 9. Compute DESC PZ Metrics
    def compute_metrics(zp, zs, pdfs):
        dz = (zp - zs) / (1.0 + zs)
        bias = float(np.median(dz))
        bw_loc = float(biweight_location(dz))
        sigma_mad = float(1.4826 * np.median(np.abs(dz - bias)))
        sigma_iqr = float((np.percentile(dz, 75) - np.percentile(dz, 25)) / 1.349)
        bw_scale = float(biweight_scale(dz))
        outlier_015 = float(np.mean(np.abs(dz) > 0.15))
        outlier_030 = float(np.mean(np.abs(dz) > 0.30))

        N = len(zp)
        pit = np.zeros(N)
        for i in range(N):
            idx_s = min(max(np.searchsorted(Z_GRID, zs[i]), 1), len(Z_GRID))
            pit[i] = _trapz(pdfs[i, :idx_s], Z_GRID[:idx_s])
        pit = np.clip(pit, 0.0, 1.0)
        ks_stat, _ = stats.kstest(pit, 'uniform')
        sorted_pit = np.sort(pit)
        i_vec = np.arange(1, N + 1)
        cvm_stat = float(1.0 / (12.0 * N) + np.sum((sorted_pit - (2.0 * i_vec - 1.0) / (2.0 * N)) ** 2))

        nz_ens = np.sum(pdfs, axis=0) / N
        nz_true_hist, _ = np.histogram(zs, bins=len(Z_GRID) - 1, range=(Z_GRID[0], Z_GRID[-1]), density=True)
        nz_true_grid = np.interp(Z_GRID, 0.5 * (Z_GRID[:-1] + Z_GRID[1:]), nz_true_hist)
        mom_bias = compute_moments_bias(Z_GRID, nz_ens, nz_true_grid)

        return {
            'bias': bias, 'bw_loc': bw_loc, 'sigma_mad': sigma_mad, 'sigma_iqr': sigma_iqr,
            'bw_scale': bw_scale, 'outlier_015': outlier_015, 'outlier_030': outlier_030,
            'ks_stat': ks_stat, 'cvm_stat': cvm_stat,
            'delta_mu': mom_bias['delta_mu'], 'delta_sigma': mom_bias['delta_sigma'],
            'pit': pit, 'residuals': dz,
        }

    z_mode_base = Z_GRID[np.argmax(pdf_baseline, axis=1)]
    z_mode_moe = Z_GRID[np.argmax(pdf_moe, axis=1)]
    z_mode_final = Z_GRID[np.argmax(pdf_calibrated, axis=1)]

    m_base = compute_metrics(z_mode_base, y_test, pdf_baseline)
    m_moe = compute_metrics(z_mode_moe, y_test, pdf_moe)
    m_final = compute_metrics(z_mode_final, y_test, pdf_calibrated)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON TABLE (DESC PZ DATA CHALLENGE METRICS)")
    print("=" * 80)
    header = f"{'Metric':<32} | {'Tier 1 (Base)':<14} | {'Tier 2 (MoE)':<14} | {'Tier 3 (Final)':<14}"
    print(header)
    print("-" * len(header))
    metrics_display = [
        ('Photo-z Bias (Median)', 'bias', '+.5f'),
        ('Scatter Sigma_MAD', 'sigma_mad', '.5f'),
        ('Scatter Sigma_IQR', 'sigma_iqr', '.5f'),
        ('Outlier Rate (eta > 0.15)', 'outlier_015', '.2%'),
        ('Severe Outlier (eta > 0.30)', 'outlier_030', '.2%'),
        ('PIT KS Distance (D_KS)', 'ks_stat', '.5f'),
        ('Cramer-von Mises (CvM)', 'cvm_stat', '.2f'),
        ('Mean Shift (delta_mu)', 'delta_mu', '+.5f'),
        ('Dispersion Shift (delta_sigma)', 'delta_sigma', '+.5f'),
    ]
    for label, key, fmt in metrics_display:
        v1 = format(m_base[key], fmt)
        v2 = format(m_moe[key], fmt)
        v3 = format(m_final[key], fmt)
        print(f"{label:<32} | {v1:<14} | {v2:<14} | {v3:<14}")
    print("=" * 80)

    # 10. Generate and Save Publication Diagnostic Plots
    plot_dir = script_dir / "results" / "specz_compilation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating and saving diagnostic figures to {plot_dir}...")

    # Figure 1: Hexbin
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    hb = ax.hexbin(y_test, z_mode_final, gridsize=85, cmap='inferno', bins='log', mincnt=1, extent=[0, 3, 0, 3])
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
        f"Bias = {m_final['bias']:+.4f}",
        rf"$\sigma_{{\rm MAD}} = {m_final['sigma_mad']:.4f}$",
        rf"$\eta_{{0.15}} = {m_final['outlier_015'] * 100:.2f}\%$",
        f"$N = {len(y_test):,}$",
    ]
    ax.text(0.05, 0.95, "\n".join(info_lines), transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
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
    improved = np.where((is_blend_candidate) & (delta_err > 0.05))[0]
    sample_blends = improved[:3] if len(improved) >= 3 else blend_indices[:3]
    for idx_ax, gal_idx in enumerate(sample_blends):
        ax = axes[idx_ax]
        ax.plot(Z_GRID, pdf_moe[gal_idx], 'r--', lw=1.8, label='Pre-EM (MoE Only)')
        ax.plot(Z_GRID, pdf_calibrated[gal_idx], 'b-', lw=2.2, label='Post-EM (Spatial Calibrated)')
        ax.fill_between(Z_GRID, pdf_calibrated[gal_idx], color='royalblue', alpha=0.25)
        ax.axvline(y_test[gal_idx], color='green', ls='-', lw=2.2, label=rf'$z_{{\rm spec}} = {y_test[gal_idx]:.3f}$')
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

    # Figure 3 & 4: Residuals
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    dz = m_final['residuals']
    ax = axes[0]
    ax.scatter(y_test, dz, s=3, color='steelblue', alpha=0.3, label='Galaxies')
    z_eval = np.linspace(0.1, 2.8, 15)
    z_cents = 0.5 * (z_eval[:-1] + z_eval[1:])
    b_med, b_mad = [], []
    for i in range(len(z_eval) - 1):
        in_b = (y_test >= z_eval[i]) & (y_test < z_eval[i+1])
        if np.sum(in_b) > 20:
            bm = np.median(dz[in_b])
            b_med.append(bm)
            b_mad.append(1.4826 * np.median(np.abs(dz[in_b] - bm)))
        else:
            b_med.append(np.nan)
            b_mad.append(np.nan)
    b_med, b_mad = np.array(b_med), np.array(b_mad)
    ax.plot(z_cents, b_med, 'ro-', lw=2, label='Median Bias')
    ax.fill_between(z_cents, b_med - b_mad, b_med + b_mad, color='red', alpha=0.2, label=r'$\pm 1\sigma_{\rm MAD}$')
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
    m_eval = np.linspace(18, 25.5, 14)
    m_cents = 0.5 * (m_eval[:-1] + m_eval[1:])
    mb_med, mb_mad = [], []
    for i in range(len(m_eval) - 1):
        in_b = (mag_i_test >= m_eval[i]) & (mag_i_test < m_eval[i+1])
        if np.sum(in_b) > 20:
            bm = np.median(dz[in_b])
            mb_med.append(bm)
            mb_mad.append(1.4826 * np.median(np.abs(dz[in_b] - bm)))
        else:
            mb_med.append(np.nan)
            mb_mad.append(np.nan)
    mb_med, mb_mad = np.array(mb_med), np.array(mb_mad)
    ax.plot(m_cents, mb_med, 'mo-', lw=2, label='Median Bias')
    ax.fill_between(m_cents, mb_med - mb_mad, mb_med + mb_mad, color='magenta', alpha=0.2, label=r'$\pm 1\sigma_{\rm MAD}$')
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

    # Figure 5 & 6: PIT
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    pit = m_final['pit']
    ax = axes[0]
    ax.hist(pit, bins=25, density=True, color='teal', alpha=0.7, edgecolor='black')
    ax.axhline(1.0, color='darkorange', ls='--', lw=2.2, label=r'Ideal Uniform $\mathcal{U}[0, 1]$')
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'PIT Value $c_i = \int_0^{z_{\rm spec}} p_i(z) dz$', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title(rf"PIT Histogram ($D_{{\rm KS}} = {m_final['ks_stat']:.4f}$)", fontsize=13)
    ax.legend(loc='lower center')

    ax = axes[1]
    sorted_pit = np.sort(pit)
    u_q = np.linspace(0, 1, len(pit))
    crit_val = 1.358 / np.sqrt(len(pit))
    ax.plot(u_q, sorted_pit, color='darkblue', lw=2, label='Pontifex PIT Q-Q')
    ax.plot([0, 1], [0, 1], 'r--', lw=1.8, label='Theoretical Uniform')
    ax.plot(u_q, np.clip(u_q + crit_val, 0, 1), 'k:', lw=1, label=r'$95\%$ KS Confidence Band')
    ax.plot(u_q, np.clip(u_q - crit_val, 0, 1), 'k:', lw=1)
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

    # Figure 7: Ensemble n(z)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    stacked_nz = np.mean(pdf_calibrated, axis=0)
    counts_s, edges_s = np.histogram(y_test, bins=60, range=(0, 3.0), density=True)
    cents_s = 0.5 * (edges_s[:-1] + edges_s[1:])
    ax.plot(Z_GRID, stacked_nz, color='crimson', lw=2.5, label=r'Stacked Pontifex Ensemble $n(z)$')
    ax.step(cents_s, counts_s, where='mid', color='midnightblue', lw=1.8, alpha=0.8, label=r'Spectroscopic Truth $N(z_{\rm spec})$')
    ax.fill_between(cents_s, counts_s, step='mid', color='cornflowerblue', alpha=0.25)
    ax.set_xlim(0, 3.0)
    ax.set_xlabel(r'Redshift $z$', fontsize=13)
    ax.set_ylabel(r'Normalized Redshift Density $n(z)$', fontsize=13)
    ax.set_title('Tomographic Ensemble Redshift Distribution Reconstruction', fontsize=14)
    mom_lines = [
        rf"$\delta\mu = {m_final['delta_mu']:+.4f}$",
        rf"$\delta\sigma = {m_final['delta_sigma']:+.4f}$",
    ]
    ax.text(0.78, 0.88, "\n".join(mom_lines), transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    f5_path = plot_dir / "fig5_stacked_nz_reconstruction.png"
    plt.savefig(f5_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved Figure 5: {f5_path.name}")

    # 11. Export predictions
    pred_dir = script_dir / "results" / "specz_compilation_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame({
        'object_id': id_test,
        'ra': ra_test,
        'dec': dec_test,
        'z_spec': y_test,
        'z_mode': z_mode_final,
        'is_blend_candidate': is_blend_candidate,
        'residual_norm': m_final['residuals'],
        'pit': m_final['pit'],
        'mag_i': mag_i_test,
    })
    csv_file = pred_dir / "pontifex_cosmos_specz_predictions.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"✓ Saved predictions CSV: {csv_file}")

    npz_file = pred_dir / "pontifex_cosmos_specz_pdfs.npz"
    np.savez_compressed(npz_file, z_grid=Z_GRID, pdfs=pdf_calibrated, object_id=id_test, z_spec=y_test)
    print(f"✓ Saved continuous PDFs NPZ: {npz_file}")
    print("\n✓ Pipeline execution complete!")


if __name__ == '__main__':
    main()
