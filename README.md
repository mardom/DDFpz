# DDFpz: Photometric Redshift PDF Pipeline for Deep Drilling Fields

**DDFpz** is an ensemble photometric redshift probability density function (PDF) estimation and spatial calibration framework designed for Rubin Observatory LSST Deep Drilling Fields (DDFs) and the Nancy Grace Roman Space Telescope High Latitude Wide Area Survey.

The pipeline combines the **Pontifex (v2.2.0 / AionPlus)** architecture: a **Committee of Experts** integrating RAIL engines (`BPZ-lite`, `FlexZBoost`, `GPz`, `kNN`, `TrainNN`) alongside the deep transformer foundation model estimator (`AION-PZ`), paired with a dynamic **Mixture-of-Experts (MoE)** gating network and an **iterative Expectation-Maximization (EM) spatial clustering loop** for candidate blended systems.

---

## 5-Fold Cross-Validation Performance on Curated COSMOS Dataset

Evaluated across the **complete curated COSMOS spectroscopic compilation** ($N = 76{,}046$ galaxies, secure flags $\ge 3$, $0.01 \le z \le 3.0$, matched with CFHT/Subaru CIGALE photometry) using strict **5-Fold Stratified Cross-Validation Out-Of-Fold (OOF)** inference:

| DESC Science Requirements Metric | Tier 2: 5-Fold OOF MoE (RAIL + AION) | Tier 3: 5-Fold OOF MoE + EM Blends | Rubin LSST DESC SRD Requirement |
| :--- | :---: | :---: | :---: |
| **Photo-z Bias (Median)** | **-0.00103** | -0.00226 | $\le \pm 0.003$ (Y10) |
| **Biweight Location** | **-0.00086** | -0.00120 | $\le \pm 0.003$ |
| **Scatter $\sigma_{\rm MAD}$** | **0.01631** | 0.01935 | $\le 0.030$ (Y10) / $\le 0.050$ (Y1) |
| **Scatter $\sigma_{\rm IQR}$** | **0.01631** | 0.01954 | — |
| **Biweight Scale $\sigma_{\rm BW}$** | **0.01917** | 0.02158 | — |
| **Outlier Fraction ($\eta_{0.15}$)** | **5.87%** | 8.18% | $\le 10.0\%$ (Y10) / $\le 15.0\%$ (Y1) |
| **Severe Outlier ($\eta_{0.30}$)** | **3.44%** | 4.62% | — |
| **Mean PIT** | **0.4829** | 0.5564 | $\sim 0.50$ (Flat coverage) |
| **PIT Variance** | **0.0282** | 0.0420 | $1/12 \approx 0.0833$ |
| **PIT KS Distance ($D_{\rm KS}$)** | 0.21573 | **0.19421** | — |
| **Cramér-von Mises ($CvM$)** | 1359.44 | **974.46** | — |
| **PIT Outlier Rate** | **0.00%** | 0.36% | $\le 1.0\%$ |
| **Mean CRPS** | **0.09657** | 0.10545 | — |
| **Wasserstein Distance ($W_1$)** | 0.11189 | **0.06819** | — |
| **DESC SRD Mean Shift ($\delta\mu$)** | +0.11053 | **-0.06500** | Minimizing moment shift |
| **DESC SRD Dispersion Shift ($\delta\sigma$)** | +0.12720 | **-0.08541** | Minimizing dispersion shift |

---

## Publication Figures

All diagnostic figures generated from the complete 5-fold cross-validation out-of-fold inference are stored in `results/specz_compilation_plots/`:

1. **`figure1_zphot_vs_zspec_hexbin.png`**:
   - 2D density hexbin heatmap of $z_{\rm phot}$ vs. $z_{\rm spec}$ with $1:1$ diagonal line, $\pm 0.15(1+z_{\rm spec})$ outlier envelopes, log-scaled galaxy count colorbar, and comprehensive statistics summary box.
2. **`figure2_pit_diagnostics.png`**:
   - Probability Integral Transform (PIT) coverage distribution histogram compared to uniform expectation, alongside cumulative empirical PIT CDF vs. ideal diagonal with Kolmogorov-Smirnov ($D_{\rm KS}$) and Cramér-von Mises ($CvM$) metrics.
3. **`figure3_stacked_nz_ensemble.png`**:
   - Stacked ensemble photometric redshift distribution $\hat{N}(z)$ vs. true spectroscopic distribution $N_{\rm spec}(z)$ with moment shifts ($\delta\mu, \delta\sigma$) and residual distribution.
4. **`figure4_error_vs_redshift.png`**:
   - Scatter $\sigma_{\rm MAD}(z)$ and catastrophic outlier fraction $\eta_{0.15}(z)$ plotted across redshift bins from $z=0$ to $z=3$ against LSST DESC Y1 and Y10 performance targets.

---

## Key Architecture & Features

1. **Photometric Calibration & 44-D Feature Engineering**:
   - Converts optical/NIR flux densities ($f_\nu$ in mJy) to AB magnitudes with propagated uncertainties.
   - Generates adjacent-band colors, optical-infrared cross-colors, signal-to-noise ratios ($\text{SNR}_b \approx 1.086 / \sigma_{m_b}$), and quadrature color error metrics.
   - Resilient `sanitize_input_catalog` guard handling non-detections, negative errors, and extreme outliers.

2. **Full Committee of Heterogeneous Estimators**:
   - **`BPZ-lite`** (Bayesian SED template fitting with CWWSB/starburst templates).
   - **`FlexZBoost`** (Nonparametric conditional density estimator expanding in cosine bases via gradient boosted trees).
   - **`GPz`** (Sparse Gaussian Process regression with heteroscedastic input-dependent noise).
   - **`kNN`** (Color-magnitude nearest neighbors density estimator).
   - **`TrainNN` / MLPs** (`NN1` and `NN2` neural network classifiers).
   - **`AION-PZ`** (Astronomical foundation model transformer encoder latent token embeddings coupled to an MLP classifier head with PIT temperature recalibration).
   - **`MiniSom`** (Unsupervised Self-Organizing Map color density mapping).

3. **Dynamic Mixture of Experts (MoE) Gating**:
   - Distance-weighted local inverse-error gating in physical photometric and error space ($\tilde{\boldsymbol{x}}$), avoiding latent-space compression distortion while allocating optimal weights to each expert.

4. **Spatial Clustering EM Loop for Blended Galaxies**:
   - Identifies candidate blends and contaminated sources via 3D PCA Mahalanobis outlier distance ($\chi^2 > 7.81$) and PDF dispersion ($\sigma_{\rm PDF} > 0.15$).
   - Cross-correlates candidates against neighboring spectroscopic reference galaxies within $\theta < 2.5'$ in the celestial KD-Tree (`cKDTree`).
   - Modulates posterior PDFs iteratively via the spatial clustering likelihood:
     $$p_i^{(t+1)}(z) \propto p_i^{(t)}(z) \cdot \left[ L_{\rm spatial}(z) + \epsilon \right]^\alpha$$
     suppressing false secondary modes and reducing Wasserstein distance from $0.11189 \to 0.06819$.

---

## Directory Structure

```
DDFpz/
├── notebooks/
│   ├── specz_compilation_photoz_pipeline.ipynb   # Main interactive pipeline notebook
│   └── pontifex_tutorial.ipynb                  # Step-by-step tutorial notebook
├── run_specz_compilation_photoz.py              # Standalone CLI 5-fold CV execution script
├── src/                                         # Core Pontifex v2.2.0 library
│   └── pontifex/
│       ├── core/                                # Feature engineering, input guard, metrics
│       ├── pz/                                  # Estimators, AION, committee, EM loop
│       └── nz/                                  # Tomographic ensemble n(z) reconstruction
├── results/
│   ├── specz_compilation_metrics.csv            # Detailed DESC PZ benchmark comparison table
│   ├── specz_compilation_metrics.json           # Machine-readable evaluation metrics
│   ├── specz_compilation_predictions/           # Predictions CSV & compressed PDFs NPZ
│   │   ├── pontifex_cosmos_specz_predictions.csv
│   │   └── pontifex_cosmos_specz_pdfs.npz
│   └── specz_compilation_plots/                 # Publication-ready diagnostic figures (PNG)
│       ├── figure1_zphot_vs_zspec_hexbin.png
│       ├── figure2_pit_diagnostics.png
│       ├── figure3_stacked_nz_ensemble.png
│       └── figure4_error_vs_redshift.png
├── requirements.txt                             # Python dependencies
└── README.md                                    # Project documentation
```

---

## Execution Guide

To reproduce the complete 5-fold cross-validation run on the curated COSMOS sample:
```bash
python run_specz_compilation_photoz.py --folds 5
```

Optional arguments:
- `--folds <int>`: Number of cross-validation folds (default: 5).
- `--max-samples <int>`: Subsample size for rapid testing (default: all 76,046 galaxies).
- `--em-iterations <int>`: Spatial EM iterations for blend candidates (default: 4).
