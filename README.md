# DDFpz: Photometric Redshift PDF Pipeline for Deep Drilling Fields

**DDFpz** is an ensemble photometric redshift probability density function (PDF) estimation and spatial calibration framework designed for Rubin Observatory LSST Deep Drilling Fields (DDFs) and the Nancy Grace Roman Space Telescope High Latitude Wide Area Survey.

The pipeline combines a **5-expert Mixture-of-Experts (MoE) committee** with an **iterative Expectation-Maximization (EM) spatial clustering loop** to break color-redshift degeneracies and resolve overlapping/blended sources.

---

## Key Architecture & Features

1. **Photometric Calibration & 44-D Feature Engineering**:
   - Converts optical/NIR flux densities ($f_\nu$ in mJy) to AB magnitudes with propagated uncertainties.
   - Generates Pogson fluxes, adjacent band colors, wide-baseline colors, and color error metrics.
   - Resilient input sanitation handling non-detections, negative errors, and extreme outliers.

2. **Full Committee of 5 Diverse Estimators**:
   - **Wide Deep MLP** (`MLPClassifier`, 128-64 units): High-capacity non-linear mapping.
   - **Very Deep MLP** (`MLPClassifier`, 256-128-64 units): Multi-layer hierarchical feature extractor.
   - **k-Nearest Neighbors KDE** (`NearestNeighbors`, $k=45$): Non-parametric local color-magnitude density posterior.
   - **Random Forest Ensemble** (`RandomForestRegressor`, 50 trees): Robust variance-weighted Gaussian PDF.
   - **Regularized Ridge Regression** (`Ridge`): Linear color track anchoring the tails.

3. **Mixture of Experts (MoE) Gating**:
   - Optimal convex combination of expert posteriors:
     $$p_{\rm MoE}(z) = \sum_{k=1}^5 w_k p_k(z)$$
     anchoring the neural network predictions with non-parametric and tree ensemble safeguards.

4. **Spatial Clustering EM Loop for Blended Galaxies**:
   - Identifies blended and contaminated sources via 3D PCA Mahalanobis outlier distance ($\chi^2 > 7.81$) and PDF variance ($\sigma_{\rm PDF} > 0.15$).
   - Cross-correlates identified candidates against neighboring spectroscopic galaxies ($\theta < 2.5'$) in the celestial KD-Tree.
   - Iterative EM updating rule:
     $$p_i^{(t+1)}(z) \propto p_i^{(t)}(z) \cdot \left[ L_{\rm spatial}(z) + \epsilon \right]^\alpha$$
     effectively suppressing false secondary modes and breaking redshift degeneracies.

5. **DESC PZ Data Challenge Metrics**:
   - Evaluates Point Metrics (Median Bias, Biweight Location, $\sigma_{\rm MAD}$, $\sigma_{\rm IQR}$, $\eta_{0.15}$, $\eta_{0.30}$).
   - Evaluates Distribution Metrics (PIT KS statistic, Cramér-von Mises $CvM$, $\delta\mu$, $\delta\sigma$).
   - Fully compliant with the Vera C. Rubin LSST Science Requirements Document (SRD).

---

## Directory Structure

```
DDFpz/
├── notebooks/
│   ├── specz_compilation_photoz_pipeline.ipynb   # Main interactive 36-cell pipeline notebook
│   └── pontifex_tutorial.ipynb                  # Synchronized step-by-step tutorial notebook
├── run_specz_compilation_photoz.py              # Standalone CLI execution script
├── src/                                         # Core Pontifex library and estimators
│   └── pontifex/
│       ├── core/                                # Feature engineering, sanitation, metrics
│       ├── pz/                                  # Estimators, blends, EM loop
│       └── nz/                                  # Ensemble tomographic n(z) reconstruction
├── results/
│   ├── specz_compilation_predictions/           # Output predictions CSV and continuous PDFs NPZ
│   └── specz_compilation_plots/                 # Publication-ready diagnostic figures (PNG)
├── requirements.txt                             # Python dependencies
└── README.md                                    # Project documentation
```

---

## Quickstart

### Environment Setup
```bash
pip install -r requirements.txt
```

### Option A: Run the Interactive Jupyter Notebook
Launch Jupyter and open:
```bash
jupyter notebook notebooks/specz_compilation_photoz_pipeline.ipynb
```

### Option B: Run Standalone CLI Script
Execute the entire pipeline and generate all diagnostic figures and metrics from the terminal:
```bash
python run_specz_compilation_photoz.py
```

Generated outputs will be saved to:
- `results/specz_compilation_predictions/pontifex_cosmos_specz_predictions.csv`
- `results/specz_compilation_predictions/pontifex_cosmos_specz_pdfs.npz`
- `results/specz_compilation_plots/*.png`
