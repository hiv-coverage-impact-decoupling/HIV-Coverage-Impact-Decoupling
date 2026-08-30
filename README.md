# HIV-1 Epidemiological Divergence Index (EDI) Pipeline

## Overview

Automated pipeline for deriving, validating, and stress-testing the **Epidemiological Divergence Index (EDI)** for comparative global HIV surveillance.

EDI is a standardized Pearson-residual measure of divergence between observed and model-expected HIV infection counts under a pooled Negative Binomial type 2 (NB2) benchmark.

The benchmark includes:

- 2-year-lagged ART coverage
- 1-year-lagged HIV incidence
- Calendar time
- Log GDP per capita
- Current health expenditure
- Population offset

EDI is intended for **surveillance prioritization and hypothesis generation**, not causal inference or assessment of ART program failure.

## Analytical Design

| Component | Period |
|---|---|
| Model training | 1990–2015 |
| Historical EDI | 2010–2015 |
| Prospective validation | 2016–2022 |
| Temporal stability | 2005–2012, 2008–2015, 2010–2017, 2012–2019, 2015–2022 |

The primary NB2 benchmark is frozen after training. Post-2015 data are not used to estimate the primary model.

## Pipeline

### Core analysis

`01_data_compiler.py`  
Builds the raw UNAIDS–World Bank country-year panel.

`02_time_safe_preprocessor.py`  
Applies time-safe preprocessing within 1990–2015 and constructs lagged variables.

`03_core_model.py`  
Fits the primary NB2 benchmark and generates frozen historical EDI.

The EDI is defined as:

$$
EDI_{it} =
\frac{Y_{it}-\hat{\mu}_{it}}
{\sqrt{\hat{\mu}_{it}+\hat{\alpha}\hat{\mu}_{it}^{2}}}
$$

`04_decoupling_analysis.py`  
Constructs the ART coverage–EDI surveillance matrix.

### Validation and robustness

`05_validation_suite.py`  
Performs prospective validation, temporal rank stability, Kendall's tau, and top-20 overlap analysis.

`07_supplementary_materials.py`  
Runs 1,000-iteration Monte Carlo uncertainty propagation.

`08_advanced_robustness.py`  
Runs specification, outlier, influence, regional, LOCO, likelihood, and circularity sensitivity analyses.

`06_visualizer.py` and `10_supplementary_extras.py`  
Generate manuscript and supplementary figures.

`11_integrity_audit.py`  
Performs final reproducibility and mathematical integrity checks.

## Execution

### Main analysis

```bash
Main analysis
`python run_pipeline.py`

Full publication analysis
`python run_pipeline.py --full`
