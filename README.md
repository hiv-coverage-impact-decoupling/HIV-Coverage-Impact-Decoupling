# HIV-1 Epidemiological Divergence Index (EDI) Pipeline

## Overview

This repository contains the automated computational and statistical pipeline for deriving, validating, and assessing the **Epidemiological Divergence Index (EDI)** for comparative global HIV surveillance.

The EDI is a standardized residual-based surveillance metric that quantifies divergence between **estimated HIV infection counts and model-based expected infection counts** under a common epidemiological benchmark. The benchmark incorporates ART coverage, epidemic history, calendar time, GDP per capita, and health expenditure, with population included as a count-model offset.

EDI is designed as a **comparative surveillance and hypothesis-generation tool**, not as a causal measure of ART effectiveness, health-system performance, or programmatic success or failure.

The framework further classifies countries according to ART coverage and epidemiological divergence to identify potential **coverage–impact surveillance discordance**, including settings characterized by comparatively high ART coverage and higher-than-expected HIV infection counts.

---

## Pipeline Architecture

The analytical pipeline enforces strict temporal separation between benchmark development and prospective validation.

- **Training period:** 1990–2015
- **Historical EDI assessment:** 2010–2015
- **Prospective validation period:** 2016–2022

Post-2015 information is not used to estimate the primary benchmark parameters.

### 1. Core EDI Derivation

#### Data Compilation & Preprocessing (`01`, `02`)

The pipeline assembles country-year data from UNAIDS HIV estimates and World Bank Open Data.

The preprocessing module:

- constructs the harmonized country-year panel;
- enforces the 1990–2015 training boundary;
- performs within-country interpolation and boundary filling only within the historical training window;
- constructs lagged variables chronologically; and
- retains missing lagged observations rather than imputing them.

Outcome estimates are not imputed.

#### EDI Core Benchmark (`03`)

A pooled **Negative Binomial type 2 (NB2)** regression is fitted by maximum likelihood to the eligible 1990–2015 training observations.

The benchmark incorporates:

- two-year-lagged ART coverage;
- calendar time;
- one-year-lagged HIV incidence rate;
- log-transformed GDP per capita;
- current health expenditure; and
- log population as an offset.

The model produces expected HIV infection counts. Annual EDI is then calculated as the standardized Pearson residual:

\[
EDI_{it}
=
\frac{Y_{it}-\hat{\mu}_{it}}
{\sqrt{\hat{\mu}_{it}+\hat{\alpha}\hat{\mu}_{it}^{2}}}.
\]

Historical country-level EDI is defined as the mean annual EDI during 2010–2015.

The primary NB2 coefficients and historical EDI outputs are treated as the **frozen benchmark artifacts** for subsequent validation and robustness analyses.

#### Coverage–Impact Decoupling (`04`)

The frozen historical EDI is cross-classified with mean ART coverage during 2010–2015.

Countries are assigned to four surveillance categories according to the global median ART coverage and median historical EDI:

1. Expected Alignment
2. Unexpected Resilience
3. Expected Vulnerability
4. Surveillance Blind Spot

Quadrant IV represents comparatively high ART coverage combined with higher-than-median epidemiological divergence.

---

## 2. Validation & Robustness Suite

### Strict Prospective Validation (`05`)

Historical EDI is evaluated prospectively against subsequent changes in newly estimated HIV infections during 2016–2022.

The primary validation model uses:

- ordinary least squares;
- HC3 heteroskedasticity-robust standard errors;
- baseline epidemic burden adjustment;
- 1,000-permutation testing; and
- Kruskal–Wallis comparison across EDI quartiles.

The validation is temporally separated from benchmark estimation and is intended to evaluate the **prospective construct validity of the EDI signal**, rather than to establish a causal effect.

### Temporal Rank Stability (`05`)

The frozen benchmark is projected across overlapping temporal windows:

- 2005–2012
- 2008–2015
- 2010–2017
- 2012–2019
- 2015–2022

Adjacent-window stability is evaluated using:

- Spearman rank correlation;
- Kendall's τ; and
- overlap among the 20 highest-EDI countries.

### Uncertainty Propagation (`07`)

UNAIDS uncertainty intervals are propagated using 1,000 Monte Carlo simulations.

Each successful iteration refits the NB2 benchmark and recalculates country-level EDI rankings. Rank preservation is summarized using Spearman correlation relative to the frozen primary EDI ranking.

### Advanced Robustness (`08`)

The advanced robustness suite includes:

- alternative NB2 and Poisson specifications;
- sensitivity to removal of the lagged-incidence covariate;
- specification-curve analysis;
- 5% symmetric tail trimming;
- Huber robust regression;
- Cook's distance influence diagnostics;
- leave-one-country-out sensitivity analysis;
- regional exclusion of Southern African countries; and
- simulation-based circularity stress testing.

These analyses evaluate the structural stability and interpretability of the EDI benchmark rather than replacing the primary frozen specification.

---

## 3. Auditing & Rendering

### Visualization (`06`, `10`)

The visualization modules generate publication-ready outputs including:

- global distribution of historical EDI; and
- the Coverage–Impact Decoupling Matrix.

Supplementary visualization modules generate figures for methodological and robustness analyses.

### Integrity Audit (`11`)

The final integrity audit verifies:

- strict 1990–2015 training boundaries;
- positivity of the NB2 dispersion parameter;
- absence of invalid model coefficients;
- mathematical consistency of the Pearson-residual EDI calculation;
- absence of infinite EDI values; and
- completeness of the publication output package.

---

## Execution

The master orchestrator (`run_pipeline.py`) controls sequential module execution and terminal logging.

### Main analysis

Runs the primary EDI derivation and core results:

```bash
python run_pipeline.py
