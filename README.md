# Statistics and Probability for Data Science

A rigorous, executable curriculum covering statistics and probability from first principles to advanced ML-relevant techniques — implemented entirely in Python across 26 Jupyter notebooks. Every concept is backed by mathematical explanation, working code, visualisations, and an explicit connection to how it is used in machine learning and data science in practice.

---

## Overview

This repository is structured as a self-contained course. It is designed for:

- **Data science practitioners** who want to strengthen their statistical foundations
- **ML engineers** who want to understand *why* training loss, regularisation, and ensembles work — not just *how* to call them
- **Interview candidates** preparing for quantitative data science or ML roles
- **Students** transitioning from theoretical coursework to applied Python implementations

The curriculum progresses from probability fundamentals through classical inference, then into modern techniques (bootstrap, MLE, Bayesian inference, causal inference, A/B testing), and closes with a dedicated bridge to machine learning and a comprehensive interview preparation notebook.

---

## Repository Structure

```
Statistics-and-Probability-For-Data-Science/
│
├── 01–07   Foundations          Combinatorics, probability, distributions, descriptive statistics
├── 08–11   Inference Setup      CLT, sampling, hypothesis testing, parametric test selection
├── 12–16   Classical Tests      Z-test, t-test, ANOVA, chi-square, effect size & power
├── 17      Real Dataset         End-to-end applied walkthrough (Ames Housing)*
├── 18–24   Advanced Topics      Bootstrap, MLE, Bayesian, multiple testing, correlation,
│                                A/B testing, causal inference
├── 25      ML Bridge            How every statistical concept maps to ML/deep learning
├── 26      Interview Prep       30+ questions with worked answers and code challenges
│
├── data/                        Reference images; Kaggle dataset folder (CSV not included)*
├── requirements.txt             Python dependency specifications
├── .github/workflows/ci.yml     GitHub Actions CI — executes all notebooks on push/PR
└── LICENSE                      MIT
```

*Notebook 17 requires the Kaggle Ames Housing `train.csv` (not bundled). All other 25 notebooks are fully self-contained.

---

## Notebooks

### Part 1 — Foundations

| # | Title | Key Concepts | Code Highlights |
|---|-------|-------------|-----------------|
| 01 | Permutations and Combinations | Factorial, nPr, nCr, combinations with replacement | `itertools.permutations`, `itertools.combinations` |
| 02 | Probability and Rules of Probability | Sample space, axioms, addition/multiplication rules, conditional probability, Law of Large Numbers | LLN coin-flip convergence simulation (10,000 flips) |
| 03 | Bayes Theorem | Prior, likelihood, posterior, base-rate neglect | Medical test posterior calculation (~25.6%), prior vs posterior bar chart |
| 04 | Variables and Data Types | Nominal, ordinal, discrete, continuous; measurement scales | `pd.Categorical`, dtype-aware visualisations |
| 05 | Probability Distributions | Bernoulli, Binomial, Poisson, Normal, Uniform, Geometric, Exponential, F-distribution | 68-95-99.7 rule, PMF bar charts, 4-panel distribution comparison |
| 06 | Measures of Central Tendency | Mean, median, mode, geometric mean, harmonic mean, percentiles, IQR | Skewness demo showing mean vs median divergence |
| 07 | Measures of Variability | Range, variance, standard deviation, skewness, kurtosis | `scipy.stats.skew`, `kurtosis`; outlier sensitivity comparison |

### Part 2 — Statistical Inference

| # | Title | Key Concepts | Code Highlights |
|---|-------|-------------|-----------------|
| 08 | Central Limit Theorem | CLT with finite-variance requirement, t vs z CI formula | CI derivation; why t-distribution, not z, when σ is unknown |
| 09 | Sampling and Sampling Errors | SRS, stratified, cluster; sampling vs coverage vs nonresponse error | SRS vs stratified comparison; sampling error vs n convergence plot |
| 10 | Hypothesis Testing | H₀/H₁, p-values, Type I/II errors, significance level | Step-by-step z-test; p-value misconception callout; 10,000-experiment false-positive simulation |
| 11 | Parametric Tests | Normality assumption; when to use parametric vs non-parametric | Shapiro-Wilk on normal vs skewed data; printed decision flowchart |

### Part 3 — Classical Hypothesis Tests

| # | Title | Key Concepts | Code Highlights |
|---|-------|-------------|-----------------|
| 12 | Z-Test | One-sample, two-sample, proportion tests; 95% CIs | CIs for all test types; Wilson CI for proportions |
| 13 | t-Test | One-sample, Welch's independent, paired; normality checks | Shapiro-Wilk pre-check; Cohen's d; Welch-Satterthwaite CI; Mann-Whitney U alternative |
| 14 | ANOVA | One-way, two-way; assumption checks; effect size; post-hoc | Levene's test; η² (eta-squared); Tukey HSD; Kruskal-Wallis non-parametric alternative |
| 15 | Chi-Square Test | Independence test; goodness-of-fit; effect size | Cramér's V; goodness-of-fit for the shop customer example |
| 16 | Effect Size and Statistical Power | Cohen's d; power analysis; sample size determination | `statsmodels.TTestIndPower`; power curves for small/medium/large effects |
| 17 | Statistical Tests (Summarised) | Applied walkthrough on real regression data | *(Requires Kaggle Ames Housing `train.csv`)* |

### Part 4 — Advanced Topics

| # | Title | Key Concepts | Code Highlights |
|---|-------|-------------|-----------------|
| 18 | Bootstrap Resampling | Percentile CI; bootstrap hypothesis test; no distributional assumptions | CI for mean and median (10,000 resamples); comparison with parametric t-interval |
| 19 | Maximum Likelihood Estimation | Likelihood function; log-likelihood; analytical and numerical MLE | MLE for Normal and Exponential; log-likelihood surface contour plot; `scipy.optimize.minimize` |
| 20 | Bayesian Statistics | Prior, likelihood, posterior; conjugate priors; credible intervals | Beta-Binomial model; prior vs posterior overlay; 5-panel sequential updating animation |
| 21 | Multiple Testing and FDR | FWER inflation; Bonferroni; Benjamini-Hochberg; false discovery rate | FWER growth curve; 1,000-simulation FDR study; `statsmodels.multipletests` |
| 22 | Correlation Analysis | Pearson, Spearman, Kendall; outlier sensitivity; partial correlation | Anscombe's Quartet; correlation matrix heatmap; outlier effect demonstration |
| 23 | A/B Testing | Experiment design; MDE; sample size; peeking problem | End-to-end simulated experiment; peeking simulation (false positives inflate 5% → ~20%) |
| 24 | Causal Inference | Confounding; Simpson's Paradox; DiD; observational vs experimental | Simpson's Paradox simulation; Difference-in-Differences with counterfactual visualisation |

### Part 5 — ML Bridge and Interview Preparation

| # | Title | Key Concepts | Code Highlights |
|---|-------|-------------|-----------------|
| 25 | Statistics for Machine Learning | MLE = cross-entropy loss; MAP = regularisation; CLT in SGD; bootstrap = bagging | Logistic regression as MLE; L2 as Gaussian prior; gradient noise vs batch size; Random Forest bagging; paired t-test for model comparison; filter-method feature selection |
| 26 | Interview Questions and Practice Problems | 30+ questions across 8 topic areas | Birthday paradox; bias-variance trade-off; Welch's t-test from scratch; bootstrap CI from scratch; Cohen's d from scratch |

---

## Installation

**Requirements:** Python 3.9 or higher.

```bash
# Clone the repository
git clone https://github.com/AmanRoland/Statistics-and-Probability-For-Data-Science.git
cd Statistics-and-Probability-For-Data-Science

# Install dependencies
pip install -r requirements.txt

# Register the Jupyter kernel
python -m ipykernel install --user

# Launch Jupyter
jupyter notebook
```

### Dependencies

| Package | Purpose | Min Version |
|---------|---------|-------------|
| `numpy` | Numerical computing, random number generation | ≥ 1.24 |
| `pandas` | Data manipulation, categorical types | ≥ 2.0 |
| `matplotlib` | Plotting and visualisations | ≥ 3.7 |
| `scipy` | Statistical distributions, tests, optimisation | ≥ 1.10 |
| `statsmodels` | Power analysis, Tukey HSD, multiple testing corrections | ≥ 0.14 |
| `jupyter` / `notebook` | Interactive notebook environment | ≥ 1.0 / ≥ 7.0 |

For notebook 25 (Statistics for Machine Learning), `scikit-learn` is also used and will be installed automatically if not already present.

---

## Usage

### Recommended Learning Path

```
Probability Basics     → 01 → 02 → 03 → 04 → 05 → 06 → 07
                                                          ↓
Inference              → 08 → 09 → 10 → 11
                                          ↓
Classical Tests        → 12 → 13 → 14 → 15 → 16
                                                ↓
Advanced Methods       → 18 → 19 → 20 → 21 → 22 → 23 → 24
                                                          ↓
ML Applications        → 25 (Statistics for ML)
                          ↓
Interview Preparation  → 26
```

Notebook 17 is an optional applied capstone requiring the [Kaggle Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). Download `train.csv` and place it at `data/house-prices-advanced-regression-techniques/train.csv`.

### Running a Single Notebook

```bash
python -m nbconvert --to notebook --execute --inplace "18. Bootstrap Resampling.ipynb"
```

### Running the Full Suite

```bash
for nb in [0-9]*.ipynb; do
  [[ "$nb" == "17."* ]] && continue  # skip (external data)
  python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=300 "$nb"
done
```

---

## Key Features

### Executable Throughout
Every notebook runs top-to-bottom without errors. All random operations use `numpy.random.default_rng(seed=42)` for reproducibility.

### Mathematically Rigorous
- CI formula uses t-distribution (not z) when σ is unknown
- CLT statement includes the finite-variance requirement (Cauchy counterexample cited)
- ANOVA alternative hypothesis written as "at least one μᵢ differs" (not the mathematically incorrect μ₁ ≠ μ₂ ≠ μ₃)
- Bernoulli and Poisson PMF formulas stated precisely
- One-tailed tests use `scipy`'s `alternative=` parameter rather than dividing p by 2

### Assumption Checks Before Every Test
- **Shapiro-Wilk** normality test before t-tests and ANOVA
- **Levene's test** for variance homogeneity before ANOVA
- Non-parametric alternatives provided when assumptions fail: Mann-Whitney U (t-test), Kruskal-Wallis (ANOVA)

### Effect Sizes Alongside Every p-Value
| Test | Effect Size Reported |
|------|---------------------|
| t-test | Cohen's d with interpretation (small/medium/large) + 95% CI for mean difference |
| ANOVA | η² (eta-squared) |
| Chi-square | Cramér's V |
| Proportions | Cohen's h |
| Non-parametric | Rank-biserial r |

### ML/AI Connections
Every notebook ends with a dedicated section linking the statistical concept to where it appears in machine learning practice. Examples:

- *Distributions* → probabilistic generative models, weight initialisation (He/Xavier)
- *MLE* → cross-entropy training loss, logistic regression, EM algorithm
- *Bayesian MAP* → L1 (Laplace prior) and L2 (Gaussian prior) regularisation
- *Bootstrap* → Random Forests, bagging, out-of-bag error estimation
- *CLT* → mini-batch gradient descent stability
- *Multiple testing* → feature selection across thousands of features
- *A/B testing* → online experimentation at scale (Google, Meta, Netflix)
- *Causal inference* → debiasing ML models, uplift modelling, offline RL evaluation

---

## Continuous Integration

All 25 self-contained notebooks are executed automatically on every push and pull request via GitHub Actions (`.github/workflows/ci.yml`), testing against Python 3.10 and 3.11.

```yaml
# Triggered on: push to main/master, pull requests
# Matrix: Python 3.10, 3.11
# Steps: install deps → register kernel → execute all notebooks
```

To run the CI check locally:

```bash
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 "01. Permutation and Combinations.ipynb"
```

---

## Repository Statistics

| Metric | Count |
|--------|-------|
| Total notebooks | 26 |
| Self-contained (no external data) | 25 |
| Total cells | 392 |
| Lines of Python code | ~2,270 |
| Lines of explanatory prose | ~2,050 |
| Statistical tests demonstrated | 15+ |
| Visualisations produced | 40+ |
| Interview questions with answers | 30+ |

---

## Topics Covered

**Probability Theory**
Combinatorics · Sample spaces · Conditional probability · Bayes' theorem · Law of Large Numbers · Common distributions (Bernoulli, Binomial, Poisson, Normal, Geometric, Exponential, Uniform, F)

**Descriptive Statistics**
Measures of central tendency (mean, median, mode, geometric mean, harmonic mean) · Measures of variability (variance, SD, IQR, skewness, kurtosis) · Variable types and measurement scales

**Statistical Inference**
Central Limit Theorem · Confidence intervals · Sampling methods · Hypothesis testing framework · Type I/II errors · p-values and their correct interpretation

**Classical Tests**
Z-test (one/two-sample, proportions) · t-test (one-sample, Welch's independent, paired) · One-way and two-way ANOVA · Chi-square independence and goodness-of-fit · Non-parametric alternatives (Mann-Whitney U, Kruskal-Wallis, Wilcoxon)

**Modern Methods**
Bootstrap resampling · Maximum likelihood estimation · Bayesian inference and credible intervals · Multiple testing corrections (Bonferroni, Benjamini-Hochberg) · Correlation analysis · A/B testing · Causal inference (DiD, confounding, Simpson's Paradox)

**Effect Sizes and Power**
Cohen's d · η² (eta-squared) · Cramér's V · Cohen's h · Power curves · Minimum sample size determination · Post-hoc power analysis

**Statistics for ML**
MLE as neural network training · MAP as regularisation · CLT in stochastic gradient descent · Bootstrap in ensemble methods · Hypothesis tests for model comparison · Correlation-based feature selection · Bias-variance trade-off

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Copyright (c) 2021–2026 Aman Roland
