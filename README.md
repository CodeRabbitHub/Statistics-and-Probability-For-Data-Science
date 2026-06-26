# Statistics and Probability for Data Science

A comprehensive, executable curriculum covering statistics and probability from first principles through to modern ML-relevant techniques — implemented in Python and structured as runnable Jupyter notebooks.

---

## Quick Start

```bash
git clone https://github.com/your-username/Statistics-and-Probability-For-Data-Science.git
cd Statistics-and-Probability-For-Data-Science
pip install -r requirements.txt
jupyter notebook
```

**Python ≥ 3.9 required.**  
All notebooks (except NB17, which needs the Kaggle Ames Housing dataset) execute from top to bottom with no external data.

---

## Curriculum — Notebook Index

### Foundations

| # | Notebook | Topics |
|---|----------|--------|
| 01 | Permutations and Combinations | Factorial, nPr, nCr, combinatorial counting |
| 02 | Probability and Rules of Probability | Sample space, axioms, addition/multiplication rules, conditional probability, Law of Large Numbers |
| 03 | Bayes Theorem | Prior, likelihood, posterior, base-rate neglect, medical test example |
| 04 | Variables and Data Types | Nominal, ordinal, discrete, continuous — dtypes, visualisations |
| 05 | Probability Distributions | Bernoulli, Binomial, Poisson, Normal, Uniform, Geometric, Exponential, F |
| 06 | Measures of Central Tendency | Mean, median, mode, geometric mean, harmonic mean, IQR, skewness effects |
| 07 | Measures of Variability | Range, variance, standard deviation, skewness, kurtosis |
| 08 | Central Limit Theorem | CLT statement (with finite-variance condition), confidence intervals, t vs z |

### Inference

| # | Notebook | Topics |
|---|----------|--------|
| 09 | Sampling and Sampling Errors | SRS, stratified, cluster sampling; sampling error vs sample size |
| 10 | Hypothesis Testing | Null/alternative hypotheses, p-values, Type I/II errors, false-positive simulation |
| 11 | Parametric Tests | When to use parametric vs non-parametric; Shapiro-Wilk decision guide |
| 12 | Z-Test | One-sample, two-sample, proportion tests; 95% CIs for all test types |
| 13 | t-Test | One-sample, independent (Welch's), paired; Cohen's d; Mann-Whitney U alternative |
| 14 | ANOVA | One-way, two-way; Levene's test; η²; Tukey HSD post-hoc; Kruskal-Wallis |
| 15 | Chi-Square Test | Independence test; Cramér's V effect size; Goodness-of-Fit |
| 16 | Effect Size and Statistical Power | Cohen's d; power curves; minimum sample size via `TTestIndPower` |
| 17 | Statistical Tests (Summarised) | End-to-end real dataset walkthrough *(requires Kaggle Ames Housing CSV)* |

### Advanced Topics (New)

| # | Notebook | Topics |
|---|----------|--------|
| 18 | Bootstrap Resampling | CI for mean and median; bootstrap hypothesis test; parametric vs bootstrap comparison |
| 19 | Maximum Likelihood Estimation | Log-likelihood; analytical MLE for Normal and Exponential; numerical MLE with `scipy.optimize` |
| 20 | Bayesian Statistics | Beta-Binomial conjugate; prior/posterior visualisation; credible intervals; sequential updating |
| 21 | Multiple Testing and FDR | FWER inflation; Bonferroni; Benjamini-Hochberg FDR; 1,000-simulation study |
| 22 | Correlation Analysis | Pearson, Spearman, Kendall; outlier sensitivity; Anscombe's Quartet; partial correlation |
| 23 | A/B Testing | End-to-end experiment design; sample size; peeking simulation; practical vs statistical significance |
| 24 | Causal Inference | Confounding; Simpson's Paradox; Difference-in-Differences; RCTs vs observational methods |
| 25 | Statistics for Machine Learning | MLE = training loss; CLT in SGD; Bayesian regularisation; bootstrap ensembles; feature selection |
| 26 | Interview Questions and Practice | 30+ interview questions with worked answers across all topics |

---

## Learning Path

```
Beginner  → 01 → 02 → 03 → 04 → 05 → 06 → 07
                                            ↓
Inference → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 15 → 16
                                                        ↓
Advanced  → 18 → 19 → 20 → 21 → 22 → 23 → 24
                                            ↓
ML Ready  → 25 → 26 (Interview Prep)
```

---

## Prerequisites

- Python 3.9+
- Basic Python familiarity (lists, loops, functions)
- No prior statistics knowledge assumed

---

## Dependencies

```
numpy       pandas      matplotlib
scipy       statsmodels jupyter
```

Install with: `pip install -r requirements.txt`

---

## Fixes and Improvements Made

This repository has been comprehensively audited and improved:

**Mathematical Accuracy (Phase 1)**
- Corrected CI formula: z → t when σ unknown (NB08)
- Fixed CLT statement to include finite-variance requirement (NB08)
- Corrected ANOVA alternative hypothesis notation (NB14)
- Fixed Bernoulli PMF notation and added Poisson PMF formula (NB05)
- Replaced KDE plots on discrete data with correct bar chart PMFs (NB05)
- Fixed variable scoping bug in two-sample proportion test (NB12)
- Switched to Welch's t-test as default (`equal_var=False`) (NB13)
- Fixed one-tailed test using `alternative=` parameter instead of dividing by 2 (NB17)

**Code Quality (Phase 1)**
- All `set` literals replaced with `list` where order matters (NB01)
- Wildcard imports removed; explicit imports throughout (NB17)
- Deprecated `fillna(inplace=True)` replaced (NB17)
- `np.random.default_rng(seed=42)` used for reproducibility throughout
- Hardcoded paths replaced with `pathlib.Path` (NB17)

**Educational Content (Phases 2–3)**
- Executable code added to all 8 previously text-only notebooks
- Assumption checks (Shapiro-Wilk, Levene's) added before every test
- Effect sizes (Cohen's d, η², Cramér's V, rank-biserial r) added after every test
- 95% CIs added after every hypothesis test
- Non-parametric alternatives (Mann-Whitney U, Kruskal-Wallis) added throughout
- Tukey HSD post-hoc added to ANOVA

---

## CI / Testing

Every notebook (except NB17) is tested automatically on push via GitHub Actions.  
See `.github/workflows/ci.yml`.

To run locally:
```bash
for nb in *.ipynb; do
  python -m nbconvert --to notebook --execute --inplace "$nb"
done
```

---

## License

See [LICENSE](LICENSE).
