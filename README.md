# SABR Implied Volatility Correction with a Neural Network

A Python project that trains a neural network to correct the pricing error of the Hagan closed-form approximation under the SABR stochastic volatility model, using Monte Carlo simulation as the accuracy benchmark.

The corrected volatility is:

    sigma_corrected = sigma_Hagan + ANN_prediction

so the network learns the *residual* between the fast analytical approximation and the slow but accurate simulation, rather than learning to price options from scratch. The goal is Monte Carlo-level accuracy at closed-form-level speed.

## Why this matters

SABR has no exact closed-form option price. In practice desks either accept the bias of Hagan's asymptotic expansion, which is fast, or run Monte Carlo, which is accurate but far too slow to reprice a volatility surface repeatedly. Learning the correction term gets most of the accuracy at close to the analytical cost.

## Method

**Phase 1 - validate the pricing toolkit.** Before anything is learned, the Black, Bachelier and Hagan implementations are checked against published benchmarks:

| Check | Result |
|---|---|
| Hagan vs published benchmark, case A | 0.89 bp max difference |
| Hagan vs published benchmark, case B | 0.74 bp |
| Hagan vs published benchmark, case C | 0.73 bp |
| Hagan vs published benchmark, case D | 0.87 bp |
| Black / Bachelier price-to-implied-vol round trip | machine precision (1e-12 to 1e-15) |

**Phase 2 - generate the benchmark.** SABR implied volatilities are produced by Monte Carlo simulation under two dynamics:

- standard SABR with an absorbing boundary at F = 0
- free-boundary SABR, which admits negative forwards (relevant in negative-rate environments)

**Phase 3 - measure the error.** The difference between the Hagan approximation and the Monte Carlo benchmark is analysed across the parameter space.

**Phase 4 - learn the correction.** A neural network is trained on that error and added back to the Hagan volatility.

## Files

| File | What it is |
|---|---|
| `SABR_ANN_option_pricing.ipynb` | Full notebook: pricing toolkit, Monte Carlo benchmark, error analysis, network training |
| `SABR_ANN_option_pricing.html` | Rendered notebook - open this to read the analysis without running anything |
| `SABR_ANN_presentation.pdf` | Slide deck summarising method and results |

## Stack

Python, NumPy, SciPy (`stats.norm`, `optimize.brentq`), Matplotlib.

## Context

Individual project completed during the MSc in Finance (Quantitative and Financial Markets track) at emlyon business school.
