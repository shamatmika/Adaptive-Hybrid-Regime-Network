# Adaptive Hybrid Regime Network (AHRN)

**A regime-aware machine-learning framework for monthly cross-sectional return forecasting on S&P 500 equities.**

AHRN studies a specific question: **does adding market-regime information and cross-sectional attention to a stock-level forecasting model improve out-of-sample portfolio performance after transaction costs?**

The project is built as a controlled model comparison rather than a single-model backtest. The same universe, target definition, portfolio construction rules, cost model, and walk-forward evaluation are applied across linear, factor, regime-aware, recurrent, tree-based, attention, and AHRN variants.

**Authors:** Abinesh S. and Shamatmika R.  
**Institution:** CHRIST (Deemed to be University)

---

## What AHRN is trying to do

At each month-end, the pipeline uses information available up to that point for each eligible S&P 500 security. The model forecasts the security's **next-month excess return**. Securities are then ranked by the forecast and converted into a long-short portfolio.

The central AHRN architecture combines three pieces of information:

1. **Stock history** — a sequence of recent monthly characteristics for each security.
2. **Cross-sectional relationships** — multi-head self-attention allows securities in the same rebalance month to interact through a shared representation.
3. **Market regime** — a Gaussian Hidden Markov Model converts market-level variables into probabilistic regime states, which gate the AHRN forecast experts.


### Pipeline

```text
Bloomberg-derived panel
        │
        ▼
Data cleaning + point-in-time feature handling
        │
        ├───────────────┐
        ▼               ▼
Market regime HMM    Stock feature sequences
        │               │
        └───────┬───────┘
                ▼
       Model matrix / forecasting
                │
                ▼
       Next-month return forecasts
                │
                ▼
       Cross-sectional ranking
                │
                ▼
      Top quintile / bottom quintile
                │
                ▼
       Portfolio + transaction costs
                │
                ▼
       Fold-level performance
                │
        ┌───────┴────────┐
        ▼                ▼
 Statistical tests     SHAP / reports
```

---

## The data

The underlying data was derived from **Bloomberg Terminal** and is kept private rather than committed to this repository. The public repository contains the code and documentation needed to work with the same data layout without distributing the underlying licensed data.

The primary modelling file is:

```text
data/raw/SP500_master_panel.csv
```

The master panel combines security-level market data, liquidity measures, valuation and profitability characteristics, and market-wide variables. The modelling pipeline works at monthly frequency.

The associated data package contains:

| Dataset | Role in the project |
|---|---|
| `SP500_master_panel.csv` | Main security-month modelling panel |
| `SP500_macro_panel.csv` | Market, volatility, rates and credit variables used to construct the regime environment |
| `SP500_sectors_clean.csv` | Sector classification used when working with the equity universe |
| `OHLC.xlsx` | Historical daily OHLC data used in the underlying market-data preparation |
| `SPX as of *.xlsx` | Historical S&P 500 constituent/weight snapshots used to reconstruct the investment universe |

The complete data construction and Bloomberg extraction workflow is documented in this repository.

### Private data setup

Because the Bloomberg-derived dataset is private, it is deliberately outside Git. After obtaining access to the project dataset, place the required files under:

```text
data/raw/
```

At minimum, the full experiment requires:

```text
data/raw/SP500_master_panel.csv
```

The `.gitignore` prevents the raw data from being committed accidentally.

---

## Target and feature construction

The prediction target is the **next-month excess return**.

For a security \(i\) at month \(t\):

\[
RF_{t}^{monthly} = \frac{RF_t}{1200}
\]

\[
Target_{i,t} = Return_{i,t+1} - RF_{t+1}^{monthly}
\]

The target is only retained when the next observation is exactly the following calendar month. This prevents a missing month in a security's history from being treated as a valid one-month-ahead label.

The default model feature set is:

```text
momentum_12_1
price_to_book
px_to_sales
oper_margin
market_cap
vix
move
credit_spread
ust10y
curve_2y10y
fed_funds
bid_ask_spread
```

Within each walk-forward training fold, numerical preprocessing is fitted on the training period only. Training medians are used for feature imputation and the scaler is then applied to the corresponding test period.

For selected quarterly-style characteristics, the pipeline allows previously available observations to be carried forward for up to three months. The sequence builder also restricts every historical sequence to observations dated at or before the prediction date.

---

## Model matrix

AHRN is evaluated as part of a model matrix so that each architectural addition can be isolated.

| ID | Model | Purpose |
|---|---|---|
| **B0** | Historical Mean | Simple unconditional return baseline |
| **B1** | Ridge Regression | Primary linear benchmark |
| **B2** | Factor Model | Fixed momentum/value/quality characteristic rule |
| **B3** | Regime Ridge | Ridge models weighted by HMM regime probabilities |
| **B4-LSTM** | LSTM | Sequential baseline without attention or regime gating |
| **B4-GRU** | GRU | Sequential baseline using a GRU |
| **B5-RF** | Random Forest | Nonlinear tree ensemble baseline |
| **B5-XGB** | XGBoost | Gradient-boosting baseline |
| **I1** | Regime-Input | Tests whether explicit HMM probabilities help a conventional model |
| **I2** | Attention-Only | Tests attention without regime gating |
| **A1** | Core AHRN | Recurrent representation + attention + regime-gated experts |
| **A2** | AHRN + cost regularisation | Adds regime-dependent training regularisation |
| **A3** | A2 + uncertainty sizing | Adds entropy-based portfolio scaling |
| **A4** | A3 + stacking | Adds a fold-trained Ridge meta-layer |

This structure is important because it separates three different questions:

- **Does nonlinear modelling help?** B1 vs. B5.
- **Does regime information help?** B1/I1/B3 vs. AHRN components.
- **Does the complete architecture add value beyond its individual components?** A1 vs. I1/I2 and A2–A4.

The implementation is in [`ahrn/models.py`](ahrn/models.py).

---

## How the AHRN model works

### 1. Market-regime model

A Gaussian HMM is fitted to market-level variables including:

```text
VIX
MOVE
credit spread
2Y–10Y curve
```

The HMM produces a probability vector for each month:

\[
p_t = [P(S_t=1), P(S_t=2), \ldots, P(S_t=K)]
\]

The default configuration uses three states.

These probabilities are used directly by the regime-aware models rather than converting the market environment into a single hard-coded label.

### 2. Stock sequence

For each security and prediction month, the pipeline constructs a sequence of the previous 12 monthly observations by default.

The sequence is passed through a GRU representation layer.

### 3. Cross-sectional attention

At a given rebalance month, the latest representation for each eligible stock becomes an attention token. Multi-head self-attention allows the model to represent relationships across the contemporaneous cross section.

### 4. Regime-gated experts

A1 produces one forecast from each regime-specific expert. The final forecast is the probability-weighted combination:

\[
\hat y_i = \sum_{k=1}^{K} p_k f_k(x_i)
\]

where \(p_k\) is the HMM probability for regime \(k\) and \(f_k\) is the corresponding forecast expert.

A2–A4 then test additional components rather than treating them as part of the basic AHRN definition.

---

## Walk-forward evaluation

The default experiment uses a monthly rolling evaluation:

- **60 months** training
- **1 month** purge gap
- **12 months** out-of-sample testing
- **12 months** forward step
- minimum **6 test months** for a retained fold

The procedure is implemented in [`ahrn/data.py`](ahrn/data.py).

The important constraint is that the test period is never used to fit the preprocessing or forecasting model for that fold.

For each fold:

1. Select the historical training window.
2. Apply the one-month purge gap.
3. Build the future test window.
4. Determine eligible securities from the training period.
5. Fit feature preprocessing using training data only.
6. Fit the HMM using the training regime observations.
7. Build stock histories using observations available by each prediction date.
8. Fit every model in the matrix.
9. Generate test-period forecasts.
10. Convert forecasts into portfolios.
11. Apply transaction costs.
12. Store fold-level metrics and predictions.

This makes the model comparison a sequence of future-facing tests rather than a single train/test split.

---

## Portfolio construction

The economic evaluation is deliberately separated from the forecasting model.

At each rebalance date:

1. Remove the lowest-liquidity 5% of eligible names according to the configured liquidity rule.
2. Rank the remaining securities by predicted excess return.
3. Buy the top 20%.
4. Short the bottom 20%.
5. Apply a maximum 2% position weight per security.
6. Calculate turnover from the change in portfolio weights.
7. Apply the same transaction-cost framework across models.

The resulting portfolio return is evaluated rather than treating prediction error alone as the final objective.

This distinction matters in cross-sectional return forecasting: a model can have a lower prediction error without producing a better ranking of the stocks that form the long and short legs.

---

## Transaction costs

The primary evaluation uses the dynamic **Role A** cost estimate. It is kept separate from model-training regularisation.

For security \(i\), the cost proxy combines half the quoted spread with a market-data-only impact term:

\[
Cost_i = \frac{Spread_i}{2} + 0.0001(0.10)\sqrt{\frac{MarketCap_i}{DollarVolume_i}}
\]

with the ratio bounded before the square-root calculation and the final cost bounded by the configured maximum.

Short positions additionally incur the configured annual short-borrow fee.

A separate regime-scaled cost tier is available as a sensitivity analysis. It is not used to redefine the primary model comparison.

The implementation lives in [`ahrn/data.py`](ahrn/data.py) and the portfolio calculations are in [`run_experiment.py`](run_experiment.py).

---

## What is evaluated

The experiment records both forecasting and portfolio-level information.

### Forecasting

- out-of-sample prediction error
- predicted long-short spread
- model-by-fold predictions

### Portfolio

- gross return
- net return
- net Sharpe ratio
- maximum drawdown
- turnover
- cost impact

### Statistical analysis

The experiment also prepares:

- fold-level A1 vs. B1 paired comparisons
- Diebold-Mariano forecast comparisons
- secondary model comparisons
- ablation comparisons with Bonferroni adjustment
- Benjamini-Hochberg handling for the secondary comparison set
- crisis-window analysis
- stationary-bootstrap confidence intervals

The statistical and reporting functions are in [`ahrn/stats.py`](ahrn/stats.py) and [`ahrn/report.py`](ahrn/report.py).

---

## Interpretability

SHAP analysis is included so that model performance is not treated as a black box.

For tree models, the pipeline uses tree-based SHAP explanations. For A1, the implementation attempts gradient-based SHAP attribution over the sequence representation and reports feature attribution by inferred regime.

Generated attribution files are written to `results/` during an experiment.

---

## Repository structure

```text
AHRN/
├── ahrn/
│   ├── data.py              # Data preparation, targets, costs and walk-forward folds
│   ├── models.py            # HMM, baseline models and AHRN variants
│   ├── stats.py             # Statistical tests and confidence intervals
│   ├── report.py            # Result summaries and reporting
│   └── __init__.py
│
├── configs/
│   ├── default.yaml         # Full experiment configuration
│   └── smoke.yaml           # Small configuration for pipeline checks
│
├── datasets/
│   ├── DATASET_DESCRIPTION.md
│   ├── BLOOMBERG_DATA_DICTIONARY_AND_FORMULAS.md
│   └── BLOOMBERG_EXTRACTION_GUIDE.md
│
├── notebooks/
│   └── 01_one_fold_validation.ipynb
│
├── tests/
│   └── test_data.py
│
├── data/
│   └── raw/                 # Private Bloomberg-derived data; ignored by Git
│
├── results/                 # Generated experiment output; ignored by Git
│
├── run_experiment.py        # Main experiment entry point
├── PROTOCOL.md              # Evaluation protocol
├── DEVELOPER_HANDOFF.md     # Implementation notes
├── VERIFICATION.md          # Environment and verification status
├── requirements.txt
└── README.md
```

---

## Running the project

### 1. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Add the private dataset

Place the Bloomberg-derived modelling panel at:

```text
data/raw/SP500_master_panel.csv
```

The dataset is intentionally excluded from Git because it is derived from Bloomberg data.

### 3. Run the smoke configuration

Start with the smaller configuration:

```bash
python run_experiment.py --config configs/smoke.yaml
```

This uses fewer models, a shorter test window and one training epoch so that the complete pipeline can be checked before starting the full experiment.

### 4. Run the full experiment

```bash
python run_experiment.py --config configs/default.yaml
```

The default run uses the 60-month / 1-month purge / 12-month test walk-forward design and the complete model matrix.

### 5. Run the tests

```bash
pytest -q
```

---

## Validation notebook

[`notebooks/01_one_fold_validation.ipynb`](notebooks/01_one_fold_validation.ipynb) is a cleaned one-fold validation artifact used to inspect the modelling workflow on a single evaluation window.

It is useful for understanding and checking the pipeline, but it should not be interpreted as a substitute for the full multi-fold walk-forward experiment. The full experiment is defined by `configs/default.yaml` and `run_experiment.py`.

---

## Outputs

A full run writes its outputs to `results/`, including:

```text
results/
├── frozen_config.yaml
├── cost_tier_scoreboard.csv
├── run_manifest.json
├── fold_XX_<model>_<cost_tier>.json
├── shap_*.csv
├── shap_*_warning.txt       # only when an attribution run fails
└── report / tear-sheet outputs
```

The exact set of files depends on which models and attribution paths complete successfully.

Generated results are intentionally ignored by Git so that a repository checkout contains code and methodology rather than a machine-specific result dump.

---

## Current research status

The repository contains both the complete multi-fold experiment implementation and a one-fold validation notebook.

The one-fold evaluation is useful as a working reference for the model matrix, but it is not sufficient to establish a stable multi-period performance conclusion. The intended analysis is the full walk-forward run, where the A1/B1 comparison and the secondary model and ablation tests are evaluated across independent out-of-sample folds.

This distinction is important to the project: the purpose of the experiment is to test whether the additional complexity of AHRN survives the same out-of-sample and cost-aware evaluation applied to the simpler benchmarks, not to assume that the proposed architecture must win.

---

## Authors

**Abinesh S.**  
**Shamatmika R.**

CHRIST (Deemed to be University), Bengaluru, India

---
## Citation

If you use AHRN, its implementation, or the accompanying methodology in academic or research work, please cite:

```bibtex
@software{ahrn2026,
  author       = {Abinesh S. and Shamatmika R.},
  title        = {Adaptive Hybrid Regime Network (AHRN)},
  year         = {2026},
  institution  = {CHRIST (Deemed to be University), Bengaluru, India},
  note         = {Regime-aware machine-learning framework for monthly cross-sectional return forecasting on S\&P 500 equities}
}
```

## License

This repository is released under the **MIT License**.

The source code may be used, modified, and redistributed under the terms of the license below. The Bloomberg-derived datasets are **not included** in this repository and remain subject to their respective data-provider terms and restrictions.

```text
MIT License

Copyright (c) 2026 Abinesh S. and Shamatmika R.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```