# U.S. Macroeconomic Time Series Analysis

Academic course project using monthly Federal Reserve Economic Data (FRED) series from January 1995 onward.

## Research Question

How do U.S. macroeconomic conditions such as unemployment, interest rates, industrial production, and money supply relate to inflation over time?

The main target variable is monthly log CPI inflation, computed from the Consumer Price Index.

## Data

The project downloads monthly FRED series:

| Variable | FRED ID | Meaning |
| --- | --- | --- |
| CPI | `CPIAUCSL` | Consumer Price Index for All Urban Consumers |
| UNRATE | `UNRATE` | Civilian Unemployment Rate |
| FEDFUNDS | `FEDFUNDS` | Effective Federal Funds Rate |
| INDPRO | `INDPRO` | Industrial Production Index |
| M2 | `M2SL` | M2 Money Stock |
| UMCSENT | `UMCSENT` | University of Michigan Consumer Sentiment |

## Methodology

1. Download and align monthly time series from FRED.
2. Transform non-stationary level variables into stationary model variables:
   - CPI inflation: monthly log CPI growth times 100.
   - Industrial production growth: monthly log INDPRO growth times 100.
   - M2 growth: monthly log M2 growth times 100.
   - Consumer sentiment: first difference.
3. Explore time plots, rolling means, and correlations.
4. Run Augmented Dickey-Fuller stationarity tests.
5. Test whether macro variables Granger-cause inflation.
6. Build VAR and VARX forecasting systems for CPI inflation and related macro variables.
7. Compare out-of-sample forecast performance against machine-learning benchmarks.

The main notebook uses a graduate-style econometrics workflow:

1. ADF tests on raw and transformed variables.
2. Pairwise Engle-Granger cointegration check for non-stationary level variables.
3. Break dummies for the 2008 financial crisis and COVID period.
4. Detailed EDA: summary statistics, correlations, scatter plots, distributions, ACF/PACF, structural break inspection.
5. VAR lag selection using AIC, BIC, HQIC, and FPE.
6. VAR and VARX residual diagnostics and stability checks.
7. Visual residual autocorrelation diagnostics: ACF plots and residual ACF/CCF matrices with confidence bounds.
8. Residual normality diagnostics: Jarque-Bera, skewness, kurtosis, histograms, and Q-Q plots.
9. Parameter significance tables with classical, HC3, and HAC/Newey-West inference.
10. Heteroskedasticity, ARCH, Breusch-Pagan, and White-style checks where appropriate.
11. Model complexity and overparameterization warnings.
12. Parallel VAR and VARX architecture, lag selection, significance, diagnostics, and forecast evaluation.
13. Granger causality heatmap.
14. Structural impulse response functions, FEVD, alternative Cholesky-ordering robustness, and VARX conditional scenario responses.
15. Rolling and multi-horizon forecast evaluation against random-walk and ML benchmarks.
16. Crisis dummy, expanding-window, and regime-split robustness checks.

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set your FRED API key as an environment variable:

```bash
export FRED_API_KEY="your_key_here"
```

Then run the project:

```bash
python src/macro_time_series_analysis.py
python src/advanced_macro_var_analysis.py
```

Outputs are written to:

- `data/raw_fred_macro.csv`
- `data/processed_macro.csv`
- `outputs/figures/`
- `outputs/tables/`

The main notebook is:

- `macro_time_series_project.ipynb`

The interactive dashboard is:

- `dashboard_app.py`

Dashboard implementation logic is split across `src/data.py`, `src/models_var.py`, `src/models_varx.py`, `src/diagnostics.py`, `src/forecasting.py`, `src/visualization.py`, and `src/dashboard_helpers.py`; `dashboard_app.py` is kept mainly as the Streamlit UI layer.

Run it with:

```bash
streamlit run dashboard_app.py
```

The notebook is the official academic methodology/report artifact. The dashboard is the presentation and sensitivity-analysis layer for interactive VAR/VARX exploration.

Dashboard controls include:

- train/test split date,
- model type (`VAR` or `VARX`),
- endogenous and exogenous variable selection with validation,
- real-time AIC/BIC/HQIC lag selection,
- manual lag order,
- residual diagnostics,
- residual normality histograms and Q-Q plots,
- stability warnings,
- overparameterization warnings,
- VAR IRF/FEVD interpretation,
- VARX conditional/scenario shock responses,
- robustness tables for crisis dummies, regimes, Diebold-Mariano tests, and IRF ordering,
- VARX conditional forecast paths,
- ML forecast comparison.

Dashboard page order:

1. Overview
2. Stationarity and Data Preparation
3. Model Architecture and Direct Results
4. Forecast Comparison
5. Residual Diagnostics
6. Significance Analysis and Granger Causality
7. IRF and FEVD
8. Robustness
9. Code quality

Key generated academic outputs include:

- `outputs/tables/academic_final_model_architecture.csv`
- `outputs/tables/academic_cholesky_ordering.csv`
- `outputs/tables/academic_all_model_forecast_ranking.csv`
- `outputs/tables/academic_varx_lag_selection.csv`
- `outputs/tables/academic_var_residual_diagnostics.csv`
- `outputs/tables/academic_varx_residual_diagnostics.csv`
- `outputs/tables/academic_var_residual_acf_summary.csv`
- `outputs/tables/academic_varx_residual_acf_summary.csv`
- `outputs/tables/academic_var_parameter_significance.csv`
- `outputs/tables/academic_varx_parameter_significance.csv`
- `outputs/tables/academic_var_arch_tests.csv`
- `outputs/tables/academic_varx_arch_tests.csv`
- `outputs/tables/academic_var_residual_normality.csv`
- `outputs/tables/academic_varx_residual_normality.csv`
- `outputs/tables/academic_var_heteroskedasticity_tests.csv`
- `outputs/tables/academic_varx_heteroskedasticity_tests.csv`
- `outputs/tables/academic_var_parameter_significance_robust.csv`
- `outputs/tables/academic_varx_parameter_significance_robust.csv`
- `outputs/tables/academic_model_complexity_overparameterization.csv`
- `outputs/tables/academic_var_varx_diagnostic_comparison.csv`
- `outputs/tables/academic_multihorizon_forecast_comparison.csv`
- `outputs/tables/academic_diebold_mariano_tests.csv`
- `outputs/tables/academic_crisis_dummy_robustness.csv`
- `outputs/tables/academic_expanding_window_robustness.csv`
- `outputs/tables/academic_regime_split_comparison.csv`
- `outputs/tables/academic_irf_robustness_summary.csv`
- `outputs/tables/academic_irf_interpretation_table.csv`
- `outputs/tables/academic_granger_causality_map.csv`
- `outputs/tables/academic_fevd_selected_horizons.csv`
- `outputs/tables/academic_varx_scenario_response.csv`
- `outputs/figures/academic_10_structural_irf.png`
- `outputs/figures/academic_11_fevd.png`
- `outputs/figures/academic_varx_fedfunds_scenario_response.png`
- `outputs/figures/academic_var_residual_acf_ccf_matrix.png`
- `outputs/figures/academic_var_residual_qq_hist.png`
- `outputs/figures/academic_varx_residual_qq_hist.png`
- `outputs/figures/academic_multihorizon_rmse.png`
- `outputs/figures/academic_irf_with_confidence_intervals.png`
- `outputs/figures/academic_14_ml_forecast_comparison.png`

## Suggested Report Structure

1. Introduction and research question.
2. Data description and source.
3. Preprocessing and transformations.
4. Exploratory time-series analysis.
5. Stationarity and Granger causality results.
6. Forecasting model and evaluation.
7. Conclusion and limitations.

## Note About API Keys

Do not hard-code your real FRED API key in a submitted project or public repository. This project reads it from the `FRED_API_KEY` environment variable.
