# U.S. Macroeconomic Time Series Analysis

## Current Official Baseline Update

This note supersedes earlier level-FEDFUNDS baseline references in older optimization tables in this file. Those older tables are retained as historical comparison output, but the official baseline after the FEDFUNDS-vs-D_FEDFUNDS experiment is:

- VAR baseline: `VAR_D_FEDFUNDS_candidate`, endogenous variables `INF, D_FEDFUNDS, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE`, lag 4, policy representation `D_FEDFUNDS`.
- VARX baseline: `VARX_level_FEDFUNDS_baseline`, endogenous variables `INF, UNRATE, INDPRO_GROWTH, M2_GROWTH`, exogenous variables `FEDFUNDS, SENTIMENT_CHANGE`, lag 4, policy representation `FEDFUNDS`.

Interpretation rule: `FEDFUNDS` is the policy-rate level / policy stance; `D_FEDFUNDS` is the monthly policy-rate change / tightening-easing movement. They should not both be treated as the official policy variable in the same baseline model.

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

## FEDFUNDS vs D_FEDFUNDS Baseline Decision

This branch compares level `FEDFUNDS` against `D_FEDFUNDS = FEDFUNDS.diff()` as the policy variable. Level `FEDFUNDS` measures policy stance; `D_FEDFUNDS` measures monthly tightening/easing movements. The dashboard keeps both variables available, but the official baseline uses only one policy representation within each model.

### Stationarity evidence

- Level `FEDFUNDS`: ADF p-value = 0.02881, KPSS p-value = 0.01, ACF(1) = 0.9936.
- `D_FEDFUNDS`: ADF p-value = 0.00147, KPSS p-value = 0.1, ACF(1) = 0.6551.
- Conclusion: `D_FEDFUNDS` is more stationary and much less persistent than level `FEDFUNDS`.

### VAR baseline decision

- Level-FEDFUNDS baseline: lag 5, inflation RMSE = 0.1760, mean RMSE = 1.1291, min Ljung-Box p = 0.1852, Portmanteau p = 1.085e-05, ACF exceedance share = 0.050, stable = yes.
- D_FEDFUNDS candidate: lag 4, inflation RMSE = 0.1770, mean RMSE = 1.0780, min Ljung-Box p = 0.2636, Portmanteau p = 1.217e-06, ACF exceedance share = 0.067, stable = yes.
- Official VAR baseline: `D_FEDFUNDS`, lag 4. Switch official VAR baseline to D_FEDFUNDS because stationarity validity improves and forecast loss is negligible.

### VARX baseline decision

- Level-FEDFUNDS VARX baseline: lag 4, inflation RMSE = 0.1920, mean RMSE = 0.5087, min Ljung-Box p = 0.0614, ACF exceedance share = 0.062, stable = yes.
- D_FEDFUNDS VARX candidate: lag 5, inflation RMSE = 0.2284, mean RMSE = 0.7753, min Ljung-Box p = 0.3829, ACF exceedance share = 0.052, stable = yes.
- Official VARX baseline: `FEDFUNDS`, lag 4. Keep official VARX baseline with level FEDFUNDS for conditional policy-stance scenario forecasting; report D_FEDFUNDS VARX as robustness.

### Granger, IRF, FEVD, and scenario conclusions

- Selected D_FEDFUNDS VAR Granger relationships: D_FEDFUNDS->INF (p=0.0248), UNRATE->D_FEDFUNDS (p=0.0213), D_FEDFUNDS->UNRATE (p=3.28e-10), INDPRO_GROWTH->UNRATE (p=9.61e-06), INF->INDPRO_GROWTH (p=0.0148), D_FEDFUNDS->INDPRO_GROWTH (p=0.000152), UNRATE->INDPRO_GROWTH (p=4.42e-07), INF->SENTIMENT_CHANGE (p=0.0135).
- Response of inflation to a D_FEDFUNDS shock: h1=0.0428, h6=0.0023, h12=0.0003, h24=0.0013. This is a response to an unexpected policy-rate change, not a policy-rate-level stance shock.
- D_FEDFUNDS contribution to INF FEVD: h12=0.040, h24=0.040.
- VARX conditional inflation scenario response to a D_FEDFUNDS exogenous shock: h1=0.3918, h6=-0.0227, h12=-0.0111, h24=-0.0015. VARX responses are conditional scenario responses, not structural IRFs.

### Final answer

`D_FEDFUNDS` is more stationary. The official VAR baseline switches to `D_FEDFUNDS` because the inflation forecast loss is negligible and stationarity validity improves. The official VARX baseline keeps level `FEDFUNDS` because conditional forecast performance is materially better with the policy-rate level. The alternative representation remains available as a sensitivity-analysis variable in the dashboard.
