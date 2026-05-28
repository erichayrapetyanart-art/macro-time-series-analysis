# Project Context: Macroeconomic Time-Series Modeling and Forecasting

## Current Official Baseline Update

This note supersedes earlier level-FEDFUNDS baseline references in older optimization tables in this file. Those older tables are retained as historical comparison output, but the official baseline after the FEDFUNDS-vs-D_FEDFUNDS experiment is:

- VAR baseline: `VAR_D_FEDFUNDS_candidate`, endogenous variables `INF, D_FEDFUNDS, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE`, lag 4, policy representation `D_FEDFUNDS`.
- VARX baseline: `VARX_level_FEDFUNDS_baseline`, endogenous variables `INF, UNRATE, INDPRO_GROWTH, M2_GROWTH`, exogenous variables `FEDFUNDS, SENTIMENT_CHANGE`, lag 4, policy representation `FEDFUNDS`.

Interpretation rule: `FEDFUNDS` is the policy-rate level / policy stance; `D_FEDFUNDS` is the monthly policy-rate change / tightening-easing movement. They should not both be treated as the official policy variable in the same baseline model.

## Project Goal

Build a complete academic econometrics/data-science project using U.S. monthly macroeconomic data from FRED. The project studies inflation dynamics using VAR and VARX econometric models, with machine-learning methods used only as forecast-comparison benchmarks.

The project has two main artifacts:

- `macro_time_series_project.ipynb`: academic notebook with methodology, theory, results, and interpretation.
- `dashboard_app.py`: interactive Streamlit dashboard for presentation and visual exploration.

## Data

Data source: FRED API.

Raw variables:

| Variable | FRED ID | Meaning |
| --- | --- | --- |
| CPI | `CPIAUCSL` | Consumer Price Index |
| UNRATE | `UNRATE` | Unemployment Rate |
| FEDFUNDS | `FEDFUNDS` | Federal Funds Rate |
| INDPRO | `INDPRO` | Industrial Production |
| M2 | `M2SL` | Money Supply |
| UMCSENT | `UMCSENT` | Consumer Sentiment |

Transformed model variables:

| Model Variable | Construction |
| --- | --- |
| `INF` | monthly log CPI growth times 100 |
| `FEDFUNDS` | federal funds rate level |
| `UNRATE` | unemployment rate level |
| `INDPRO_GROWTH` | monthly log industrial production growth times 100 |
| `M2_GROWTH` | monthly log M2 growth times 100 |
| `SENTIMENT_CHANGE` | first difference of consumer sentiment |

Break dummies:

- `D_2008`: equals 1 from September 2008 onward.
- `D_COVID`: equals 1 from March 2020 onward.

## Final Model Architecture

### Primary Model: VAR(4)

The baseline reduced-form system is a VAR with 4 monthly lags.

Endogenous variables:

`FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, M2_GROWTH, SENTIMENT_CHANGE`

Exogenous controls:

`D_2008, D_COVID`

Lag order:

- AIC selects 4 lags.
- BIC/HQIC and residual autocorrelation robustness are also reported.

Main use:

- dynamic macroeconomic system modeling,
- Granger causality,
- residual diagnostics,
- IRF,
- FEVD.

### Conditional Forecasting Model: VARX(4)

Endogenous variables:

`INF, UNRATE, INDPRO_GROWTH, M2_GROWTH`

Exogenous variables:

`FEDFUNDS, SENTIMENT_CHANGE, D_2008, D_COVID`

Economic logic:

- `FEDFUNDS` can be treated as a policy/scenario path.
- `SENTIMENT_CHANGE` can be treated as an expectations/conditions path.
- crisis dummies are structural break controls.

Main use:

- conditional forecasting,
- scenario forecasting with externally supplied policy/sentiment paths,
- conditional shock/scenario response analysis for exogenous paths,
- train/test forecast comparison,
- rolling 3-month forecast evaluation.

### Structural Layer: Recursive SVAR / Cholesky

IRF and FEVD are based on Cholesky decomposition of the VAR residual covariance matrix.

Ordering:

`FEDFUNDS -> INF -> UNRATE -> INDPRO_GROWTH -> M2_GROWTH -> SENTIMENT_CHANGE`

Interpretation:

- variables ordered earlier can contemporaneously affect later variables;
- variables ordered later cannot contemporaneously affect earlier variables;
- this is an imposed identification assumption, not directly estimated.

Motivation:

- `FEDFUNDS` is fast-moving policy/financial variable;
- `INF` is the central policy target;
- labor and real activity adjust more slowly;
- money and sentiment are placed later as channels that absorb macro news;
- Granger results support the central predictive role of `FEDFUNDS`.

Limitation:

Changing the ordering may change short-run IRFs.

## Machine-Learning Benchmarks

ML models use lagged macroeconomic features to forecast inflation:

- Ridge Regression
- Random Forest
- Gradient Boosting
- Random Walk benchmark

Current result:

Ridge Regression has the best one-step RMSE, but econometric models remain more useful for economic interpretation because they support IRF, FEVD, Granger causality, and structural discussion.

## Important Results

- Raw CPI, M2, and Consumer Sentiment are non-stationary in levels.
- Transformed model variables are stationary at the 5% level.
- Pairwise Engle-Granger tests do not support cointegration among non-stationary level variables.
- VAR(4) is stable.
- Some residual autocorrelation remains visually, so the model is not perfectly white.
- Multivariate whiteness tests reject for the baseline systems; this is reported as a limitation.
- Residual normality is rejected in several equations, consistent with fat-tailed crisis shocks around 2008 and COVID.
- Lag robustness shows higher lag orders reduce residual ACF exceedances but add parameters.
- ARCH and heteroskedasticity effects appear in selected equations, especially inflation and M2 growth.
- HC3 and HAC/Newey-West robust inference is now included as a coefficient-significance sensitivity check.
- Model complexity tables warn that full VAR parameter count is high relative to sample size.
- `FEDFUNDS` Granger-causes inflation at the 5% level.
- Rolling VARX 3-month forecasts beat random walk for inflation.
- Multi-horizon forecast tables compare VAR, VARX, Ridge Regression, and Random Walk at 1, 3, 6, and 12 months.
- Crisis dummy, expanding-window, regime-split, and alternative Cholesky-ordering robustness outputs are generated.
- VARX conditional scenario responses are generated for a temporary `FEDFUNDS` path shock. These are explicitly not structural IRFs.

## Main Files

| File | Purpose |
| --- | --- |
| `macro_time_series_project.ipynb` | main academic notebook |
| `dashboard_app.py` | interactive Streamlit dashboard |
| `src/macro_time_series_analysis.py` | base FRED download and baseline analysis |
| `src/advanced_macro_var_analysis.py` | full academic econometric and ML pipeline |
| `src/data.py` / `src/models_var.py` / `src/models_varx.py` / `src/diagnostics.py` / `src/forecasting.py` / `src/visualization.py` / `src/dashboard_helpers.py` | modular dashboard support code |
| `project_report.md` | compact report summary |
| `README.md` | setup and run instructions |
| `requirements.txt` | Python dependencies |

## Generated Data

| File | Purpose |
| --- | --- |
| `data/raw_fred_macro.csv` | raw FRED data |
| `data/processed_macro.csv` | baseline processed data |
| `data/academic_model_data.csv` | transformed model data |
| `data/academic_break_dummies.csv` | crisis dummy variables |

## Key Output Tables

Important tables are in `outputs/tables/`.

Examples:

- `academic_final_model_architecture.csv`
- `academic_cholesky_ordering.csv`
- `academic_all_model_forecast_ranking.csv`
- `academic_var_lag_selection.csv`
- `academic_varx_lag_selection.csv`
- `academic_var_residual_diagnostics.csv`
- `academic_varx_residual_diagnostics.csv`
- `academic_var_residual_acf_summary.csv`
- `academic_var_residual_ccf_summary.csv`
- `academic_var_arch_tests.csv`
- `academic_varx_arch_tests.csv`
- `academic_var_residual_normality.csv`
- `academic_varx_residual_normality.csv`
- `academic_var_heteroskedasticity_tests.csv`
- `academic_varx_heteroskedasticity_tests.csv`
- `academic_var_parameter_significance.csv`
- `academic_varx_parameter_significance.csv`
- `academic_var_parameter_significance_robust.csv`
- `academic_varx_parameter_significance_robust.csv`
- `academic_model_complexity_overparameterization.csv`
- `academic_var_varx_diagnostic_comparison.csv`
- `academic_multihorizon_forecast_comparison.csv`
- `academic_diebold_mariano_tests.csv`
- `academic_crisis_dummy_robustness.csv`
- `academic_expanding_window_robustness.csv`
- `academic_regime_split_comparison.csv`
- `academic_irf_robustness_summary.csv`
- `academic_alternative_cholesky_orderings.csv`
- `academic_irf_paths.csv`
- `academic_irf_interpretation_table.csv`
- `academic_varx_scenario_response.csv`
- `academic_fevd_full.csv`
- `academic_fevd_dominant_shocks.csv`
- `academic_granger_causality_map.csv`

## Key Output Figures

Important figures are in `outputs/figures/`.

Examples:

- `academic_01_raw_series.png`
- `academic_02_transformed_series.png`
- `academic_04_correlation_matrix.png`
- `academic_09_granger_causality_heatmap.png`
- `academic_10_structural_irf.png`
- `academic_11_fevd.png`
- `academic_12_econometric_forecast_comparison.png`
- `academic_14_ml_forecast_comparison.png`
- `academic_var_residual_acf.png`
- `academic_var_residual_acf_ccf_matrix.png`
- `academic_var_residual_qq_hist.png`
- `academic_varx_residual_qq_hist.png`
- `academic_varx_residual_acf_ccf_matrix.png`
- `academic_multihorizon_rmse.png`
- `academic_irf_with_confidence_intervals.png`
- `academic_varx_fedfunds_scenario_response.png`

## How To Run

Activate the environment:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the base FRED download and baseline outputs:

```bash
export FRED_API_KEY="your_key_here"
python src/macro_time_series_analysis.py
```

Run the full academic pipeline:

```bash
python src/advanced_macro_var_analysis.py
```

Run the dashboard:

```bash
streamlit run dashboard_app.py
```

Dashboard local URL:

```text
http://127.0.0.1:8501/
```

## Current Dashboard Status

The dashboard is intended as the presentation/demo layer. Sidebar page order:

1. Overview
2. Stationarity and Data Preparation
3. Model Architecture and Direct Results
4. Forecast Comparison
5. Residual Diagnostics
6. Significance Analysis and Granger Causality
7. IRF and FEVD
8. Robustness
9. Code quality

It includes:

- official baseline model architecture,
- an interactive VAR/VARX model lab,
- train/test split selection,
- endogenous and exogenous variable selection with validation,
- real-time AIC/BIC/HQIC lag selection,
- manual lag-order choice,
- forecasts and ML comparison,
- residual time-series, ACF, CCF, Ljung-Box, and ARCH diagnostics,
- residual normality histograms, Q-Q plots, Jarque-Bera tests, skewness, and kurtosis,
- stability roots and overparameterization warnings,
- baseline robustness tables for robust significance, crisis dummies, multi-horizon forecasts, Diebold-Mariano tests, expanding windows, and regimes,
- VAR IRF/FEVD with Cholesky-ordering warnings and alternative ordering input,
- VARX conditional scenario responses with explicit warnings that exogenous paths are assumed,
- data exploration, correlations, and stationarity tables.

## Notes for Future Improvement

Strong possible extensions:

- add exchange rates, fiscal variables, financial conditions, labor-market detail, or inflation expectations;
- test alternative Cholesky orderings;
- estimate Bayesian VAR or restricted VAR;
- expand bootstrap IRF confidence intervals beyond the current baseline FEDFUNDS shock figure;
- consider time-varying parameter VAR or stochastic-volatility VAR;
- compare expanding-window vs rolling-window forecasts;
- add real-time data vintage discussion if the course requires forecasting realism.

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
