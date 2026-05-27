# Project Context: Macroeconomic Time-Series Modeling and Forecasting

This file summarizes the current state, structure, methodology, and runnable commands for the project.

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
- Lag robustness shows higher lag orders reduce residual ACF exceedances but add parameters.
- ARCH effects appear in selected equations, especially inflation and M2 growth.
- `FEDFUNDS` Granger-causes inflation at the 5% level.
- Rolling VARX 3-month forecasts beat random walk for inflation.

## Main Files

| File | Purpose |
| --- | --- |
| `macro_time_series_project.ipynb` | main academic notebook |
| `dashboard_app.py` | interactive Streamlit dashboard |
| `src/macro_time_series_analysis.py` | base FRED download and baseline analysis |
| `src/advanced_macro_var_analysis.py` | full academic econometric and ML pipeline |
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
- `academic_var_parameter_significance.csv`
- `academic_varx_parameter_significance.csv`
- `academic_irf_paths.csv`
- `academic_irf_interpretation_table.csv`
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
- `academic_varx_residual_acf_ccf_matrix.png`

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

The dashboard is intended as the presentation/demo layer. It includes tabs for:

- official baseline model architecture,
- an interactive VAR/VARX model lab,
- train/test split selection,
- endogenous and exogenous variable selection with validation,
- real-time AIC/BIC/HQIC lag selection,
- manual lag-order choice,
- forecasts and ML comparison,
- residual time-series, ACF, CCF, Ljung-Box, and ARCH diagnostics,
- stability roots and overparameterization warnings,
- VAR IRF/FEVD with Cholesky-ordering warnings and alternative ordering input,
- data exploration, correlations, and stationarity tables.

## Notes for Future Improvement

Strong possible extensions:

- add oil prices, exchange rates, financial conditions, or inflation expectations;
- test alternative Cholesky orderings;
- estimate Bayesian VAR or restricted VAR;
- add bootstrap IRF confidence intervals;
- consider time-varying parameter VAR or stochastic-volatility VAR;
- add Diebold-Mariano forecast comparison tests;
- compare expanding-window vs rolling-window forecasts;
- add real-time data vintage discussion if the course requires forecasting realism.
