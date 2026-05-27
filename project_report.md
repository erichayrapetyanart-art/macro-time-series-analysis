# Macroeconomic Time-Series Modeling and Forecasting

This report summarizes the final academic project implemented in `macro_time_series_project.ipynb`.

## Objective

The project studies whether U.S. inflation can be explained and forecast using a monthly macroeconomic system containing the federal funds rate, unemployment, industrial production, M2 money supply, and consumer sentiment.

The analysis compares interpretable econometric models against machine-learning forecasting methods.

## Data

All data are collected from FRED:

| Variable | FRED ID | Interpretation |
| --- | --- | --- |
| CPI | `CPIAUCSL` | Price level used to construct inflation |
| FEDFUNDS | `FEDFUNDS` | Monetary-policy stance |
| UNRATE | `UNRATE` | Labor market slack |
| INDPRO | `INDPRO` | Real activity |
| M2 | `M2SL` | Broad money supply |
| UMCSENT | `UMCSENT` | Consumer expectations and sentiment |

The raw dataset contains 376 monthly observations and 6 variables. The transformed model dataset contains 375 observations.

## Main Transformations

- Inflation: monthly log CPI growth multiplied by 100.
- Industrial production growth: monthly log growth multiplied by 100.
- M2 growth: monthly log growth multiplied by 100.
- Sentiment change: first difference of consumer sentiment.
- Federal funds rate and unemployment rate are kept in levels.
- Step dummies are added for the 2008 financial crisis and COVID period.

## Econometric Workflow

The notebook includes:

- Raw and transformed ADF stationarity tests.
- Pairwise Engle-Granger cointegration checks.
- Summary statistics, correlation analysis, scatter plots, distributions, ACF/PACF, and structural break inspection.
- VAR lag selection using AIC, BIC, HQIC, and FPE.
- VAR estimation with crisis dummies.
- VARX estimation with policy/sentiment conditioning variables and explicit conditional forecast interpretation.
- Residual diagnostics: Durbin-Watson, Ljung-Box, residual correlation, and stability roots.
- Visual residual autocorrelation diagnostics: residual ACF plots and residual ACF/CCF matrices with confidence bounds.
- Parameter significance analysis using coefficients, standard errors, t-statistics, p-values, and confidence intervals.
- Heteroskedasticity and ARCH tests for VAR and VARX residuals.
- Granger causality tests.
- Impulse Response Functions.
- Forecast Error Variance Decomposition.
- Forecast evaluation using RMSE and MAE.

## Final Model Architecture

The project uses a layered final architecture:

| Component | Model | Variables |
| --- | --- | --- |
| Primary system model | VAR(4) | Endogenous: FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, M2_GROWTH, SENTIMENT_CHANGE; exogenous: D_2008, D_COVID |
| Conditional forecast model | VARX(4) | Endogenous: INF, UNRATE, INDPRO_GROWTH, M2_GROWTH; exogenous: FEDFUNDS, SENTIMENT_CHANGE, D_2008, D_COVID |
| Structural interpretation | Recursive SVAR | Cholesky decomposition applied to the reduced-form VAR residual covariance matrix |

The VAR lag order is 4 because AIC selects 4 monthly lags among the candidate lag orders. BIC/HQIC and residual autocorrelation robustness are reported as checks.

The VARX lag-order table is also reported. The official VARX keeps 4 lags for comparability with the baseline VAR, even though alternative information criteria may favor other lag lengths. This is a deliberate presentation choice: the notebook preserves one fixed baseline model, while the dashboard allows sensitivity analysis over train/test dates, variable sets, and lag orders.

The Cholesky ordering is:

`FEDFUNDS -> INF -> UNRATE -> INDPRO_GROWTH -> M2_GROWTH -> SENTIMENT_CHANGE`

This ordering is motivated by policy transmission logic, speed of adjustment, and Granger evidence. FEDFUNDS is treated as a fast-moving policy/financial variable. Inflation is placed early because price dynamics are the central policy target. Labor and real activity adjust more slowly. Money growth and sentiment are placed later so they can contemporaneously absorb macroeconomic news.

The ordering imposes contemporaneous restrictions, so IRFs are conditional on those assumptions. Alternative orderings may change short-run impulse responses.

## Dashboard

The project also includes `dashboard_app.py`, a Streamlit dashboard that interactively presents:

- time-series plots,
- correlations,
- stationarity tests,
- Granger causality,
- forecasts,
- IRFs,
- FEVD,
- residual diagnostics,
- stability and significance tables,
- model comparison metrics.

The dashboard is not only a static presentation layer. It includes an interactive model lab where users can choose the train/test split date, switch between VAR and VARX, select endogenous and exogenous variables, run real-time AIC/BIC/HQIC lag selection, manually choose lag order, refit the model, inspect stability and residual diagnostics, compare forecasts with ML benchmarks, and evaluate VAR IRF/FEVD results under alternative Cholesky orderings.

The notebook is intended for academic methodology and written explanation. The dashboard is intended for presentation and interactive interpretation.

## Key Results

- CPI, M2, and consumer sentiment are non-stationary in levels.
- Pairwise cointegration tests do not find cointegration among the non-stationary level variables at the 5% level.
- All transformed model variables are stationary at the 5% level.
- AIC selects a VAR lag order of 4.
- The VAR system is stable.
- Visual residual diagnostics show that several residual ACF values still exceed approximate 95% confidence bounds, so the model is not perfectly white despite being stable.
- Lag robustness indicates that higher lag orders reduce residual autocorrelation exceedances, but at the cost of additional parameters.
- ARCH tests detect heteroskedasticity in selected equations, especially inflation and M2 growth, so inference and confidence intervals should be interpreted with caution.
- The federal funds rate has significant Granger-predictive content for inflation.
- Rolling 3-month VARX forecasts beat random walk for inflation.

## Forecast Ranking

Final inflation forecast ranking by RMSE:

| Model | Design | RMSE | MAE |
| --- | --- | ---: | ---: |
| Ridge Regression | One-step lagged features | 0.159 | 0.117 |
| Random Walk | One-step | 0.174 | 0.132 |
| Random Forest | One-step lagged features | 0.179 | 0.130 |
| Gradient Boosting | One-step lagged features | 0.184 | 0.139 |
| Random Walk | Direct 36-month | 0.188 | 0.151 |
| VAR | Final 36 months | 0.199 | 0.170 |
| VARX | Final 36 months | 0.218 | 0.183 |

## Interpretation

Machine-learning methods, especially regularized linear regression, perform well for pure short-horizon prediction. Econometric models remain stronger for interpretation because they support system-level analysis, Granger causality, impulse responses, forecast error variance decomposition, and economic discussion of shocks.

The IRF section is especially important for policy interpretation. It shows how monetary policy, inflation, unemployment, output, money, and sentiment shocks propagate over time. These results are conditional on the Cholesky ordering, so they should be interpreted as recursively identified scenario evidence rather than definitive structural causality.

FEVD results show how forecast uncertainty is allocated across structural shocks. Inflation forecast variance is mostly explained by its own shock in the reported horizons, while other variables show different dominant shock sources.

## Limitations

- Granger causality is predictive, not structural causality.
- Cholesky IRFs depend on the chosen variable ordering.
- Residual autocorrelation remains in some equations, suggesting possible improvements through lag-order changes, seasonal effects, additional exogenous variables, or restricted/Bayesian VARs.
- ARCH effects suggest that robust standard errors, bootstrap IRF bands, GARCH-type volatility modeling, or stochastic-volatility VARs could improve inference.
- VAR parameters may be unstable across major regimes.
- VARX conditional forecasts use future exogenous values; fully real-time forecasting would require forecasting those variables too.
- Important omitted variables include oil prices, exchange rates, fiscal policy, financial conditions, and inflation expectations.
