# Macroeconomic Time-Series Modeling and Forecasting

## Current Official Baseline Update

This note supersedes earlier level-FEDFUNDS baseline references in older optimization tables in this file. Those older tables are retained as historical comparison output, but the official baseline after the FEDFUNDS-vs-D_FEDFUNDS experiment is:

- VAR baseline: `VAR_D_FEDFUNDS_candidate`, endogenous variables `INF, D_FEDFUNDS, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE`, lag 4, policy representation `D_FEDFUNDS`.
- VARX baseline: `VARX_level_FEDFUNDS_baseline`, endogenous variables `INF, UNRATE, INDPRO_GROWTH, M2_GROWTH`, exogenous variables `FEDFUNDS, SENTIMENT_CHANGE`, lag 4, policy representation `FEDFUNDS`.

Interpretation rule: `FEDFUNDS` is the policy-rate level / policy stance; `D_FEDFUNDS` is the monthly policy-rate change / tightening-easing movement. They should not both be treated as the official policy variable in the same baseline model.

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
- Residual normality diagnostics using Jarque-Bera, skewness, kurtosis, histograms, Q-Q plots, and VAR system normality tests.
- Parameter significance analysis using coefficients, classical standard errors, HC3 robust p-values, HAC/Newey-West p-values, and confidence intervals.
- Heteroskedasticity and ARCH tests for VAR and VARX residuals.
- Model complexity and overparameterization warnings.
- Crisis dummy, expanding-window, regime-split, and Cholesky-ordering robustness checks.
- Multi-horizon forecast comparison and approximate Diebold-Mariano tests.
- Granger causality tests.
- Impulse Response Functions.
- Forecast Error Variance Decomposition.
- VARX conditional scenario-response analysis for externally supplied policy paths.
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
- VARX scenario responses.

The dashboard is not only a static presentation layer. It includes an interactive model lab where users can choose the train/test split date, switch between VAR and VARX, select endogenous and exogenous variables, run real-time AIC/BIC/HQIC lag selection, manually choose lag order, refit the model, inspect stability and residual diagnostics, compare forecasts with ML benchmarks, and evaluate VAR IRF/FEVD results under alternative Cholesky orderings.

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

The notebook is intended for academic methodology and written explanation. The dashboard is intended for presentation and interactive interpretation.

## Key Results

- CPI, M2, and consumer sentiment are non-stationary in levels.
- Pairwise cointegration tests do not find cointegration among the non-stationary level variables at the 5% level.
- All transformed model variables are stationary at the 5% level.
- AIC selects a VAR lag order of 4.
- The VAR system is stable.
- Visual residual diagnostics show that several residual ACF values still exceed approximate 95% confidence bounds, and multivariate whiteness is rejected, so the model is not perfectly white despite being stable.
- Residual normality is rejected in several equations. This is plausible for monthly macro data with 2008 and COVID tail events; it mainly affects inference and confidence bands rather than automatically invalidating forecasts.
- Lag robustness indicates that higher lag orders reduce residual autocorrelation exceedances, but at the cost of additional parameters.
- ARCH and heteroskedasticity tests detect variance instability in selected equations, so robust inference and bootstrap/Monte Carlo confidence intervals should be preferred for sensitivity analysis.
- Robust HC3/HAC checks show that some individual VAR coefficient significance conclusions are sensitive to covariance assumptions.
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

VARX scenario responses are reported separately from VAR/SVAR IRFs. They show how endogenous VARX variables respond when an exogenous path, such as FEDFUNDS, is temporarily shocked. This is conditional scenario analysis, not structural identification.

FEVD results show how forecast uncertainty is allocated across structural shocks. Inflation forecast variance is mostly explained by its own shock in the reported horizons, while other variables show different dominant shock sources.

If a positive federal funds rate shock is associated with a short-run inflation increase, the result should be interpreted as a possible price puzzle or endogenous policy reaction, not as mechanical causal evidence. Medium-run responses and robustness across plausible orderings are more credible than one impact response.

## Limitations

- Granger causality is predictive, not structural causality.
- Cholesky IRFs depend on the chosen variable ordering.
- Residual non-normality and heteroskedasticity make classical p-values and confidence intervals less reliable.
- Residual autocorrelation remains in some equations, suggesting possible improvements through lag-order changes, seasonal effects, additional exogenous variables, or restricted/Bayesian VARs.
- ARCH effects suggest that robust standard errors, bootstrap IRF bands, GARCH-type volatility modeling, or stochastic-volatility VARs could improve inference.
- VAR parameters may be unstable across major regimes.
- VARX conditional forecasts use future exogenous values; fully real-time forecasting would require forecasting those variables too.
- Important omitted variables include exchange rates, fiscal policy, financial conditions, labor-market detail, and inflation expectations.

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
