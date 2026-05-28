# Project Results Context: Optimized VAR / VARX Macroeconomic Time-Series Project

This file is designed to be pasted into ChatGPT for external discussion. It contains the optimized model-selection results, key diagnostics, and main economic findings without requiring access to the full repository.

## 1. Project Overview

- Research motivation: model U.S. inflation and macroeconomic policy dynamics using interpretable time-series econometrics, while comparing forecast performance against ML benchmarks.
- Main economic question: how inflation interacts with monetary policy, unemployment, industrial production, money growth, and sentiment, and whether these relationships are useful for forecasting and policy interpretation.
- Dataset: monthly FRED macroeconomic data.
- Raw sample: 1995-01-01 to 2026-04-01; transformed model sample: 1995-02-01 to 2026-04-01.
- Train/test split used in optimization: train through 2023-04-01; test from 2023-05-01 to 2026-04-01 (36 months).
- Final modeling goal: select a defensible VAR for dynamic policy analysis, Granger causality, IRF, and FEVD; select a defensible VARX for conditional/scenario forecasting; keep ML only as forecast benchmarks. VAR is the main policy-interpretation model; VARX is mainly the conditional/scenario model; Ridge-style ML can be strongest for pure one-step prediction but does not replace econometric interpretation.

Variables and transformations:

| Variable | Role | Transformation |
| --- | --- | --- |
| INF | inflation target / price dynamics | `log(CPI).diff() * 100` |
| FEDFUNDS | monetary policy stance | level |
| UNRATE | labor-market slack | level |
| INDPRO_GROWTH | real activity | `log(INDPRO).diff() * 100` |
| M2_GROWTH | money growth channel | `log(M2).diff() * 100` |
| SENTIMENT_CHANGE | expectations / sentiment | `UMCSENT.diff()` |
| D_2008 | crisis control | dummy |
| D_COVID | pandemic control | dummy |

## 2. Stationarity and Data Preparation

Raw ADF results:

| variable | adf_statistic | p_value | used_lag | stationary_at_5pct |
| -------- | ------------- | ------- | -------- | ------------------ |
| CPI      | 1.8927        | 0.9985  | 15       | False              |
| UNRATE   | -3.1612       | 0.0223  | 0        | True               |
| FEDFUNDS | -2.8714       | 0.0488  | 9        | True               |
| INDPRO   | -2.8679       | 0.0492  | 2        | True               |
| M2       | 0.9286        | 0.9935  | 13       | False              |
| UMCSENT  | -1.1492       | 0.6951  | 6        | False              |

Raw KPSS results:

| variable | kpss_statistic | p_value | used_lag | stationary_at_5pct |
| -------- | -------------- | ------- | -------- | ------------------ |
| CPI      | 3.0778         | 0.01    | 11       | False              |
| UNRATE   | 0.3676         | 0.0911  | 11       | True               |
| FEDFUNDS | 0.9888         | 0.01    | 11       | False              |
| INDPRO   | 2.1002         | 0.01    | 11       | False              |
| M2       | 3.0184         | 0.01    | 11       | False              |
| UMCSENT  | 1.1896         | 0.01    | 11       | False              |

Integration-order classification:

| variable | integration_order | level_adf_p_value | level_kpss_p_value | diff_adf_p_value | diff_kpss_p_value |
| -------- | ----------------- | ----------------- | ------------------ | ---------------- | ----------------- |
| CPI      | mixed/problematic | 0.9985            | 0.01               | 0.0685           | 0.01              |
| UNRATE   | I(0)              | 0.0223            | 0.0911             | 0.0              | 0.1               |
| FEDFUNDS | I(1)              | 0.0488            | 0.01               | 0.0015           | 0.1               |
| INDPRO   | I(1)              | 0.0492            | 0.01               | 0.0              | 0.1               |
| M2       | mixed/problematic | 0.9935            | 0.01               | 0.0076           | 0.0196            |
| UMCSENT  | I(1)              | 0.6951            | 0.01               | 0.0              | 0.1               |

Cointegration evidence among non-stationary level variables:

| test                       | Acceptable if                                               | left_variable | right_variable | residual_adf_p_value | cointegrated_at_5pct |
| -------------------------- | ----------------------------------------------------------- | ------------- | -------------- | -------------------- | -------------------- |
| Engle-Granger residual ADF | residual ADF p-value < 0.05 supports pairwise cointegration | log_CPI       | log_M2         | 0.2065               | False                |
| Engle-Granger residual ADF | residual ADF p-value < 0.05 supports pairwise cointegration | log_CPI       | UMCSENT        | 0.365                | False                |
| Engle-Granger residual ADF | residual ADF p-value < 0.05 supports pairwise cointegration | log_M2        | UMCSENT        | 0.4267               | False                |

Final transformed-variable stationarity tests:

| variable         | adf_statistic | p_value | used_lag | stationary_at_5pct |
| ---------------- | ------------- | ------- | -------- | ------------------ |
| FEDFUNDS         | -3.0706       | 0.0288  | 8        | True               |
| INF              | -3.5731       | 0.0063  | 14       | True               |
| UNRATE           | -3.1566       | 0.0226  | 0        | True               |
| INDPRO_GROWTH    | -14.407       | 0.0     | 1        | True               |
| M2_GROWTH        | -5.6467       | 0.0     | 4        | True               |
| SENTIMENT_CHANGE | -11.6379      | 0.0     | 4        | True               |

| variable         | kpss_statistic | p_value | used_lag | stationary_at_5pct |
| ---------------- | -------------- | ------- | -------- | ------------------ |
| FEDFUNDS         | 0.9776         | 0.01    | 11       | False              |
| INF              | 0.289          | 0.1     | 6        | True               |
| UNRATE           | 0.3686         | 0.0907  | 11       | True               |
| INDPRO_GROWTH    | 0.2814         | 0.1     | 3        | True               |
| M2_GROWTH        | 0.0702         | 0.1     | 9        | True               |
| SENTIMENT_CHANGE | 0.0874         | 0.1     | 14       | True               |

ACF/PACF summary for final transformed variables:

| variable         | acf_lag1 | pacf_lag1 | acf_exceedances_lag1_to_12 | pacf_exceedances_lag1_to_12 | interpretation                         |
| ---------------- | -------- | --------- | -------------------------- | --------------------------- | -------------------------------------- |
| INF              | 0.4635   | 0.4635    | 4                          | 5                           | limited/moderate short-run persistence |
| UNRATE           | 0.9461   | 0.9461    | 12                         | 1                           | AR(1)-like persistence                 |
| FEDFUNDS         | 0.9936   | 0.9936    | 12                         | 4                           | AR(1)-like persistence                 |
| INDPRO_GROWTH    | 0.1911   | 0.1911    | 3                          | 3                           | limited/moderate short-run persistence |
| M2_GROWTH        | 0.6043   | 0.6043    | 12                         | 2                           | AR(1)-like persistence                 |
| SENTIMENT_CHANGE | -0.0028  | -0.0028   | 2                          | 3                           | limited/moderate short-run persistence |

Interpretation: transformed inflation, growth, and sentiment-change variables are designed to be stationary. UNRATE and FEDFUNDS are retained in levels because they are policy-relevant macro rates and pass or nearly pass stationarity diagnostics in this sample, but they remain persistent and require careful residual diagnostics. UNRATE especially should be treated as persistent/AR-like when its lag-1 ACF is high.

## 3. Model Optimization Summary

Candidate VAR systems tested:
- VAR_core4: INF, FEDFUNDS, UNRATE, INDPRO_GROWTH
- VAR_core_plus_M2: core + M2_GROWTH
- VAR_core_plus_sentiment: core + SENTIMENT_CHANGE
- VAR_full6: all six transformed variables

Candidate VARX systems tested:
- VARX_A_policy_sentiment_exog: endogenous INF, UNRATE, INDPRO_GROWTH, M2_GROWTH; exogenous FEDFUNDS, SENTIMENT_CHANGE
- VARX_B_policy_money_sentiment_exog: endogenous INF, UNRATE, INDPRO_GROWTH; exogenous FEDFUNDS, M2_GROWTH, SENTIMENT_CHANGE
- VARX_C_policy_endogenous: endogenous INF, FEDFUNDS, UNRATE, INDPRO_GROWTH; exogenous M2_GROWTH, SENTIMENT_CHANGE
- VARX_D_policy_exog_sentiment_endogenous: endogenous INF, UNRATE, INDPRO_GROWTH, M2_GROWTH, SENTIMENT_CHANGE; exogenous FEDFUNDS

Lag orders tested in the controlled re-check: 1 through 8. Lags above 8 were excluded from the final re-check because the previous search did not show a defensible gain that justified the added parameter burden. Crisis dummy alternatives tested for each candidate: no dummies, D_2008 only, D_COVID only, both D_2008 and D_COVID.

Balanced selection rule: models are ranked by stability, Portmanteau/Ljung-Box residual whiteness, residual ACF/CCF behavior, inflation and all-variable forecast performance, parameter count, and economic interpretability. High lag orders are penalized when they add complexity without clear diagnostic or forecasting gains.

Top VAR candidate ranking:

| model_type | candidate_name          | dummy_specification | lag_order | selection_score | stable | portmanteau_whiteness_p_value | acf_exceedance_share | inflation_RMSE | mean_relative_RMSE_vs_naive | obs_per_parameter_per_equation |
| ---------- | ----------------------- | ------------------- | --------- | --------------- | ------ | ----------------------------- | -------------------- | -------------- | --------------------------- | ------------------------------ |
| VAR        | VAR_core_plus_sentiment | no_dummies          | 5         | 77.7029         | True   | 0.0                           | 0.0667               | 0.176          | 0.7295                      | 12.8462                        |
| VAR        | VAR_core_plus_sentiment | no_dummies          | 4         | 77.1318         | True   | 0.0                           | 0.0833               | 0.177          | 0.7148                      | 15.9524                        |
| VAR        | VAR_core_plus_sentiment | no_dummies          | 6         | 76.9629         | True   | 0.0001                        | 0.0333               | 0.1808         | 0.7662                      | 10.7419                        |
| VAR        | VAR_core4               | no_dummies          | 4         | 76.804          | True   | 0.0                           | 0.0833               | 0.1775         | 0.6858                      | 19.7059                        |
| VAR        | VAR_core4               | no_dummies          | 5         | 76.6477         | True   | 0.0                           | 0.0833               | 0.1779         | 0.6941                      | 15.9048                        |
| VAR        | VAR_core4               | no_dummies          | 6         | 76.447          | True   | 0.0                           | 0.0417               | 0.1832         | 0.7367                      | 13.32                          |
| VAR        | VAR_core4               | no_dummies          | 3         | 75.7273         | True   | 0.0                           | 0.1042               | 0.1798         | 0.7362                      | 25.8462                        |
| VAR        | VAR_core_plus_sentiment | no_dummies          | 3         | 75.5367         | True   | 0.0                           | 0.1                  | 0.1799         | 0.7557                      | 21.0                           |
| VAR        | VAR_core_plus_sentiment | no_dummies          | 2         | 74.5078         | True   | 0.0                           | 0.1                  | 0.1808         | 0.8026                      | 30.6364                        |
| VAR        | VAR_core_plus_sentiment | no_dummies          | 7         | 74.1528         | True   | 0.0                           | 0.05                 | 0.1883         | 0.7798                      | 9.2222                         |

Top VARX candidate ranking:

| model_type | candidate_name                          | dummy_specification | lag_order | selection_score | stable | portmanteau_whiteness_p_value | acf_exceedance_share | inflation_RMSE | mean_relative_RMSE_vs_naive | obs_per_parameter_per_equation |
| ---------- | --------------------------------------- | ------------------- | --------- | --------------- | ------ | ----------------------------- | -------------------- | -------------- | --------------------------- | ------------------------------ |
| VARX       | VARX_D_policy_exog_sentiment_endogenous | no_dummies          | 3         | 73.4561         | True   | 0.0                           | 0.1333               | 0.186          | 0.7688                      | 19.7647                        |
| VARX       | VARX_A_policy_sentiment_exog            | no_dummies          | 4         | 72.9358         | True   | 0.0                           | 0.0625               | 0.192          | 0.9084                      | 17.6316                        |
| VARX       | VARX_A_policy_sentiment_exog            | D_2008_only         | 4         | 72.8892         | True   | 0.0                           | 0.0625               | 0.2083         | 0.75                        | 16.75                          |
| VARX       | VARX_D_policy_exog_sentiment_endogenous | no_dummies          | 4         | 72.8562         | True   | 0.0                           | 0.1                  | 0.1891         | 0.8425                      | 15.2273                        |
| VARX       | VARX_D_policy_exog_sentiment_endogenous | D_2008_only         | 3         | 72.4249         | True   | 0.0                           | 0.1333               | 0.1987         | 0.6881                      | 18.6667                        |
| VARX       | VARX_D_policy_exog_sentiment_endogenous | D_2008_only         | 4         | 71.9891         | True   | 0.0                           | 0.1                  | 0.2083         | 0.7228                      | 14.5652                        |
| VARX       | VARX_D_policy_exog_sentiment_endogenous | no_dummies          | 5         | 71.4502         | True   | 0.0003                        | 0.0667               | 0.1989         | 0.9329                      | 12.3704                        |
| VARX       | VARX_D_policy_exog_sentiment_endogenous | D_2008_only         | 5         | 71.1543         | True   | 0.0003                        | 0.0667               | 0.2161         | 0.7935                      | 11.9286                        |
| VARX       | VARX_A_policy_sentiment_exog            | no_dummies          | 5         | 70.7208         | True   | 0.0001                        | 0.0625               | 0.2039         | 1.0193                      | 14.5217                        |
| VARX       | VARX_A_policy_sentiment_exog            | D_2008_only         | 5         | 70.6915         | True   | 0.0001                        | 0.0625               | 0.2175         | 0.8367                      | 13.9167                        |

Best lag by information criterion:

| model_type | candidate_name   | dummy_specification | criterion | best_lag | criterion_value | Acceptable if                                                                                                    |
| ---------- | ---------------- | ------------------- | --------- | -------- | --------------- | ---------------------------------------------------------------------------------------------------------------- |
| VAR        | VAR_core4        | D_2008_and_D_COVID  | AIC       | 3        | -8.8439         | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core4        | D_2008_and_D_COVID  | BIC       | 2        | -8.2346         | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core4        | D_2008_and_D_COVID  | HQIC      | 3        | -8.5722         | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core4        | D_2008_and_D_COVID  | FPE       | 3        | 0.0001          | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core4        | D_2008_only         | AIC       | 3        | -8.8093         | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core4        | D_2008_only         | BIC       | 2        | -8.2414         | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core4        | D_2008_only         | HQIC      | 3        | -8.5557         | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core4        | D_2008_only         | FPE       | 3        | 0.0001          | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core4        | D_COVID_only        | AIC       | 3        | -8.8405         | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core4        | D_COVID_only        | BIC       | 2        | -8.2768         | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core4        | D_COVID_only        | HQIC      | 3        | -8.5869         | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core4        | D_COVID_only        | FPE       | 3        | 0.0001          | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core4        | no_dummies          | AIC       | 3        | -8.8155         | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core4        | no_dummies          | BIC       | 2        | -8.2961         | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core4        | no_dummies          | HQIC      | 3        | -8.58           | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core4        | no_dummies          | FPE       | 3        | 0.0001          | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core_plus_M2 | D_2008_and_D_COVID  | AIC       | 4        | -11.0236        | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core_plus_M2 | D_2008_and_D_COVID  | BIC       | 2        | -10.0832        | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core_plus_M2 | D_2008_and_D_COVID  | HQIC      | 3        | -10.5341        | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core_plus_M2 | D_2008_and_D_COVID  | FPE       | 4        | 0.0             | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |

Crisis dummy comparison:

| model_type | dummy_specification | best_candidate                          | best_lag | selection_score | aic      | bic     | portmanteau_whiteness_p_value | acf_exceedance_share | inflation_RMSE | mean_relative_RMSE_vs_naive | stable | Acceptable if                                                                                          |
| ---------- | ------------------- | --------------------------------------- | -------- | --------------- | -------- | ------- | ----------------------------- | -------------------- | -------------- | --------------------------- | ------ | ------------------------------------------------------------------------------------------------------ |
| VAR        | no_dummies          | VAR_core_plus_sentiment                 | 5        | 77.7029         | -6.0749  | -4.5915 | 0.0                           | 0.0667               | 0.176          | 0.7295                      | True   | dummies are preferred only if they improve diagnostics, forecast performance, or crisis interpretation |
| VAR        | D_2008_and_D_COVID  | VAR_core_plus_M2                        | 3        | 73.0565         | -10.9417 | -9.9193 | 0.0                           | 0.0833               | 0.1804         | 1.2533                      | True   | dummies are preferred only if they improve diagnostics, forecast performance, or crisis interpretation |
| VAR        | D_2008_only         | VAR_core_plus_sentiment                 | 4        | 69.6062         | -6.0743  | -4.8219 |                               | 0.0833               | 0.1938         | 1.3469                      | True   | dummies are preferred only if they improve diagnostics, forecast performance, or crisis interpretation |
| VAR        | D_COVID_only        | VAR_core_plus_M2                        | 3        | 66.2909         | -10.9436 | -9.9779 | 0.0                           | 0.0833               | 0.2331         | 1.1684                      | True   | dummies are preferred only if they improve diagnostics, forecast performance, or crisis interpretation |
| VARX       | no_dummies          | VARX_D_policy_exog_sentiment_endogenous | 3        | 73.4561         | -3.9382  | -2.9725 | 0.0                           | 0.1333               | 0.186          | 0.7688                      | True   | dummies are preferred only if they improve diagnostics, forecast performance, or crisis interpretation |
| VARX       | D_2008_only         | VARX_A_policy_sentiment_exog            | 4        | 72.8892         | -6.7659  | -5.8551 | 0.0                           | 0.0625               | 0.2083         | 0.75                        | True   | dummies are preferred only if they improve diagnostics, forecast performance, or crisis interpretation |
| VARX       | D_2008_and_D_COVID  | VARX_D_policy_exog_sentiment_endogenous | 3        | 69.9252         | -3.9439  | -2.8647 | 0.0                           | 0.15                 | 0.2115         | 0.7564                      | True   | dummies are preferred only if they improve diagnostics, forecast performance, or crisis interpretation |
| VARX       | D_COVID_only        | VARX_A_policy_sentiment_exog            | 7        | 65.5387         | -6.7221  | -5.255  |                               | 0.0417               | 0.2272         | 1.4267                      | True   | dummies are preferred only if they improve diagnostics, forecast performance, or crisis interpretation |

Final selected VAR:

| model_type | candidate_name          | dummy_specification | lag_order | selection_score | stable | portmanteau_whiteness_p_value | acf_exceedance_share | inflation_RMSE | mean_relative_RMSE_vs_naive | obs_per_parameter_per_equation | endogenous_variables                                   | exogenous_variables | bic     | hqic    | fpe    |
| ---------- | ----------------------- | ------------------- | --------- | --------------- | ------ | ----------------------------- | -------------------- | -------------- | --------------------------- | ------------------------------ | ------------------------------------------------------ | ------------------- | ------- | ------- | ------ |
| VAR        | VAR_core_plus_sentiment | no_dummies          | 5         | 77.7029         | True   | 0.0                           | 0.0667               | 0.176          | 0.7295                      | 12.8462                        | INF, FEDFUNDS, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | none                | -4.5915 | -5.4835 | 0.0023 |

Lag-selection interpretation for the final VAR: AIC preferred lag 3, BIC preferred lag 2, HQIC preferred lag 2, and FPE preferred lag 3. The selected lag 5 was therefore not chosen directly by information criteria. It was selected by the broader optimization rule because it remained stable, had strong out-of-sample inflation forecasting, limited positive-lag residual ACF exceedances, and preserved policy interpretability without severe overparameterization.

Final selected VARX:

| model_type | candidate_name               | dummy_specification | lag_order | selection_score | stable | portmanteau_whiteness_p_value | acf_exceedance_share | inflation_RMSE | mean_relative_RMSE_vs_naive | obs_per_parameter_per_equation | endogenous_variables                  | exogenous_variables        | bic     | hqic   | fpe    |
| ---------- | ---------------------------- | ------------------- | --------- | --------------- | ------ | ----------------------------- | -------------------- | -------------- | --------------------------- | ------------------------------ | ------------------------------------- | -------------------------- | ------- | ------ | ------ |
| VARX       | VARX_A_policy_sentiment_exog | no_dummies          | 4         | 72.9358         | True   | 0.0                           | 0.0625               | 0.192          | 0.9084                      | 17.6316                        | INF, UNRATE, INDPRO_GROWTH, M2_GROWTH | FEDFUNDS, SENTIMENT_CHANGE | -5.9157 | -6.436 | 0.0011 |

Lag-selection interpretation for the final VARX: AIC and FPE preferred lag 4, BIC preferred lag 1, and HQIC preferred lag 3. Lag 4 is defensible mainly because VARX is used for conditional forecasting and scenario analysis with externally supplied FEDFUNDS and sentiment paths.

VARX challenger note: `VARX_D_policy_exog_sentiment_endogenous` at lag 3 scored slightly higher in the mechanical composite ranking, mainly because of lower inflation RMSE and fewer total parameters. It was not adopted as the official VARX because it treats SENTIMENT_CHANGE as endogenous, reducing the intended scenario-design role, and it has weaker residual ACF behavior than VARX_A. It should be reported as a close robustness alternative, not ignored.

Rejected alternatives: lower-scoring alternatives were rejected mainly when they had weaker residual whiteness/autocorrelation diagnostics, higher overparameterization risk, worse inflation forecast RMSE, or less useful policy/scenario interpretation. A model with lower RMSE was not automatically selected if it was unstable, too highly parameterized, or weak for economic interpretation.

## 4. Final VAR Results

- Selected VAR specification: VAR_core_plus_sentiment
- Endogenous variables: INF, FEDFUNDS, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE
- Exogenous controls: none
- Lag order: 5
- Train/test split: 1995-02-01 to 2023-04-01 / 2023-05-01 to 2026-04-01
- Effective observations: 334
- Parameters per equation: 26; total parameters: 130; observations per parameter per equation: 12.85
- Stability: True; max inverse companion-root modulus: 0.9513 (desired < 1 in this display)
- Portmanteau whiteness p-value: 3.254e-06; min Ljung-Box p-value: 0.1852
- Inflation forecast RMSE/MAE: 0.1760 / 0.1230; relative RMSE vs no-leak naive: 0.9346

VAR lag-selection criteria for the selected specification:

| model_type | candidate_name          | dummy_specification | criterion | best_lag | criterion_value | Acceptable if                                                                                                    |
| ---------- | ----------------------- | ------------------- | --------- | -------- | --------------- | ---------------------------------------------------------------------------------------------------------------- |
| VAR        | VAR_core_plus_sentiment | no_dummies          | AIC       | 3        | -6.1224         | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core_plus_sentiment | no_dummies          | BIC       | 2        | -5.4137         | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core_plus_sentiment | no_dummies          | HQIC      | 2        | -5.7887         | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VAR        | VAR_core_plus_sentiment | no_dummies          | FPE       | 3        | 0.0022          | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |

Equation-level fit metrics:

| equation         | R_squared | residual_std_error | n_effective_obs | parameters_per_equation | Acceptable if                                                                                                                        |
| ---------------- | --------- | ------------------ | --------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| INF              | 0.3119    | 0.2526             | 334             | 26                      | higher R-squared and lower residual standard error indicate better in-sample fit, but forecasting and diagnostics are more important |
| FEDFUNDS         | 0.9967    | 0.1301             | 334             | 26                      | higher R-squared and lower residual standard error indicate better in-sample fit, but forecasting and diagnostics are more important |
| UNRATE           | 0.9198    | 0.544              | 334             | 26                      | higher R-squared and lower residual standard error indicate better in-sample fit, but forecasting and diagnostics are more important |
| INDPRO_GROWTH    | 0.3254    | 0.9693             | 334             | 26                      | higher R-squared and lower residual standard error indicate better in-sample fit, but forecasting and diagnostics are more important |
| SENTIMENT_CHANGE | 0.1715    | 3.8799             | 334             | 26                      | higher R-squared and lower residual standard error indicate better in-sample fit, but forecasting and diagnostics are more important |

Coefficient/significance summary:

| model_type | equation         | n_parameters | significant_parameters | share_significant | min_p_value |
| ---------- | ---------------- | ------------ | ---------------------- | ----------------- | ----------- |
| VAR        | INDPRO_GROWTH    | 26           | 9                      | 0.3462            | 0.0         |
| VAR        | SENTIMENT_CHANGE | 26           | 8                      | 0.3077            | 0.0         |
| VAR        | FEDFUNDS         | 26           | 7                      | 0.2692            | 0.0         |
| VAR        | UNRATE           | 26           | 6                      | 0.2308            | 0.0         |
| VAR        | INF              | 26           | 5                      | 0.1923            | 0.0         |

Robust inference sensitivity:

| model_type | equation         | n_parameters | classical_significant | hc3_significant | hac_significant | hc3_changed | hac_changed |
| ---------- | ---------------- | ------------ | --------------------- | --------------- | --------------- | ----------- | ----------- |
| VAR        | FEDFUNDS         | 26           | 7                     | 2               | 6               | 5           | 3           |
| VAR        | INDPRO_GROWTH    | 26           | 9                     | 2               | 7               | 7           | 6           |
| VAR        | INF              | 26           | 5                     | 2               | 3               | 3           | 2           |
| VAR        | SENTIMENT_CHANGE | 26           | 8                     | 7               | 9               | 1           | 1           |
| VAR        | UNRATE           | 26           | 6                     | 0               | 2               | 6           | 6           |

Residual tests:

| equation         | test                                | Acceptable if                                               | durbin_watson | ljung_box_p_value | arch_lm_p_value |
| ---------------- | ----------------------------------- | ----------------------------------------------------------- | ------------- | ----------------- | --------------- |
| INF              | Durbin-Watson / Ljung-Box / ARCH-LM | DW near 2; Ljung-Box p-value > 0.05; ARCH-LM p-value > 0.05 | 2.0076        | 0.1852            | 0.0009          |
| FEDFUNDS         | Durbin-Watson / Ljung-Box / ARCH-LM | DW near 2; Ljung-Box p-value > 0.05; ARCH-LM p-value > 0.05 | 2.0227        | 0.4153            | 0.918           |
| UNRATE           | Durbin-Watson / Ljung-Box / ARCH-LM | DW near 2; Ljung-Box p-value > 0.05; ARCH-LM p-value > 0.05 | 2.0115        | 0.7267            | 1.0             |
| INDPRO_GROWTH    | Durbin-Watson / Ljung-Box / ARCH-LM | DW near 2; Ljung-Box p-value > 0.05; ARCH-LM p-value > 0.05 | 2.0479        | 0.4722            | 0.4467          |
| SENTIMENT_CHANGE | Durbin-Watson / Ljung-Box / ARCH-LM | DW near 2; Ljung-Box p-value > 0.05; ARCH-LM p-value > 0.05 | 2.0176        | 0.9804            | 0.4018          |

Residual interpretation: equation-level Ljung-Box tests mostly pass, Durbin-Watson values are near 2, and positive-lag ACF exceedances are limited. However, the multivariate Portmanteau whiteness test rejects for the system. The model is useful but not perfectly white; IRF and FEVD interpretation requires caution.

Residual ACF summary, excluding lag 0:

| equation         | acf_exceedance_count | acf_exceedance_share | max_abs_acf_lag_1_to_12 | lag_of_max_abs_acf | Acceptable if                                                                      |
| ---------------- | -------------------- | -------------------- | ----------------------- | ------------------ | ---------------------------------------------------------------------------------- |
| INF              | 2                    | 0.1667               | 0.1596                  | 11                 | most positive-lag residual ACF bars stay inside +/-1.96/sqrt(T); lag 0 is excluded |
| FEDFUNDS         | 1                    | 0.0833               | 0.124                   | 6                  | most positive-lag residual ACF bars stay inside +/-1.96/sqrt(T); lag 0 is excluded |
| UNRATE           | 0                    | 0.0                  | 0.0952                  | 12                 | most positive-lag residual ACF bars stay inside +/-1.96/sqrt(T); lag 0 is excluded |
| INDPRO_GROWTH    | 1                    | 0.0833               | 0.1189                  | 10                 | most positive-lag residual ACF bars stay inside +/-1.96/sqrt(T); lag 0 is excluded |
| SENTIMENT_CHANGE | 0                    | 0.0                  | 0.0512                  | 7                  | most positive-lag residual ACF bars stay inside +/-1.96/sqrt(T); lag 0 is excluded |

Residual cross-correlation summary, including lag 0 for cross-equation pairs:

| source_residual  | target_residual  | lag_of_max_abs_ccf | ccf_at_max_abs_lag | max_abs_ccf | Acceptable if                                                                 |
| ---------------- | ---------------- | ------------------ | ------------------ | ----------- | ----------------------------------------------------------------------------- |
| UNRATE           | INDPRO_GROWTH    | 0                  | -0.7443            | 0.7443      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| INDPRO_GROWTH    | UNRATE           | 0                  | -0.7443            | 0.7443      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| SENTIMENT_CHANGE | UNRATE           | 0                  | -0.2528            | 0.2528      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| UNRATE           | SENTIMENT_CHANGE | 0                  | -0.2528            | 0.2528      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| FEDFUNDS         | INF              | 0                  | 0.2294             | 0.2294      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| INF              | FEDFUNDS         | 0                  | 0.2294             | 0.2294      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| FEDFUNDS         | SENTIMENT_CHANGE | 0                  | 0.2048             | 0.2048      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| SENTIMENT_CHANGE | FEDFUNDS         | 0                  | 0.2048             | 0.2048      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| INDPRO_GROWTH    | SENTIMENT_CHANGE | 0                  | 0.1917             | 0.1917      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| SENTIMENT_CHANGE | INDPRO_GROWTH    | 0                  | 0.1917             | 0.1917      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| INDPRO_GROWTH    | FEDFUNDS         | -9                 | -0.1713            | 0.1713      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| FEDFUNDS         | INDPRO_GROWTH    | 9                  | -0.1713            | 0.1713      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| INF              | INF              | 11                 | 0.1596             | 0.1596      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| UNRATE           | FEDFUNDS         | -9                 | 0.1271             | 0.1271      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| FEDFUNDS         | UNRATE           | 9                  | 0.1271             | 0.1271      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |

Residual normality:

| equation         | test        | Acceptable if                                   | jarque_bera_stat | jarque_bera_p_value | skewness | kurtosis_pearson | reject_normality_at_5pct |
| ---------------- | ----------- | ----------------------------------------------- | ---------------- | ------------------- | -------- | ---------------- | ------------------------ |
| INF              | Jarque-Bera | p-value > 0.05 for approximate normal residuals | 86.6879          | 0.0                 | -0.0343  | 5.5508           | True                     |
| FEDFUNDS         | Jarque-Bera | p-value > 0.05 for approximate normal residuals | 2432.3001        | 0.0                 | -2.0328  | 15.7941          | True                     |
| UNRATE           | Jarque-Bera | p-value > 0.05 for approximate normal residuals | 294475.3048      | 0.0                 | 9.7834   | 149.354          | True                     |
| INDPRO_GROWTH    | Jarque-Bera | p-value > 0.05 for approximate normal residuals | 29050.4413       | 0.0                 | -4.3978  | 48.5386          | True                     |
| SENTIMENT_CHANGE | Jarque-Bera | p-value > 0.05 for approximate normal residuals | 20.7319          | 0.0                 | -0.3387  | 4.0509           | True                     |

Normality/ARCH interpretation: Jarque-Bera normality is strongly rejected and the inflation equation shows ARCH effects. This is common in monthly macro data around crisis periods. It does not automatically invalidate point forecasts, but it weakens classical p-values and confidence intervals, motivating robust standard errors and Monte Carlo IRF bands.

Granger causality, significant predictive relationships:

| model_type | source        | target           | test                                | p_value | test_statistic | significant_at_5pct | Acceptable if                                                           |
| ---------- | ------------- | ---------------- | ----------------------------------- | ------- | -------------- | ------------------- | ----------------------------------------------------------------------- |
| VAR        | FEDFUNDS      | UNRATE           | VAR-system Granger causality F-test | 0.0     | 10.1581        | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VAR        | UNRATE        | INDPRO_GROWTH    | VAR-system Granger causality F-test | 0.0     | 9.2047         | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VAR        | FEDFUNDS      | INDPRO_GROWTH    | VAR-system Granger causality F-test | 0.0001  | 5.471          | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VAR        | INF           | SENTIMENT_CHANGE | VAR-system Granger causality F-test | 0.0002  | 4.9058         | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VAR        | INDPRO_GROWTH | UNRATE           | VAR-system Granger causality F-test | 0.0003  | 4.7555         | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VAR        | UNRATE        | FEDFUNDS         | VAR-system Granger causality F-test | 0.0028  | 3.6451         | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VAR        | INF           | INDPRO_GROWTH    | VAR-system Granger causality F-test | 0.0111  | 2.9761         | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VAR        | INDPRO_GROWTH | FEDFUNDS         | VAR-system Granger causality F-test | 0.0123  | 2.928          | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VAR        | INF           | FEDFUNDS         | VAR-system Granger causality F-test | 0.0463  | 2.2593         | True                | p-value < 0.05 indicates predictive causality, not structural causality |

FEDFUNDS shock IRF summary:

| response         | shock    | horizon | value   |
| ---------------- | -------- | ------- | ------- |
| FEDFUNDS         | FEDFUNDS | 1       | 0.1959  |
| FEDFUNDS         | FEDFUNDS | 3       | 0.2708  |
| FEDFUNDS         | FEDFUNDS | 6       | 0.3474  |
| FEDFUNDS         | FEDFUNDS | 12      | 0.4003  |
| FEDFUNDS         | FEDFUNDS | 24      | 0.32    |
| INDPRO_GROWTH    | FEDFUNDS | 1       | 0.3252  |
| INDPRO_GROWTH    | FEDFUNDS | 3       | -0.0705 |
| INDPRO_GROWTH    | FEDFUNDS | 6       | 0.059   |
| INDPRO_GROWTH    | FEDFUNDS | 12      | 0.0121  |
| INDPRO_GROWTH    | FEDFUNDS | 24      | 0.0004  |
| INF              | FEDFUNDS | 1       | 0.0424  |
| INF              | FEDFUNDS | 3       | -0.0101 |
| INF              | FEDFUNDS | 6       | 0.0006  |
| INF              | FEDFUNDS | 12      | -0.0002 |
| INF              | FEDFUNDS | 24      | 0.0008  |
| SENTIMENT_CHANGE | FEDFUNDS | 1       | 0.1424  |
| SENTIMENT_CHANGE | FEDFUNDS | 3       | -0.0276 |
| SENTIMENT_CHANGE | FEDFUNDS | 6       | -0.124  |
| SENTIMENT_CHANGE | FEDFUNDS | 12      | -0.0197 |
| SENTIMENT_CHANGE | FEDFUNDS | 24      | -0.0394 |
| UNRATE           | FEDFUNDS | 1       | -0.2546 |
| UNRATE           | FEDFUNDS | 3       | -0.1715 |
| UNRATE           | FEDFUNDS | 6       | -0.1788 |
| UNRATE           | FEDFUNDS | 12      | -0.1887 |
| UNRATE           | FEDFUNDS | 24      | -0.1249 |

Alternative Cholesky ordering robustness for FEDFUNDS shock:

| ordering_name                   | ordering                                               | shock    | response         | horizon | value   | lower_95 | upper_95 | Acceptable if                                                                                                               |
| ------------------------------- | ------------------------------------------------------ | -------- | ---------------- | ------- | ------- | -------- | -------- | --------------------------------------------------------------------------------------------------------------------------- |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | INF              | 1       | 0.0721  | 0.0402   | 0.1      | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | INF              | 3       | -0.008  | -0.0391  | 0.0263   | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | INF              | 6       | 0.002   | -0.021   | 0.023    | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | INF              | 12      | -0.0001 | -0.014   | 0.011    | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | INF              | 24      | 0.001   | -0.0103  | 0.0087   | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | UNRATE           | 1       | -0.284  | -0.3649  | -0.2033  | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | UNRATE           | 3       | -0.2074 | -0.3195  | -0.0891  | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | UNRATE           | 6       | -0.2097 | -0.3228  | -0.0861  | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | UNRATE           | 12      | -0.2158 | -0.337   | -0.0577  | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | UNRATE           | 24      | -0.1402 | -0.2535  | 0.0451   | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | INDPRO_GROWTH    | 1       | 0.371   | 0.2594   | 0.491    | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | INDPRO_GROWTH    | 3       | -0.0785 | -0.2085  | 0.0392   | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | INDPRO_GROWTH    | 6       | 0.0515  | -0.0292  | 0.1125   | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | INDPRO_GROWTH    | 12      | 0.0066  | -0.0359  | 0.0362   | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | INDPRO_GROWTH    | 24      | -0.0014 | -0.0297  | 0.0199   | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | SENTIMENT_CHANGE | 1       | 0.0106  | -0.3615  | 0.403    | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | SENTIMENT_CHANGE | 3       | -0.0335 | -0.4077  | 0.38     | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | SENTIMENT_CHANGE | 6       | -0.172  | -0.4145  | 0.0478   | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | SENTIMENT_CHANGE | 12      | -0.0383 | -0.1406  | 0.0373   | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| A_policy_first                  | FEDFUNDS, INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | FEDFUNDS | SENTIMENT_CHANGE | 24      | -0.0414 | -0.1016  | 0.0125   | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| B_slow_macro_first_policy_later | INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE, FEDFUNDS | FEDFUNDS | INF              | 1       | 0.0378  | 0.0106   | 0.0662   | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| B_slow_macro_first_policy_later | INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE, FEDFUNDS | FEDFUNDS | INF              | 3       | -0.0078 | -0.0377  | 0.0243   | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| B_slow_macro_first_policy_later | INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE, FEDFUNDS | FEDFUNDS | INF              | 6       | -0.0036 | -0.0204  | 0.0141   | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| B_slow_macro_first_policy_later | INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE, FEDFUNDS | FEDFUNDS | INF              | 12      | -0.0001 | -0.0127  | 0.01     | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| B_slow_macro_first_policy_later | INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE, FEDFUNDS | FEDFUNDS | INF              | 24      | 0.0007  | -0.0087  | 0.0086   | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| B_slow_macro_first_policy_later | INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE, FEDFUNDS | FEDFUNDS | UNRATE           | 1       | -0.2092 | -0.2633  | -0.1501  | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| B_slow_macro_first_policy_later | INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE, FEDFUNDS | FEDFUNDS | UNRATE           | 3       | -0.1379 | -0.2329  | -0.0275  | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| B_slow_macro_first_policy_later | INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE, FEDFUNDS | FEDFUNDS | UNRATE           | 6       | -0.1486 | -0.2558  | -0.0365  | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| B_slow_macro_first_policy_later | INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE, FEDFUNDS | FEDFUNDS | UNRATE           | 12      | -0.153  | -0.2651  | -0.0015  | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |
| B_slow_macro_first_policy_later | INF, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE, FEDFUNDS | FEDFUNDS | UNRATE           | 24      | -0.0997 | -0.2183  | 0.0688   | similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity |

IRF interpretation: the FEDFUNDS shock is not a clean textbook contractionary policy shock. Inflation rises slightly in the short run, industrial production rises initially, and unemployment falls after the shock in the baseline ordering. This likely mixes monetary tightening with the Federal Reserve's endogenous reaction to strong macroeconomic conditions and inflation pressure. It should be interpreted as a price-puzzle / identification issue, not as evidence that higher interest rates causally reduce unemployment. The IRF confidence bands are Monte Carlo bands generated with independent simulation seeds. Horizon-0 zero-width intervals can occur only where recursive Cholesky identification imposes an exact contemporaneous zero response; nonzero horizons should have separate lower and upper bounds.

Inflation FEVD summary:

| response | horizon | shock            | variance_share |
| -------- | ------- | ---------------- | -------------- |
| INF      | 1       | INF              | 1.0            |
| INF      | 1       | FEDFUNDS         | 0.0            |
| INF      | 1       | UNRATE           | 0.0            |
| INF      | 1       | INDPRO_GROWTH    | 0.0            |
| INF      | 1       | SENTIMENT_CHANGE | 0.0            |
| INF      | 6       | INF              | 0.9105         |
| INF      | 6       | FEDFUNDS         | 0.0387         |
| INF      | 6       | INDPRO_GROWTH    | 0.0339         |
| INF      | 6       | SENTIMENT_CHANGE | 0.0131         |
| INF      | 6       | UNRATE           | 0.0038         |
| INF      | 12      | INF              | 0.9053         |
| INF      | 12      | FEDFUNDS         | 0.0384         |
| INF      | 12      | INDPRO_GROWTH    | 0.036          |
| INF      | 12      | SENTIMENT_CHANGE | 0.0154         |
| INF      | 12      | UNRATE           | 0.0048         |
| INF      | 24      | INF              | 0.9049         |
| INF      | 24      | FEDFUNDS         | 0.0384         |
| INF      | 24      | INDPRO_GROWTH    | 0.0361         |
| INF      | 24      | SENTIMENT_CHANGE | 0.0155         |
| INF      | 24      | UNRATE           | 0.005          |

FEVD conclusion: inflation forecast-error variance is dominated by inflation's own innovations. At horizons 12 and 24, INF own-shock share is about 90%, while FEDFUNDS contributes only about 3.8%. Monetary-policy shocks contribute a smaller but nonzero share; they do not explain most inflation variation.

VAR forecast metrics:

| variable         | RMSE   | MAE    | naive_RMSE | relative_RMSE_vs_no_leak_naive | directional_accuracy | Acceptable if                                                                              |
| ---------------- | ------ | ------ | ---------- | ------------------------------ | -------------------- | ------------------------------------------------------------------------------------------ |
| INF              | 0.176  | 0.123  | 0.1883     | 0.9346                         | 0.6944               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |
| FEDFUNDS         | 0.2219 | 0.1804 | 0.6328     | 0.3507                         | 0.6429               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |
| UNRATE           | 0.3388 | 0.286  | 0.7184     | 0.4716                         | 0.5417               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |
| INDPRO_GROWTH    | 0.5778 | 0.467  | 0.5882     | 0.9822                         | 0.8889               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |
| SENTIMENT_CHANGE | 4.331  | 3.4629 | 4.7689     | 0.9082                         | 0.6111               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |

## 5. Final VARX Results

- Selected VARX specification: VARX_A_policy_sentiment_exog
- Endogenous variables: INF, UNRATE, INDPRO_GROWTH, M2_GROWTH
- Exogenous variables: FEDFUNDS, SENTIMENT_CHANGE
- Lag order: 4
- Train/test split: 1995-02-01 to 2023-04-01 / 2023-05-01 to 2026-04-01
- Effective observations: 335
- Parameters per equation: 19; total parameters: 76; observations per parameter per equation: 17.63
- Stability: True; max inverse companion-root modulus: 0.9457
- Portmanteau whiteness p-value: 9.098e-06; min Ljung-Box p-value: 0.06136
- Inflation forecast RMSE/MAE: 0.1920 / 0.1393; relative RMSE vs no-leak naive: 1.0196

VARX lag-selection criteria for the selected specification:

| model_type | candidate_name               | dummy_specification | criterion | best_lag | criterion_value | Acceptable if                                                                                                    |
| ---------- | ---------------------------- | ------------------- | --------- | -------- | --------------- | ---------------------------------------------------------------------------------------------------------------- |
| VARX       | VARX_A_policy_sentiment_exog | no_dummies          | AIC       | 4        | -6.781          | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VARX       | VARX_A_policy_sentiment_exog | no_dummies          | BIC       | 1        | -6.1477         | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VARX       | VARX_A_policy_sentiment_exog | no_dummies          | HQIC      | 3        | -6.4513         | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |
| VARX       | VARX_A_policy_sentiment_exog | no_dummies          | FPE       | 4        | 0.0011          | lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count |

Equation-level fit metrics:

| equation      | R_squared | residual_std_error | n_effective_obs | parameters_per_equation | Acceptable if                                                                                                                        |
| ------------- | --------- | ------------------ | --------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| INF           | 0.3004    | 0.2515             | 335             | 19                      | higher R-squared and lower residual standard error indicate better in-sample fit, but forecasting and diagnostics are more important |
| UNRATE        | 0.9205    | 0.5346             | 335             | 19                      | higher R-squared and lower residual standard error indicate better in-sample fit, but forecasting and diagnostics are more important |
| INDPRO_GROWTH | 0.3011    | 0.9741             | 335             | 19                      | higher R-squared and lower residual standard error indicate better in-sample fit, but forecasting and diagnostics are more important |
| M2_GROWTH     | 0.5214    | 0.4097             | 335             | 19                      | higher R-squared and lower residual standard error indicate better in-sample fit, but forecasting and diagnostics are more important |

Coefficient/significance summary:

| model_type | equation      | n_parameters | significant_parameters | share_significant | min_p_value |
| ---------- | ------------- | ------------ | ---------------------- | ----------------- | ----------- |
| VARX       | UNRATE        | 19           | 9                      | 0.4737            | 0.0         |
| VARX       | INDPRO_GROWTH | 19           | 8                      | 0.4211            | 0.0         |
| VARX       | M2_GROWTH     | 19           | 8                      | 0.4211            | 0.0         |
| VARX       | INF           | 19           | 4                      | 0.2105            | 0.0         |

Robust inference sensitivity:

| model_type | equation      | n_parameters | classical_significant | hc3_significant | hac_significant | hc3_changed | hac_changed |
| ---------- | ------------- | ------------ | --------------------- | --------------- | --------------- | ----------- | ----------- |
| VARX       | INDPRO_GROWTH | 19           | 8                     | 2               | 4               | 6           | 6           |
| VARX       | INF           | 19           | 4                     | 3               | 3               | 1           | 1           |
| VARX       | M2_GROWTH     | 19           | 8                     | 5               | 8               | 3           | 0           |
| VARX       | UNRATE        | 19           | 9                     | 0               | 1               | 9           | 8           |

Residual tests:

| equation      | test                                | Acceptable if                                               | durbin_watson | ljung_box_p_value | arch_lm_p_value |
| ------------- | ----------------------------------- | ----------------------------------------------------------- | ------------- | ----------------- | --------------- |
| INF           | Durbin-Watson / Ljung-Box / ARCH-LM | DW near 2; Ljung-Box p-value > 0.05; ARCH-LM p-value > 0.05 | 1.9765        | 0.331             | 0.0             |
| UNRATE        | Durbin-Watson / Ljung-Box / ARCH-LM | DW near 2; Ljung-Box p-value > 0.05; ARCH-LM p-value > 0.05 | 2.0576        | 0.5963            | 1.0             |
| INDPRO_GROWTH | Durbin-Watson / Ljung-Box / ARCH-LM | DW near 2; Ljung-Box p-value > 0.05; ARCH-LM p-value > 0.05 | 2.0774        | 0.0614            | 0.74            |
| M2_GROWTH     | Durbin-Watson / Ljung-Box / ARCH-LM | DW near 2; Ljung-Box p-value > 0.05; ARCH-LM p-value > 0.05 | 2.022         | 0.323             | 0.0             |

Residual interpretation: equation-level tests are mostly acceptable, but the VARX system-level Portmanteau whiteness test rejects. VARX should therefore be treated as a useful conditional forecasting/scenario tool, not a fully specified structural system.

Residual ACF summary, excluding lag 0:

| equation      | acf_exceedance_count | acf_exceedance_share | max_abs_acf_lag_1_to_12 | lag_of_max_abs_acf | Acceptable if                                                                      |
| ------------- | -------------------- | -------------------- | ----------------------- | ------------------ | ---------------------------------------------------------------------------------- |
| INF           | 1                    | 0.0833               | 0.1205                  | 11                 | most positive-lag residual ACF bars stay inside +/-1.96/sqrt(T); lag 0 is excluded |
| UNRATE        | 0                    | 0.0                  | 0.0921                  | 12                 | most positive-lag residual ACF bars stay inside +/-1.96/sqrt(T); lag 0 is excluded |
| INDPRO_GROWTH | 2                    | 0.1667               | 0.1617                  | 10                 | most positive-lag residual ACF bars stay inside +/-1.96/sqrt(T); lag 0 is excluded |
| M2_GROWTH     | 0                    | 0.0                  | 0.0946                  | 6                  | most positive-lag residual ACF bars stay inside +/-1.96/sqrt(T); lag 0 is excluded |

Residual cross-correlation summary, including lag 0 for cross-equation pairs:

| source_residual | target_residual | lag_of_max_abs_ccf | ccf_at_max_abs_lag | max_abs_ccf | Acceptable if                                                                 |
| --------------- | --------------- | ------------------ | ------------------ | ----------- | ----------------------------------------------------------------------------- |
| UNRATE          | INDPRO_GROWTH   | 0                  | -0.7347            | 0.7347      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| INDPRO_GROWTH   | UNRATE          | 0                  | -0.7347            | 0.7347      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| M2_GROWTH       | UNRATE          | 0                  | 0.5057             | 0.5057      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| UNRATE          | M2_GROWTH       | 0                  | 0.5057             | 0.5057      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| INDPRO_GROWTH   | M2_GROWTH       | 0                  | -0.4839            | 0.4839      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| M2_GROWTH       | INDPRO_GROWTH   | 0                  | -0.4839            | 0.4839      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| INF             | M2_GROWTH       | 0                  | -0.1876            | 0.1876      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| M2_GROWTH       | INF             | 0                  | -0.1876            | 0.1876      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| INDPRO_GROWTH   | INDPRO_GROWTH   | 10                 | 0.1617             | 0.1617      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| INF             | UNRATE          | 0                  | -0.1526            | 0.1526      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| UNRATE          | INF             | 0                  | -0.1526            | 0.1526      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| INF             | INDPRO_GROWTH   | 12                 | -0.1425            | 0.1425      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| INDPRO_GROWTH   | INF             | -12                | -0.1425            | 0.1425      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| INF             | INF             | 11                 | 0.1205             | 0.1205      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |
| M2_GROWTH       | M2_GROWTH       | 6                  | 0.0946             | 0.0946      | cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0 |

Residual normality:

| equation      | test        | Acceptable if                                   | jarque_bera_stat | jarque_bera_p_value | skewness | kurtosis_pearson | reject_normality_at_5pct |
| ------------- | ----------- | ----------------------------------------------- | ---------------- | ------------------- | -------- | ---------------- | ------------------------ |
| INF           | Jarque-Bera | p-value > 0.05 for approximate normal residuals | 173.7615         | 0.0                 | -0.3047  | 6.5463           | True                     |
| UNRATE        | Jarque-Bera | p-value > 0.05 for approximate normal residuals | 303892.4147      | 0.0                 | 9.9881   | 151.429          | True                     |
| INDPRO_GROWTH | Jarque-Bera | p-value > 0.05 for approximate normal residuals | 25836.4947       | 0.0                 | -4.224   | 45.8476          | True                     |
| M2_GROWTH     | Jarque-Bera | p-value > 0.05 for approximate normal residuals | 3892.7493        | 0.0                 | 1.9107   | 19.5242          | True                     |

Normality/ARCH interpretation: VARX residual normality is strongly rejected, and ARCH effects remain especially in INF and M2_GROWTH. This weakens classical inference and supports robust standard errors and scenario-response caution.

VARX endogenous Granger-style predictive relationships:

| model_type | source        | target        | test                                | p_value | test_statistic | significant_at_5pct | Acceptable if                                                           |
| ---------- | ------------- | ------------- | ----------------------------------- | ------- | -------------- | ------------------- | ----------------------------------------------------------------------- |
| VARX       | M2_GROWTH     | UNRATE        | VAR-system Granger causality F-test | 0.0     | 11.9485        | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VARX       | UNRATE        | INDPRO_GROWTH | VAR-system Granger causality F-test | 0.0     | 8.6513         | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VARX       | M2_GROWTH     | INDPRO_GROWTH | VAR-system Granger causality F-test | 0.0001  | 6.2664         | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VARX       | INF           | M2_GROWTH     | VAR-system Granger causality F-test | 0.0002  | 5.6725         | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VARX       | INF           | INDPRO_GROWTH | VAR-system Granger causality F-test | 0.0008  | 4.7524         | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VARX       | INDPRO_GROWTH | UNRATE        | VAR-system Granger causality F-test | 0.001   | 4.6603         | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VARX       | INDPRO_GROWTH | M2_GROWTH     | VAR-system Granger causality F-test | 0.0022  | 4.2146         | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VARX       | M2_GROWTH     | INF           | VAR-system Granger causality F-test | 0.0165  | 3.0432         | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VARX       | UNRATE        | M2_GROWTH     | VAR-system Granger causality F-test | 0.0405  | 2.5065         | True                | p-value < 0.05 indicates predictive causality, not structural causality |
| VARX       | INF           | UNRATE        | VAR-system Granger causality F-test | 0.0493  | 2.3876         | True                | p-value < 0.05 indicates predictive causality, not structural causality |

VARX forecast metrics:

| variable      | RMSE   | MAE    | naive_RMSE | relative_RMSE_vs_no_leak_naive | directional_accuracy | Acceptable if                                                                              |
| ------------- | ------ | ------ | ---------- | ------------------------------ | -------------------- | ------------------------------------------------------------------------------------------ |
| INF           | 0.192  | 0.1393 | 0.1883     | 1.0196                         | 0.5833               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |
| M2_GROWTH     | 0.2848 | 0.2509 | 1.1686     | 0.2437                         | 0.5833               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |
| INDPRO_GROWTH | 0.6539 | 0.4952 | 0.5882     | 1.1117                         | 0.8056               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |
| UNRATE        | 0.9043 | 0.8582 | 0.7184     | 1.2587                         | 0.5833               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |

VARX FEDFUNDS conditional/scenario response:

| shock    | response      | horizon | value   |
| -------- | ------------- | ------- | ------- |
| FEDFUNDS | INF           | 1       | -0.0036 |
| FEDFUNDS | INF           | 3       | 0.0045  |
| FEDFUNDS | INF           | 6       | -0.0007 |
| FEDFUNDS | INF           | 12      | 0.0001  |
| FEDFUNDS | INF           | 24      | -0.0    |
| FEDFUNDS | UNRATE        | 1       | -0.0033 |
| FEDFUNDS | UNRATE        | 3       | -0.0156 |
| FEDFUNDS | UNRATE        | 6       | -0.016  |
| FEDFUNDS | UNRATE        | 12      | -0.0135 |
| FEDFUNDS | UNRATE        | 24      | -0.0072 |
| FEDFUNDS | INDPRO_GROWTH | 1       | 0.0813  |
| FEDFUNDS | INDPRO_GROWTH | 3       | 0.0003  |
| FEDFUNDS | INDPRO_GROWTH | 6       | -0.0079 |
| FEDFUNDS | INDPRO_GROWTH | 12      | -0.0018 |
| FEDFUNDS | INDPRO_GROWTH | 24      | -0.0014 |
| FEDFUNDS | M2_GROWTH     | 1       | -0.0047 |
| FEDFUNDS | M2_GROWTH     | 3       | -0.0051 |
| FEDFUNDS | M2_GROWTH     | 6       | -0.0002 |
| FEDFUNDS | M2_GROWTH     | 12      | -0.0001 |
| FEDFUNDS | M2_GROWTH     | 24      | 0.0     |

Interpretation: VARX responses are conditional scenario responses, not structural IRFs. Future exogenous paths are imposed externally, so scenario results depend on the assumed exogenous shock path. VARX is useful because FEDFUNDS and SENTIMENT_CHANGE can be externally specified, but it is not the strongest selected inflation forecasting model.

## 6. Forecast Comparison

Selected VAR forecast metrics:

| variable         | RMSE   | MAE    | naive_RMSE | relative_RMSE_vs_no_leak_naive | directional_accuracy | Acceptable if                                                                              |
| ---------------- | ------ | ------ | ---------- | ------------------------------ | -------------------- | ------------------------------------------------------------------------------------------ |
| INF              | 0.176  | 0.123  | 0.1883     | 0.9346                         | 0.6944               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |
| FEDFUNDS         | 0.2219 | 0.1804 | 0.6328     | 0.3507                         | 0.6429               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |
| UNRATE           | 0.3388 | 0.286  | 0.7184     | 0.4716                         | 0.5417               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |
| INDPRO_GROWTH    | 0.5778 | 0.467  | 0.5882     | 0.9822                         | 0.8889               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |
| SENTIMENT_CHANGE | 4.331  | 3.4629 | 4.7689     | 0.9082                         | 0.6111               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |

Selected VARX forecast metrics:

| variable      | RMSE   | MAE    | naive_RMSE | relative_RMSE_vs_no_leak_naive | directional_accuracy | Acceptable if                                                                              |
| ------------- | ------ | ------ | ---------- | ------------------------------ | -------------------- | ------------------------------------------------------------------------------------------ |
| INF           | 0.192  | 0.1393 | 0.1883     | 1.0196                         | 0.5833               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |
| M2_GROWTH     | 0.2848 | 0.2509 | 1.1686     | 0.2437                         | 0.5833               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |
| INDPRO_GROWTH | 0.6539 | 0.4952 | 0.5882     | 1.1117                         | 0.8056               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |
| UNRATE        | 0.9043 | 0.8582 | 0.7184     | 1.2587                         | 0.5833               | lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better |

Inflation forecasts from all existing benchmark models:

| model                         | Acceptable if                                                   | target | forecast_design                          | rmse   | mae    | directional_accuracy |
| ----------------------------- | --------------------------------------------------------------- | ------ | ---------------------------------------- | ------ | ------ | -------------------- |
| Ridge Regression              | Lower RMSE/MAE is better; higher directional accuracy is better | inf    | one_step_lagged_features_final_36_months | 0.1595 | 0.1168 | 0.6111               |
| Random Walk (one-step)        | Lower RMSE/MAE is better; higher directional accuracy is better | inf    | one_step_lagged_features_final_36_months | 0.1739 | 0.132  |                      |
| Random Forest                 | Lower RMSE/MAE is better; higher directional accuracy is better | inf    | one_step_lagged_features_final_36_months | 0.1794 | 0.1303 | 0.6667               |
| Gradient Boosting             | Lower RMSE/MAE is better; higher directional accuracy is better | inf    | one_step_lagged_features_final_36_months | 0.1836 | 0.139  | 0.6111               |
| Random Walk (direct 36-month) | Lower RMSE/MAE is better; higher directional accuracy is better | inf    | final_36_months                          | 0.1883 | 0.1515 |                      |
| VAR                           | Lower RMSE/MAE is better; higher directional accuracy is better | inf    | final_36_months                          | 0.1991 | 0.1696 | 0.2778               |
| VARX                          | Lower RMSE/MAE is better; higher directional accuracy is better | inf    | final_36_months                          | 0.2176 | 0.1831 | 0.2778               |

Multi-horizon inflation forecast comparison:

| horizon | model            | Acceptable if                                                                              | n_forecasts | rmse   | mae    | directional_accuracy | relative_rmse_vs_random_walk |
| ------- | ---------------- | ------------------------------------------------------------------------------------------ | ----------- | ------ | ------ | -------------------- | ---------------------------- |
| 1       | VAR              | Lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better | 23          | 0.1611 | 0.1172 | 0.7391               | 0.8681                       |
| 1       | Ridge Regression | Lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better | 23          | 0.1612 | 0.1133 | 0.6087               | 0.8686                       |
| 1       | VARX             | Lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better | 23          | 0.171  | 0.1236 | 0.6522               | 0.9218                       |
| 1       | Random Walk      | Lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better | 23          | 0.1855 | 0.137  | 0.0                  | 1.0                          |
| 3       | VARX             | Lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better | 21          | 0.2013 | 0.1487 | 0.7619               | 0.8923                       |
| 3       | VAR              | Lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better | 21          | 0.2038 | 0.1552 | 0.8095               | 0.9034                       |
| 3       | Ridge Regression | Lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better | 21          | 0.208  | 0.1433 | 0.5714               | 0.9219                       |
| 3       | Random Walk      | Lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better | 21          | 0.2257 | 0.1786 | 0.0                  | 1.0                          |
| 6       | VAR              | Lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better | 18          | 0.2045 | 0.1456 | 0.6667               | 0.7687                       |
| 6       | VARX             | Lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better | 18          | 0.2063 | 0.1505 | 0.7222               | 0.7757                       |
| 6       | Ridge Regression | Lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better | 18          | 0.2392 | 0.1772 | 0.6667               | 0.8993                       |
| 6       | Random Walk      | Lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better | 18          | 0.266  | 0.2111 | 0.0                  | 1.0                          |
| 12      | VAR              | Lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better | 12          | 0.2253 | 0.1576 | 0.8333               | 0.7121                       |
| 12      | VARX             | Lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better | 12          | 0.2287 | 0.1708 | 0.8333               | 0.7228                       |
| 12      | Ridge Regression | Lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better | 12          | 0.3048 | 0.2322 | 0.5833               | 0.9634                       |
| 12      | Random Walk      | Lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better | 12          | 0.3163 | 0.22   | 0.0                  | 1.0                          |

Forecast conclusion: among the selected econometric models, VAR has the lower optimized inflation RMSE (0.1760). For inflation specifically, selected VAR RMSE is about 0.176, the no-leak naive RMSE is about 0.188, and selected VARX RMSE is about 0.192. These are optimized selected-model recursive holdout metrics. They differ from the all-benchmark one-step/direct forecast table, where the VAR RMSE can appear around 0.199 because the forecast protocol is different. Ridge may be best for pure one-step prediction, but it does not provide Granger causality, IRF, FEVD, Cholesky identification, or structural macroeconomic transmission interpretation. VAR is the main policy-interpretation model; VARX remains useful for conditional policy/scenario forecasting even when it is not the strongest inflation forecaster.

## 7. Restricted VAR/VARX Parsimony Robustness

Restricted models were added only as robustness checks. They are not automatic replacements for the official unrestricted baselines. Restrictions remove whole lag blocks only when the source block lacks Granger/predictive evidence, is jointly insignificant in the equation-level OLS regression, has weak HC3/HAC robust lag-level evidence, and is not protected by economic logic. Own lags, the central FEDFUNDS policy channel, INF -> FEDFUNDS policy-reaction logic, and the UNRATE/INDPRO_GROWTH real-side pair are retained conservatively.

Restricted VAR summary:

| model_type | restricted_model               | baseline_model          | endogenous_variables                                   | exogenous_variables | lag_order | n_train_effective | n_test | k_endogenous | k_exogenous | baseline_total_parameters | restricted_total_parameters | parameters_removed | parameter_reduction_share | remaining_parameters_per_equation                                          | stable | max_companion_eigenvalue_modulus | inflation_RMSE | inflation_MAE | baseline_inflation_RMSE | inflation_RMSE_change_restricted_minus_baseline | mean_RMSE | mean_MAE | min_ljung_box_p_value | baseline_min_ljung_box_p_value | mean_ljung_box_p_value | max_acf_exceedance_share | baseline_acf_exceedance_share | max_abs_cross_ccf_including_lag0 | baseline_max_abs_cross_ccf | min_jarque_bera_p_value | baseline_min_jarque_bera_p_value | min_arch_lm_p_value | baseline_min_arch_lm_p_value | portmanteau_p_value_approx | baseline_portmanteau_whiteness_p_value | interpretation                  | Acceptable if                                                                                                   |
| ---------- | ------------------------------ | ----------------------- | ------------------------------------------------------ | ------------------- | --------- | ----------------- | ------ | ------------ | ----------- | ------------------------- | --------------------------- | ------------------ | ------------------------- | -------------------------------------------------------------------------- | ------ | -------------------------------- | -------------- | ------------- | ----------------------- | ----------------------------------------------- | --------- | -------- | --------------------- | ------------------------------ | ---------------------- | ------------------------ | ----------------------------- | -------------------------------- | -------------------------- | ----------------------- | -------------------------------- | ------------------- | ---------------------------- | -------------------------- | -------------------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| VAR        | Restricted_VAR_block_parsimony | VAR_core_plus_sentiment | INF, FEDFUNDS, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE | none                | 5         | 334               | 36     | 5            | 0           | 130.0                     | 100                         | 30                 | 0.2308                    | INF: 11; FEDFUNDS: 21; UNRATE: 16; INDPRO_GROWTH: 26; SENTIMENT_CHANGE: 26 | True   | 0.9517                           | 0.1785         | 0.1244        | 0.176                   | 0.0025                                          | 1.1312    | 0.9054   | 0.3683                | 0.1852                         | 0.616                  | 0.1667                   | 0.0667                        | 0.7331                           | 0.7443                     | 0.0                     | 0.0                              | 0.0                 | 0.0009                       | 0.2771                     | 0.0                                    | useful parsimonious alternative | restricted model is stable, materially more parsimonious, and does not worsen forecasts or residual diagnostics |

Restricted VAR block restrictions imposed:

| equation | source_variable  | removed_parameters | granger_p_value | classical_block_f_p_value | min_hc3_p_value_in_block | min_hac_p_value_in_block | reason                                                                                                      |
| -------- | ---------------- | ------------------ | --------------- | ------------------------- | ------------------------ | ------------------------ | ----------------------------------------------------------------------------------------------------------- |
| INF      | UNRATE           | 5                  | 0.375           | 0.3769                    | 0.4328                   | 0.1254                   | removed because the source block does not show Granger/block evidence and robust lag-level evidence is weak |
| INF      | INDPRO_GROWTH    | 5                  | 0.1668          | 0.1696                    | 0.3049                   | 0.1941                   | removed because the source block does not show Granger/block evidence and robust lag-level evidence is weak |
| INF      | SENTIMENT_CHANGE | 5                  | 0.4293          | 0.4307                    | 0.1651                   | 0.1393                   | removed because the source block does not show Granger/block evidence and robust lag-level evidence is weak |
| FEDFUNDS | SENTIMENT_CHANGE | 5                  | 0.5835          | 0.584                     | 0.3288                   | 0.3403                   | removed because the source block does not show Granger/block evidence and robust lag-level evidence is weak |
| UNRATE   | INF              | 5                  | 0.7867          | 0.7864                    | 0.2553                   | 0.1899                   | removed because the source block does not show Granger/block evidence and robust lag-level evidence is weak |
| UNRATE   | SENTIMENT_CHANGE | 5                  | 0.2822          | 0.2845                    | 0.2924                   | 0.0724                   | removed because the source block does not show Granger/block evidence and robust lag-level evidence is weak |

Restricted VAR forecast comparison:

| model_type | restricted_model               | variable         | baseline_RMSE | restricted_RMSE | RMSE_change_restricted_minus_baseline | restricted_better_RMSE | baseline_MAE | restricted_MAE | MAE_change_restricted_minus_baseline | restricted_better_MAE | restricted_relative_RMSE_vs_naive | restricted_directional_accuracy | Acceptable if                                                              |
| ---------- | ------------------------------ | ---------------- | ------------- | --------------- | ------------------------------------- | ---------------------- | ------------ | -------------- | ------------------------------------ | --------------------- | --------------------------------- | ------------------------------- | -------------------------------------------------------------------------- |
| VAR        | Restricted_VAR_block_parsimony | INF              | 0.176         | 0.1785          | 0.0025                                | False                  | 0.123        | 0.1244         | 0.0015                               | False                 | 0.948                             | 0.6944                          | restricted model should preserve or improve RMSE/MAE with fewer parameters |
| VAR        | Restricted_VAR_block_parsimony | FEDFUNDS         | 0.2219        | 0.2318          | 0.0098                                | False                  | 0.1804       | 0.1879         | 0.0075                               | False                 | 0.3662                            | 0.7143                          | restricted model should preserve or improve RMSE/MAE with fewer parameters |
| VAR        | Restricted_VAR_block_parsimony | UNRATE           | 0.3388        | 0.3427          | 0.0039                                | False                  | 0.286        | 0.2886         | 0.0026                               | False                 | 0.4771                            | 0.5417                          | restricted model should preserve or improve RMSE/MAE with fewer parameters |
| VAR        | Restricted_VAR_block_parsimony | INDPRO_GROWTH    | 0.5778        | 0.5806          | 0.0028                                | False                  | 0.467        | 0.4683         | 0.0013                               | False                 | 0.987                             | 0.8889                          | restricted model should preserve or improve RMSE/MAE with fewer parameters |
| VAR        | Restricted_VAR_block_parsimony | SENTIMENT_CHANGE | 4.331         | 4.3225          | -0.0085                               | True                   | 3.4629       | 3.4575         | -0.0054                              | True                  | 0.9064                            | 0.6111                          | restricted model should preserve or improve RMSE/MAE with fewer parameters |

Restricted VAR interpretation: the restricted VAR is a useful parsimonious alternative if it preserves diagnostics and forecast accuracy, but the unrestricted VAR remains the official policy-interpretation baseline because it preserves complete dynamic channels for IRF/FEVD analysis.

Restricted VARX summary:

| model_type | restricted_model                | baseline_model               | endogenous_variables                  | exogenous_variables        | lag_order | n_train_effective | n_test | k_endogenous | k_exogenous | baseline_total_parameters | restricted_total_parameters | parameters_removed | parameter_reduction_share | remaining_parameters_per_equation                     | stable | max_companion_eigenvalue_modulus | inflation_RMSE | inflation_MAE | baseline_inflation_RMSE | inflation_RMSE_change_restricted_minus_baseline | mean_RMSE | mean_MAE | min_ljung_box_p_value | baseline_min_ljung_box_p_value | mean_ljung_box_p_value | max_acf_exceedance_share | baseline_acf_exceedance_share | max_abs_cross_ccf_including_lag0 | baseline_max_abs_cross_ccf | min_jarque_bera_p_value | baseline_min_jarque_bera_p_value | min_arch_lm_p_value | baseline_min_arch_lm_p_value | portmanteau_p_value_approx | baseline_portmanteau_whiteness_p_value | interpretation                  | Acceptable if                                                                                                   |
| ---------- | ------------------------------- | ---------------------------- | ------------------------------------- | -------------------------- | --------- | ----------------- | ------ | ------------ | ----------- | ------------------------- | --------------------------- | ------------------ | ------------------------- | ----------------------------------------------------- | ------ | -------------------------------- | -------------- | ------------- | ----------------------- | ----------------------------------------------- | --------- | -------- | --------------------- | ------------------------------ | ---------------------- | ------------------------ | ----------------------------- | -------------------------------- | -------------------------- | ----------------------- | -------------------------------- | ------------------- | ---------------------------- | -------------------------- | -------------------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| VARX       | Restricted_VARX_block_parsimony | VARX_A_policy_sentiment_exog | INF, UNRATE, INDPRO_GROWTH, M2_GROWTH | FEDFUNDS, SENTIMENT_CHANGE | 4         | 335               | 36     | 4            | 2           | 76.0                      | 64                          | 12                 | 0.1579                    | INF: 11; UNRATE: 15; INDPRO_GROWTH: 19; M2_GROWTH: 19 | True   | 0.9486                           | 0.184          | 0.1325        | 0.192                   | -0.008                                          | 0.4832    | 0.4118   | 0.0614                | 0.0614                         | 0.3611                 | 0.1667                   | 0.0625                        | 0.7238                           | 0.7347                     | 0.0                     | 0.0                              | 0.0                 | 0.0                          | 0.0037                     | 0.0                                    | useful parsimonious alternative | restricted model is stable, materially more parsimonious, and does not worsen forecasts or residual diagnostics |

Restricted VARX block restrictions imposed:

| equation | source_variable | removed_parameters | granger_p_value | classical_block_f_p_value | min_hc3_p_value_in_block | min_hac_p_value_in_block | reason                                                                                                      |
| -------- | --------------- | ------------------ | --------------- | ------------------------- | ------------------------ | ------------------------ | ----------------------------------------------------------------------------------------------------------- |
| INF      | UNRATE          | 4                  | 0.3853          | 0.3543                    | 0.3818                   | 0.0747                   | removed because the source block does not show Granger/block evidence and robust lag-level evidence is weak |
| INF      | INDPRO_GROWTH   | 4                  | 0.0938          | 0.0973                    | 0.3025                   | 0.2149                   | removed because the source block does not show Granger/block evidence and robust lag-level evidence is weak |
| UNRATE   | INF             | 4                  | 0.2086          | 0.051                     | 0.2207                   | 0.133                    | removed because the source block does not show Granger/block evidence and robust lag-level evidence is weak |

Restricted VARX forecast comparison:

| model_type | restricted_model                | variable      | baseline_RMSE | restricted_RMSE | RMSE_change_restricted_minus_baseline | restricted_better_RMSE | baseline_MAE | restricted_MAE | MAE_change_restricted_minus_baseline | restricted_better_MAE | restricted_relative_RMSE_vs_naive | restricted_directional_accuracy | Acceptable if                                                              |
| ---------- | ------------------------------- | ------------- | ------------- | --------------- | ------------------------------------- | ---------------------- | ------------ | -------------- | ------------------------------------ | --------------------- | --------------------------------- | ------------------------------- | -------------------------------------------------------------------------- |
| VARX       | Restricted_VARX_block_parsimony | INF           | 0.192         | 0.184           | -0.008                                | True                   | 0.1393       | 0.1325         | -0.0068                              | True                  | 0.9772                            | 0.6389                          | restricted model should preserve or improve RMSE/MAE with fewer parameters |
| VARX       | Restricted_VARX_block_parsimony | UNRATE        | 0.9043        | 0.8125          | -0.0918                               | True                   | 0.8582       | 0.7738         | -0.0844                              | True                  | 1.1309                            | 0.5833                          | restricted model should preserve or improve RMSE/MAE with fewer parameters |
| VARX       | Restricted_VARX_block_parsimony | INDPRO_GROWTH | 0.6539        | 0.6507          | -0.0033                               | True                   | 0.4952       | 0.4908         | -0.0043                              | True                  | 1.1061                            | 0.8333                          | restricted model should preserve or improve RMSE/MAE with fewer parameters |
| VARX       | Restricted_VARX_block_parsimony | M2_GROWTH     | 0.2848        | 0.2855          | 0.0008                                | False                  | 0.2509       | 0.2502         | -0.0008                              | True                  | 0.2443                            | 0.6111                          | restricted model should preserve or improve RMSE/MAE with fewer parameters |

Restricted VARX interpretation: the restricted VARX is a parsimonious conditional-forecasting robustness check. Exogenous FEDFUNDS and SENTIMENT_CHANGE are retained for scenario design, even when some individual coefficients are weak.


## 8. Main Economic Conclusions

- Inflation dynamics: inflation is forecast using its own lagged dynamics plus policy, labor-market, production, money, and sentiment channels. Granger-significant relationships above show which variables have predictive content in the optimized system.
- FEDFUNDS predictive content: Granger results show FEDFUNDS predicts UNRATE and INDPRO_GROWTH more strongly than it predicts inflation directly. Inflation predicting FEDFUNDS is consistent with a policy-reaction function.
- FEDFUNDS shocks: a positive short-run inflation response appears after a FEDFUNDS shock (h1=0.0424, h6=0.0006), so the price puzzle should be discussed with identification caveats.
- Unemployment response: unemployment falls after a FEDFUNDS shock in the baseline ordering, which should not be read as a clean causal contractionary-policy effect. It likely reflects endogenous policy reaction and identification limitations.
- Industrial production response: industrial production rises initially and then becomes weak/sign-changing after a FEDFUNDS shock, reinforcing the identification warning.
- FEVD: inflation forecast-error variance is mostly own inflation shocks. FEDFUNDS contributes around 3.8% by horizons 12 and 24, so monetary policy is present but not dominant in FEVD.
- VAR vs VARX: VAR is better for policy interpretation because it supports Granger causality, IRF, FEVD, and endogenous feedback. VARX is better for conditional/scenario forecasting when externally supplied FEDFUNDS and sentiment paths are substantively meaningful, but it is weaker for selected inflation forecasting.

## 9. Weaknesses and Warnings

- Residual autocorrelation: macroeconomic VAR residuals are rarely perfectly white. Equation-level diagnostics are mostly acceptable, but system-level Portmanteau whiteness rejects for both selected VAR and VARX.
- Non-normality: Jarque-Bera/system normality rejects strongly because crisis periods create fat tails. This affects classical p-values and confidence intervals more than point forecasts.
- ARCH effects: low ARCH-LM p-values imply time-varying volatility, especially in VAR INF and VARX INF/M2_GROWTH. Robust or bootstrap inference is preferable.
- Overparameterization: high lags and full six-variable systems can weaken degrees of freedom. The selected models balance diagnostics and interpretability rather than blindly choosing AIC.
- Cholesky ordering: VAR IRFs depend on recursive identification and variable ordering. Short-run responses are conditional, not automatic causal truth.
- Price puzzle: if inflation rises after a positive FEDFUNDS shock, discuss endogenous policy reaction, omitted expectations/commodity channels, and identification limitations.
- VARX limitation: scenario responses condition on imposed exogenous paths and are not standard structural IRFs.
- Restricted-model limitation: block restrictions improve parsimony but are still data-driven. They are robustness checks, not evidence that excluded channels are structurally zero.
- Data limitation: monthly U.S. macro data contain regime shifts from 2008 and COVID; results may be sensitive to crisis dummy treatment and train/test split.

## 10. Files Produced

Key optimization outputs:

- `outputs/tables/optimized_var_candidate_search.csv`
- `outputs/tables/optimized_varx_candidate_search.csv`
- `outputs/tables/optimized_candidate_model_ranking.csv`
- `outputs/tables/optimized_lag_selection_full.csv`
- `outputs/tables/optimized_lag_selection_best_by_criterion.csv`
- `outputs/tables/optimized_crisis_dummy_search.csv`
- `outputs/tables/optimized_final_model_specs.csv`
- `outputs/tables/optimized_final_var_metrics.csv`
- `outputs/tables/optimized_final_var_residual_tests.csv`
- `outputs/tables/optimized_final_var_residual_acf.csv`
- `outputs/tables/optimized_final_var_residual_ccf.csv`
- `outputs/tables/optimized_final_var_granger.csv`
- `outputs/tables/optimized_final_var_irf_key_fedfunds.csv`
- `outputs/tables/optimized_final_var_fevd_key_inflation.csv`
- `outputs/tables/academic_var_irf_confidence_intervals.csv`
- `outputs/tables/academic_cholesky_ordering_robustness.csv`
- `outputs/figures/academic_cholesky_ordering_comparison.png`
- `outputs/tables/academic_var_parameter_significance_robust.csv`
- `outputs/tables/academic_varx_parameter_significance_robust.csv`
- `outputs/tables/optimized_final_varx_metrics.csv`
- `outputs/tables/optimized_final_varx_residual_tests.csv`
- `outputs/tables/optimized_final_varx_residual_acf.csv`
- `outputs/tables/optimized_final_varx_residual_ccf.csv`
- `outputs/tables/optimized_final_varx_granger.csv`
- `outputs/tables/optimized_final_varx_scenario_response.csv`
- `outputs/tables/restricted_var_restrictions.csv`
- `outputs/tables/restricted_var_metrics.csv`
- `outputs/tables/restricted_var_residual_diagnostics.csv`
- `outputs/tables/restricted_var_forecast_comparison.csv`
- `outputs/tables/restricted_varx_restrictions.csv`
- `outputs/tables/restricted_varx_metrics.csv`
- `outputs/tables/restricted_varx_residual_diagnostics.csv`
- `outputs/tables/restricted_varx_forecast_comparison.csv`

Context file:

- `PROJECT_RESULTS_CONTEXT.md`
