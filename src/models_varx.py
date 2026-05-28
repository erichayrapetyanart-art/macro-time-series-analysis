from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX


def equation_fit_metrics_from_residuals(
    residuals: pd.DataFrame,
    actual: pd.DataFrame,
    params_per_equation: int,
) -> pd.DataFrame:
    aligned_actual = actual.loc[residuals.index, residuals.columns]
    rows = []
    for column in residuals.columns:
        sse = float(np.sum(residuals[column] ** 2))
        centered = aligned_actual[column] - aligned_actual[column].mean()
        sst = float(np.sum(centered**2))
        rows.append(
            {
                "equation": column,
                "R_squared": 1 - sse / sst if sst > 0 else np.nan,
                "residual_std_error": float(np.sqrt(sse / max(len(residuals) - params_per_equation, 1))),
                "n_effective_obs": int(len(residuals)),
                "parameters_per_equation": int(params_per_equation),
                "Acceptable if": "higher R-squared and lower residual standard error indicate better in-sample fit, but forecasting and diagnostics are more important",
            }
        )
    return pd.DataFrame(rows)


def select_varx_lags(
    train_endog: pd.DataFrame,
    train_exog: pd.DataFrame | None,
    max_lag: int,
) -> pd.DataFrame:
    if train_endog.shape[1] >= 2:
        lag_sel = VAR(train_endog, exog=train_exog).select_order(maxlags=max_lag)
        return pd.DataFrame(
            {
                "lag": list(range(len(lag_sel.ics["aic"]))),
                "AIC": lag_sel.ics["aic"],
                "BIC": lag_sel.ics["bic"],
                "HQIC": lag_sel.ics["hqic"],
                "FPE": lag_sel.ics["fpe"],
                "Acceptable if": "Lower information criterion is preferred; final lag must preserve degrees of freedom",
            }
        )

    y = train_endog.iloc[:, 0]
    rows = []
    for lag in range(1, max_lag + 1):
        try:
            fitted = SARIMAX(
                y,
                exog=train_exog,
                order=(lag, 0, 0),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            n = fitted.nobs
            k = len(fitted.params)
            rows.append(
                {
                    "lag": lag,
                    "AIC": fitted.aic,
                    "BIC": fitted.bic,
                    "HQIC": -2 * fitted.llf + 2 * k * np.log(np.log(n)),
                    "FPE": np.nan,
                    "Acceptable if": "Lower information criterion is preferred; final lag must preserve degrees of freedom",
                }
            )
        except Exception:
            rows.append({"lag": lag, "AIC": np.nan, "BIC": np.nan, "HQIC": np.nan, "FPE": np.nan})
    return pd.DataFrame(rows)


def fit_varx_system(
    data: pd.DataFrame,
    split_date: str,
    endog: list[str],
    exog: list[str],
    lag_order: int,
    target: str,
) -> dict:
    split_ts = pd.Timestamp(split_date)
    train = data.loc[:split_ts].copy()
    test = data.loc[data.index > split_ts].copy()
    train_endog = train[endog]
    test_endog = test[endog]
    train_exog = train[exog] if exog else None
    test_exog = test[exog] if exog else None

    warning = ""
    if len(endog) >= 2:
        fitted = VAR(train_endog, exog=train_exog).fit(lag_order, trend="c")
        points, lower, upper = fitted.forecast_interval(
            train_endog.values[-lag_order:],
            steps=len(test_endog),
            alpha=0.05,
            exog_future=test_exog.values if test_exog is not None else None,
        )
        forecasts = pd.DataFrame(points, index=test.index, columns=endog)
        lower_df = pd.DataFrame(lower, index=test.index, columns=[f"{col}_lower_95" for col in endog])
        upper_df = pd.DataFrame(upper, index=test.index, columns=[f"{col}_upper_95" for col in endog])
        residuals = fitted.resid
        roots = 1 / fitted.roots
        roots_df = pd.DataFrame({"real": roots.real, "imag": roots.imag, "modulus": np.abs(roots)})
        stable = bool(fitted.is_stable())
        params = fitted.params
        stderr = fitted.stderr
        pvalues = fitted.pvalues
        tvalues = fitted.tvalues
        try:
            whiteness_p_value = float(fitted.test_whiteness(nlags=min(12, max(lag_order + 1, 2))).pvalue)
        except Exception:
            whiteness_p_value = None
        try:
            normality_p_value = float(fitted.test_normality().pvalue)
        except Exception:
            normality_p_value = None
        fit_model_name = "VARX"
        fit_aic = fitted.aic
        fit_bic = fitted.bic
        fit_hqic = fitted.hqic
        effective_obs = int(fitted.nobs)
        params_per_equation = int(fitted.params.shape[0])
    else:
        col = endog[0]
        fitted = SARIMAX(
            train[col],
            exog=train_exog,
            order=(lag_order, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        pred = fitted.get_forecast(steps=len(test), exog=test_exog)
        conf = pred.conf_int()
        forecasts = pd.DataFrame({col: pred.predicted_mean}, index=test.index)
        lower_df = pd.DataFrame({f"{col}_lower_95": conf.iloc[:, 0].values}, index=test.index)
        upper_df = pd.DataFrame({f"{col}_upper_95": conf.iloc[:, 1].values}, index=test.index)
        residuals = pd.DataFrame({col: fitted.resid.dropna()})
        roots = 1 / fitted.arroots if len(fitted.arroots) else np.array([])
        roots_df = pd.DataFrame({"real": roots.real, "imag": roots.imag, "modulus": np.abs(roots)})
        stable = bool(np.all(np.abs(roots) < 1)) if len(roots) else None
        params = pd.DataFrame({col: fitted.params})
        stderr = pd.DataFrame({col: fitted.bse})
        pvalues = pd.DataFrame({col: fitted.pvalues})
        tvalues = pd.DataFrame({col: fitted.tvalues})
        whiteness_p_value = None
        normality_p_value = None
        warning = "Single-endogenous VARX is estimated as an ARX/SARIMAX-style conditional equation; IRF/FEVD are not available."
        fit_model_name = "ARX/SARIMAX-style VARX"
        fit_aic = fitted.aic
        fit_bic = fitted.bic
        fit_hqic = np.nan
        effective_obs = int(fitted.nobs)
        params_per_equation = int(len(fitted.params))

    forecasts = pd.concat([forecasts, lower_df, upper_df], axis=1)
    metrics = []
    for col in endog:
        error = test_endog[col] - forecasts[col]
        metrics.append(
            {
                "variable": col,
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "MAE": float(np.mean(np.abs(error))),
                "Acceptable if": "Lower RMSE/MAE is better",
            }
        )

    parameter_rows = []
    for param in params.index:
        for equation in params.columns:
            coef = params.loc[param, equation]
            se = stderr.loc[param, equation]
            p_value = pvalues.loc[param, equation]
            parameter_rows.append(
                {
                    "equation": equation,
                    "parameter": param,
                    "coefficient": coef,
                    "std_error": se,
                    "t_stat": tvalues.loc[param, equation],
                    "p_value": p_value,
                    "Acceptable if": "p-value < 0.05 indicates statistical significance; interpret individual VARX coefficients cautiously",
                    "lower_95": coef - 1.96 * se,
                    "upper_95": coef + 1.96 * se,
                    "significant_at_5pct": p_value < 0.05,
                }
            )

    fit_metrics = equation_fit_metrics_from_residuals(residuals, train_endog, params_per_equation)
    fit_info = pd.DataFrame(
        [
            {
                "model": fit_model_name,
                "lag_order": lag_order,
                "aic": fit_aic,
                "bic": fit_bic,
                "hqic": fit_hqic,
                "effective_observations": effective_obs,
                "parameters_per_equation": params_per_equation,
                "total_parameters": int(params_per_equation * len(endog)),
                "stable": stable,
                "Acceptable if": "lower IC values, stable roots, and reasonable parameter count are preferred",
            }
        ]
    )

    return {
        "train": train_endog,
        "test": test_endog,
        "fitted": fitted,
        "residuals": residuals,
        "forecasts": forecasts,
        "metrics": pd.DataFrame(metrics),
        "fit_metrics": fit_metrics,
        "fit_info": fit_info,
        "parameter_table": pd.DataFrame(parameter_rows),
        "stable": stable,
        "roots": roots_df,
        "whiteness_p_value": whiteness_p_value,
        "normality_p_value": normality_p_value,
        "warning": warning,
        "target": target,
    }


def varx_exogenous_scenario_response(
    fitted: object,
    endog_history: pd.DataFrame,
    exog_columns: list[str],
    shock_variable: str,
    horizon: int,
    shock_size: float = 1.0,
) -> pd.DataFrame:
    if not hasattr(fitted, "forecast") or shock_variable not in exog_columns:
        return pd.DataFrame()

    base_exog = pd.DataFrame(
        np.zeros((horizon, len(exog_columns))),
        columns=exog_columns,
    )
    shocked_exog = base_exog.copy()
    shocked_exog.loc[0, shock_variable] = shock_size

    base = fitted.forecast(endog_history.values[-fitted.k_ar :], steps=horizon, exog_future=base_exog.values)
    shocked = fitted.forecast(
        endog_history.values[-fitted.k_ar :],
        steps=horizon,
        exog_future=shocked_exog.values,
    )
    diff = shocked - base
    rows = []
    for response_idx, response in enumerate(fitted.names):
        for h in range(horizon):
            rows.append(
                {
                    "shock": shock_variable,
                    "response": response,
                    "horizon": h + 1,
                    "value": diff[h, response_idx],
                }
            )
    return pd.DataFrame(rows)
