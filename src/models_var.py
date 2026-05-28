from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def select_var_lags(train: pd.DataFrame, max_lag: int, exog: pd.DataFrame | None = None) -> pd.DataFrame:
    lag_sel = VAR(train, exog=exog).select_order(maxlags=max_lag)
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


def fit_var_system(
    data: pd.DataFrame,
    split_date: str,
    endog: list[str],
    lag_order: int,
    target: str,
    exog: pd.DataFrame | None = None,
) -> dict:
    split_ts = pd.Timestamp(split_date)
    train = data.loc[:split_ts, endog].copy()
    test = data.loc[data.index > split_ts, endog].copy()
    train_exog = exog.loc[train.index] if exog is not None else None
    test_exog = exog.loc[test.index] if exog is not None else None

    fitted = VAR(train, exog=train_exog).fit(lag_order, trend="c")
    points, lower, upper = fitted.forecast_interval(
        train.values[-lag_order:],
        steps=len(test),
        alpha=0.05,
        exog_future=test_exog.values if test_exog is not None else None,
    )
    forecasts = pd.DataFrame(points, index=test.index, columns=endog)
    lower_df = pd.DataFrame(lower, index=test.index, columns=[f"{col}_lower_95" for col in endog])
    upper_df = pd.DataFrame(upper, index=test.index, columns=[f"{col}_upper_95" for col in endog])
    forecasts = pd.concat([forecasts, lower_df, upper_df], axis=1)

    roots = 1 / fitted.roots
    roots_df = pd.DataFrame({"real": roots.real, "imag": roots.imag, "modulus": np.abs(roots)})
    metrics = []
    for col in endog:
        if col in test:
            error = test[col] - forecasts[col]
            metrics.append(
                {
                    "variable": col,
                    "RMSE": float(np.sqrt(np.mean(error**2))),
                    "MAE": float(np.mean(np.abs(error))),
                    "Acceptable if": "Lower RMSE/MAE is better",
                }
            )

    parameter_rows = []
    for param in fitted.params.index:
        for equation in fitted.params.columns:
            coef = fitted.params.loc[param, equation]
            se = fitted.stderr.loc[param, equation]
            p_value = fitted.pvalues.loc[param, equation]
            parameter_rows.append(
                {
                    "equation": equation,
                    "parameter": param,
                    "coefficient": coef,
                    "std_error": se,
                    "t_stat": fitted.tvalues.loc[param, equation],
                    "p_value": p_value,
                    "Acceptable if": "p-value < 0.05 indicates statistical significance; interpret individual VAR coefficients cautiously",
                    "lower_95": coef - 1.96 * se,
                    "upper_95": coef + 1.96 * se,
                    "significant_at_5pct": p_value < 0.05,
                }
            )

    try:
        whiteness_p_value = float(fitted.test_whiteness(nlags=min(12, max(lag_order + 1, 2))).pvalue)
    except Exception:
        whiteness_p_value = None
    try:
        normality_p_value = float(fitted.test_normality().pvalue)
    except Exception:
        normality_p_value = None
    fit_metrics = equation_fit_metrics(fitted, train)
    fit_info = pd.DataFrame(
        [
            {
                "model": "VAR",
                "lag_order": lag_order,
                "aic": fitted.aic,
                "bic": fitted.bic,
                "hqic": fitted.hqic,
                "effective_observations": int(fitted.nobs),
                "parameters_per_equation": int(fitted.params.shape[0]),
                "total_parameters": int(fitted.params.shape[0] * len(fitted.names)),
                "stable": bool(fitted.is_stable()),
                "Acceptable if": "lower IC values, stable roots, and reasonable parameter count are preferred",
            }
        ]
    )

    return {
        "train": train,
        "test": test,
        "fitted": fitted,
        "residuals": fitted.resid,
        "forecasts": forecasts,
        "metrics": pd.DataFrame(metrics),
        "fit_metrics": fit_metrics,
        "fit_info": fit_info,
        "parameter_table": pd.DataFrame(parameter_rows),
        "stable": bool(fitted.is_stable()),
        "roots": roots_df,
        "whiteness_p_value": whiteness_p_value,
        "normality_p_value": normality_p_value,
        "warning": "",
        "target": target,
    }


def var_irf_paths(fitted: object, horizon: int, order: list[str] | None = None) -> pd.DataFrame:
    names = list(fitted.names)
    irf = fitted.irf(horizon)
    rows = []
    for response_idx, response in enumerate(names):
        for shock_idx, shock in enumerate(names):
            for h, value in enumerate(irf.orth_irfs[:, response_idx, shock_idx]):
                rows.append({"response": response, "shock": shock, "horizon": h, "value": value})
    return pd.DataFrame(rows)


def var_fevd_paths(fitted: object, horizon: int) -> pd.DataFrame:
    names = list(fitted.names)
    fevd = fitted.fevd(horizon)
    rows = []
    for response_idx, response in enumerate(names):
        for h in range(1, horizon + 1):
            for shock_idx, shock in enumerate(names):
                rows.append(
                    {
                        "response": response,
                        "horizon": h,
                        "shock": shock,
                        "variance_share": fevd.decomp[response_idx, h - 1, shock_idx],
                    }
                )
    return pd.DataFrame(rows)


def equation_fit_metrics(fitted: object, endog_data: pd.DataFrame) -> pd.DataFrame:
    residuals = fitted.resid
    actual = endog_data.loc[residuals.index, residuals.columns]
    rows = []
    for column in residuals.columns:
        sse = float(np.sum(residuals[column] ** 2))
        centered = actual[column] - actual[column].mean()
        sst = float(np.sum(centered**2))
        rows.append(
            {
                "equation": column,
                "R_squared": 1 - sse / sst if sst > 0 else np.nan,
                "residual_std_error": float(np.sqrt(sse / max(len(residuals) - fitted.params.shape[0], 1))),
                "n_effective_obs": int(fitted.nobs),
                "parameters_per_equation": int(fitted.params.shape[0]),
                "Acceptable if": "higher R-squared and lower residual standard error indicate better in-sample fit, but forecasting and diagnostics are more important",
            }
        )
    return pd.DataFrame(rows)
