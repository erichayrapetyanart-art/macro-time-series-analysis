from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR

from src.dashboard_helpers import make_lagged_features


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def directional_accuracy(actual: pd.Series, predicted: pd.Series, baseline: pd.Series) -> float:
    actual_direction = np.sign(actual.values - baseline.values)
    predicted_direction = np.sign(predicted.values - baseline.values)
    valid = actual_direction != 0
    if valid.sum() == 0:
        return np.nan
    return float((actual_direction[valid] == predicted_direction[valid]).mean())


def ml_benchmark(
    data: pd.DataFrame,
    split_date: pd.Timestamp,
    feature_cols: list[str],
    target: str,
    lag_order: int,
    min_test_obs: int,
) -> pd.DataFrame:
    x, y = make_lagged_features(data, feature_cols, target, lag_order)
    train_mask = x.index <= split_date
    test_mask = x.index > split_date
    if train_mask.sum() < 20 or test_mask.sum() < min_test_obs:
        return pd.DataFrame()
    x_train, x_test = x.loc[train_mask], x.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]
    models = {
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=3),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, max_depth=2),
    }
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = pd.Series(model.predict(x_test), index=y_test.index)
        rows.append(
            {
                "model": name,
                "target": target,
                "RMSE": rmse(y_test, pred),
                "MAE": float(mean_absolute_error(y_test, pred)),
                "Acceptable if": "Lower RMSE/MAE is better",
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE")


def combined_inflation_forecasts(
    econometric: pd.DataFrame,
    ml: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    if not econometric.empty:
        econ = econometric.copy()
        if "actual_INF" in econ:
            econ = econ.rename(columns={"actual_INF": "Actual INF"})
        frames.append(econ)
    if not ml.empty:
        ml_df = ml.copy()
        if "actual_INF" in ml_df:
            ml_df = ml_df.rename(columns={"actual_INF": "Actual INF"})
        frames.append(ml_df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


def official_var_varx_forecasts(
    model_df: pd.DataFrame,
    dummies: pd.DataFrame,
    var_columns: list[str],
    varx_endog: list[str],
    varx_exog: list[str],
    var_lag_order: int = 5,
    varx_lag_order: int = 4,
    test_months: int = 36,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = model_df.iloc[:-test_months]
    test = model_df.iloc[-test_months:]
    train_full = pd.concat([train, dummies.loc[train.index]], axis=1)
    test_full = pd.concat([test, dummies.loc[test.index]], axis=1)

    forecast_rows = []
    metric_rows = []

    var = VAR(train[var_columns]).fit(var_lag_order, trend="c")
    var_forecast_values = var.forecast(
        train[var_columns].values[-var_lag_order:],
        steps=len(test),
    )
    var_forecast = pd.DataFrame(var_forecast_values, index=test.index, columns=var_columns)

    train_varx_exog = train_full[varx_exog] if varx_exog else None
    test_varx_exog = test_full[varx_exog] if varx_exog else None
    varx = VAR(train[varx_endog], exog=train_varx_exog).fit(varx_lag_order, trend="c")
    varx_forecast_values = varx.forecast(
        train[varx_endog].values[-varx_lag_order:],
        steps=len(test),
        exog_future=test_varx_exog.values if varx_exog else None,
    )
    varx_forecast = pd.DataFrame(varx_forecast_values, index=test.index, columns=varx_endog)

    model_forecasts = {"VAR": var_forecast, "VARX": varx_forecast}
    for model_name, forecast in model_forecasts.items():
        for variable in forecast.columns:
            actual = test[variable]
            predicted = forecast[variable]
            previous_actual = actual.shift(1)
            previous_actual.iloc[0] = train[variable].iloc[-1]
            metric_rows.append(
                {
                    "model": model_name,
                    "variable": variable,
                    "RMSE": rmse(actual, predicted),
                    "MAE": float(mean_absolute_error(actual, predicted)),
                    "directional_accuracy": directional_accuracy(actual, predicted, previous_actual),
                    "Acceptable if": "Lower RMSE/MAE is better; higher directional accuracy is better",
                }
            )
            for date, actual_value, forecast_value in zip(test.index, actual, predicted):
                forecast_rows.append(
                    {
                        "date": date,
                        "model": model_name,
                        "variable": variable,
                        "actual": actual_value,
                        "forecast": forecast_value,
                    }
                )

    forecasts = pd.DataFrame(forecast_rows)
    metrics = pd.DataFrame(metric_rows).sort_values(["variable", "RMSE"]).reset_index(drop=True)
    return forecasts, metrics
