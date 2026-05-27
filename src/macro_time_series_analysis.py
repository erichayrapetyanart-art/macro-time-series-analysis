from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fredapi import Fred
from statsmodels.tsa.api import SARIMAX, VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

START_DATE = "1995-01-01"
TEST_MONTHS = 36

SERIES = {
    "CPI": "CPIAUCSL",
    "UNRATE": "UNRATE",
    "FEDFUNDS": "FEDFUNDS",
    "INDPRO": "INDPRO",
    "M2": "M2SL",
    "UMCSENT": "UMCSENT",
}


def ensure_directories() -> None:
    for directory in [DATA_DIR, FIGURE_DIR, TABLE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_fred_client() -> Fred:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Run: export FRED_API_KEY='your_key_here'"
        )
    return Fred(api_key=api_key)


def download_fred_data() -> pd.DataFrame:
    fred = get_fred_client()
    data = pd.DataFrame()

    for name, fred_id in SERIES.items():
        data[name] = fred.get_series(fred_id, observation_start=START_DATE)

    data.index = pd.to_datetime(data.index)
    data = data.asfreq("MS")
    data = data.ffill()
    data.to_csv(DATA_DIR / "raw_fred_macro.csv", index_label="date")
    return data


def transform_data(raw: pd.DataFrame) -> pd.DataFrame:
    processed = pd.DataFrame(index=raw.index)
    processed["inflation_yoy"] = raw["CPI"].pct_change(12) * 100
    processed["unemployment_rate"] = raw["UNRATE"]
    processed["fed_funds_rate"] = raw["FEDFUNDS"]
    processed["industrial_production_growth_yoy"] = raw["INDPRO"].pct_change(12) * 100
    processed["m2_growth_yoy"] = raw["M2"].pct_change(12) * 100
    processed["consumer_sentiment_change_yoy"] = raw["UMCSENT"].diff(12)
    processed = processed.dropna()
    processed.to_csv(DATA_DIR / "processed_macro.csv", index_label="date")
    return processed


def save_line_plots(raw: pd.DataFrame, processed: pd.DataFrame) -> None:
    axes = raw.plot(
        subplots=True,
        figsize=(12, 10),
        title="Raw Monthly FRED Macroeconomic Series",
        linewidth=1.4,
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "01_raw_series.png", dpi=180)
    plt.close()

    axes = processed.plot(
        subplots=True,
        figsize=(12, 10),
        title="Transformed Monthly Macroeconomic Series",
        linewidth=1.4,
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "02_transformed_series.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 5))
    processed["inflation_yoy"].plot(label="Inflation YoY", linewidth=1.5)
    processed["inflation_yoy"].rolling(12).mean().plot(
        label="12-month rolling mean", linewidth=2
    )
    plt.title("U.S. CPI Inflation and Rolling Mean")
    plt.ylabel("Percent")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "03_inflation_rolling_mean.png", dpi=180)
    plt.close()


def save_correlation_table_and_plot(processed: pd.DataFrame) -> None:
    corr = processed.corr()
    corr.to_csv(TABLE_DIR / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)), labels=corr.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(corr.index)), labels=corr.index)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Matrix of Transformed Variables")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "04_correlation_matrix.png", dpi=180)
    plt.close()


def run_adf_tests(processed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in processed.columns:
        statistic, p_value, used_lag, n_obs, critical_values, _ = adfuller(
            processed[column].dropna(), autolag="AIC"
        )
        rows.append(
            {
                "variable": column,
                "adf_statistic": statistic,
                "p_value": p_value,
                "used_lag": used_lag,
                "n_obs": n_obs,
                "critical_value_1pct": critical_values["1%"],
                "critical_value_5pct": critical_values["5%"],
                "critical_value_10pct": critical_values["10%"],
                "stationary_at_5pct": p_value < 0.05,
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(TABLE_DIR / "adf_stationarity_tests.csv", index=False)
    return results


def run_granger_tests(processed: pd.DataFrame, max_lag: int = 6) -> pd.DataFrame:
    rows = []
    predictors = [column for column in processed.columns if column != "inflation_yoy"]

    for predictor in predictors:
        test_data = processed[["inflation_yoy", predictor]].dropna()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="verbose is deprecated", category=FutureWarning
            )
            results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
        best_lag = None
        best_p_value = np.inf

        for lag, lag_results in results.items():
            p_value = lag_results[0]["ssr_ftest"][1]
            if p_value < best_p_value:
                best_lag = lag
                best_p_value = p_value

        rows.append(
            {
                "predictor": predictor,
                "best_lag_months": best_lag,
                "best_ssr_ftest_p_value": best_p_value,
                "granger_causes_inflation_at_5pct": best_p_value < 0.05,
            }
        )

    granger = pd.DataFrame(rows)
    granger.to_csv(TABLE_DIR / "granger_causality_to_inflation.csv", index=False)
    return granger


def fit_sarimax_forecast(processed: pd.DataFrame) -> pd.DataFrame:
    target = "inflation_yoy"
    exog_columns = [
        "unemployment_rate",
        "fed_funds_rate",
        "industrial_production_growth_yoy",
        "m2_growth_yoy",
        "consumer_sentiment_change_yoy",
    ]

    train = processed.iloc[:-TEST_MONTHS]
    test = processed.iloc[-TEST_MONTHS:]

    model = SARIMAX(
        train[target],
        exog=train[exog_columns],
        order=(1, 0, 1),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.get_forecast(steps=len(test), exog=test[exog_columns])
    forecast_mean = forecast.predicted_mean
    forecast_interval = forecast.conf_int()

    evaluation = pd.DataFrame(
        {
            "actual_inflation_yoy": test[target],
            "forecast_inflation_yoy": forecast_mean,
            "lower_95": forecast_interval.iloc[:, 0],
            "upper_95": forecast_interval.iloc[:, 1],
        }
    )
    evaluation.to_csv(TABLE_DIR / "sarimax_forecast_results.csv", index_label="date")

    errors = evaluation["actual_inflation_yoy"] - evaluation["forecast_inflation_yoy"]
    metrics = pd.DataFrame(
        [
            {
                "mae": errors.abs().mean(),
                "rmse": np.sqrt(np.mean(errors**2)),
                "test_months": len(test),
            }
        ]
    )
    metrics.to_csv(TABLE_DIR / "sarimax_forecast_metrics.csv", index=False)

    plt.figure(figsize=(12, 5))
    train[target].iloc[-84:].plot(label="Training actual", linewidth=1.4)
    test[target].plot(label="Test actual", linewidth=1.8)
    forecast_mean.plot(label="SARIMAX forecast", linewidth=1.8)
    plt.fill_between(
        test.index,
        evaluation["lower_95"],
        evaluation["upper_95"],
        alpha=0.2,
        label="95% interval",
    )
    plt.title("Out-of-Sample Forecast of CPI Inflation")
    plt.ylabel("Percent")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "05_sarimax_inflation_forecast.png", dpi=180)
    plt.close()

    with open(TABLE_DIR / "sarimax_model_summary.txt", "w", encoding="utf-8") as file:
        file.write(fitted.summary().as_text())

    return metrics


def fit_var_model(processed: pd.DataFrame) -> None:
    model_data = processed[
        [
            "inflation_yoy",
            "unemployment_rate",
            "fed_funds_rate",
            "industrial_production_growth_yoy",
            "m2_growth_yoy",
        ]
    ]
    lag_order = VAR(model_data).select_order(maxlags=12)
    selected_lag = int(lag_order.aic)
    selected_lag = max(selected_lag, 1)
    fitted = VAR(model_data).fit(selected_lag)

    with open(TABLE_DIR / "var_model_summary.txt", "w", encoding="utf-8") as file:
        file.write(f"Selected AIC lag order: {selected_lag}\n\n")
        file.write(str(fitted.summary()))


def main() -> None:
    ensure_directories()
    raw = download_fred_data()
    processed = transform_data(raw)

    save_line_plots(raw, processed)
    save_correlation_table_and_plot(processed)
    adf_results = run_adf_tests(processed)
    granger_results = run_granger_tests(processed)
    metrics = fit_sarimax_forecast(processed)
    fit_var_model(processed)

    print("Project completed successfully.")
    print(f"Raw observations: {len(raw)}")
    print(f"Processed observations: {len(processed)}")
    print("\nADF stationarity results:")
    print(adf_results[["variable", "p_value", "stationary_at_5pct"]])
    print("\nGranger causality results:")
    print(granger_results)
    print("\nSARIMAX forecast metrics:")
    print(metrics)
    print(f"\nOutputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
