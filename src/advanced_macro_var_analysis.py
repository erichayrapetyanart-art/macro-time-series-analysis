from __future__ import annotations

import itertools
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
from pandas.plotting import scatter_matrix
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

RAW_DATA_PATH = DATA_DIR / "raw_fred_macro.csv"
TEST_MONTHS = 36
ROLLING_TEST_START = "2024-01-01"
ROLLING_HORIZON = 3
MAX_VAR_LAGS = 12
VAR_COLUMNS = [
    "FEDFUNDS",
    "INF",
    "UNRATE",
    "INDPRO_GROWTH",
    "M2_GROWTH",
    "SENTIMENT_CHANGE",
]

VARX_ENDOG = ["INF", "UNRATE", "INDPRO_GROWTH", "M2_GROWTH"]
VARX_EXOG = ["FEDFUNDS", "SENTIMENT_CHANGE", "D_2008", "D_COVID"]


def rmse(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    return mean_squared_error(actual, predicted) ** 0.5


def ensure_directories() -> None:
    for directory in [DATA_DIR, FIGURE_DIR, TABLE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Remove generated files from the earlier single-equation version so the
    # current VAR/VARX project does not show stale outputs in the notebook/dashboard.
    for stale_path in [
        TABLE_DIR / "academic_ardl_model_summary.txt",
        TABLE_DIR / "academic_ardl_residual_diagnostics.csv",
        TABLE_DIR / "academic_ardl_coefficients.csv",
        TABLE_DIR / "academic_ardl_robust_coefficients.csv",
        FIGURE_DIR / "academic_ardl_residual_acf.png",
        FIGURE_DIR / "academic_ardl_residual_plot.png",
    ]:
        if stale_path.exists():
            stale_path.unlink()


def load_raw_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_DATA_PATH} does not exist. Run src/macro_time_series_analysis.py first."
        )

    raw = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"], index_col="date")
    raw = raw.asfreq("MS").ffill()
    return raw


def save_variable_dictionary() -> pd.DataFrame:
    rows = [
        {
            "variable": "CPI",
            "fred_id": "CPIAUCSL",
            "economic_role": "Price level used to construct inflation.",
            "expected_use": "Target transformation",
        },
        {
            "variable": "UNRATE",
            "fred_id": "UNRATE",
            "economic_role": "Labor market slack and cyclical pressure.",
            "expected_use": "Endogenous macro state",
        },
        {
            "variable": "FEDFUNDS",
            "fred_id": "FEDFUNDS",
            "economic_role": "Monetary-policy stance and short-term interest rate.",
            "expected_use": "Endogenous in VAR, exogenous in VARX",
        },
        {
            "variable": "INDPRO",
            "fred_id": "INDPRO",
            "economic_role": "Real production and business-cycle activity.",
            "expected_use": "Growth transformation",
        },
        {
            "variable": "M2",
            "fred_id": "M2SL",
            "economic_role": "Broad money supply and liquidity conditions.",
            "expected_use": "Growth transformation",
        },
        {
            "variable": "UMCSENT",
            "fred_id": "UMCSENT",
            "economic_role": "Household expectations and confidence channel.",
            "expected_use": "Change transformation",
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "academic_variable_dictionary.csv", index=False)
    return table


def adf_table(data: pd.DataFrame, output_name: str) -> pd.DataFrame:
    rows = []
    for column in data.columns:
        result = adfuller(data[column].dropna(), autolag="AIC")
        rows.append(
            {
                "variable": column,
                "adf_statistic": result[0],
                "p_value": result[1],
                "used_lag": result[2],
                "n_obs": result[3],
                "stationary_at_5pct": result[1] < 0.05,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / output_name, index=False)
    return table


def run_pairwise_cointegration(raw: pd.DataFrame, raw_adf: pd.DataFrame) -> pd.DataFrame:
    candidates = raw_adf.loc[~raw_adf["stationary_at_5pct"], "variable"].tolist()
    transformed = {}

    for column in candidates:
        series = raw[column].dropna()
        if (series > 0).all() and column in {"CPI", "INDPRO", "M2"}:
            transformed[f"log_{column}"] = np.log(series)
        else:
            transformed[column] = series

    rows = []
    for left, right in itertools.combinations(transformed, 2):
        pair = pd.concat([transformed[left], transformed[right]], axis=1).dropna()
        y = pair.iloc[:, 0]
        x = np.column_stack([np.ones(len(pair)), pair.iloc[:, 1]])
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        residuals = y - x @ beta
        adf_result = adfuller(residuals, autolag="AIC")
        rows.append(
            {
                "left_variable": left,
                "right_variable": right,
                "residual_adf_p_value": adf_result[1],
                "cointegrated_at_5pct": adf_result[1] < 0.05,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "academic_pairwise_cointegration.csv", index=False)
    return table


def transform_for_models(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    df["FEDFUNDS"] = raw["FEDFUNDS"]
    df["INF"] = np.log(raw["CPI"]).diff() * 100
    df["UNRATE"] = raw["UNRATE"]
    df["INDPRO_GROWTH"] = np.log(raw["INDPRO"]).diff() * 100
    df["M2_GROWTH"] = np.log(raw["M2"]).diff() * 100
    df["SENTIMENT_CHANGE"] = raw["UMCSENT"].diff()
    df = df.dropna()
    df.to_csv(DATA_DIR / "academic_model_data.csv", index_label="date")
    return df


def make_break_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    dummies = pd.DataFrame(index=index)
    dummies["D_2008"] = (index >= "2008-09-01").astype(int)
    dummies["D_COVID"] = (index >= "2020-03-01").astype(int)
    dummies.to_csv(DATA_DIR / "academic_break_dummies.csv", index_label="date")
    return dummies


def save_eda_tables(raw: pd.DataFrame, model_df: pd.DataFrame) -> None:
    raw.describe().T.to_csv(TABLE_DIR / "academic_raw_summary_statistics.csv")
    model_df.describe().T.to_csv(TABLE_DIR / "academic_model_summary_statistics.csv")
    raw.isna().sum().rename("missing_values").to_csv(TABLE_DIR / "academic_missing_values.csv")
    model_df.corr().to_csv(TABLE_DIR / "academic_model_correlation_matrix.csv")


def save_eda_figures(raw: pd.DataFrame, model_df: pd.DataFrame) -> None:
    raw.plot(subplots=True, figsize=(13, 11), linewidth=1.2, title="Raw FRED Series")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "academic_01_raw_series.png", dpi=180)
    plt.close()

    model_df.plot(
        subplots=True,
        figsize=(13, 11),
        linewidth=1.2,
        title="Stationary Model Variables After Transformation",
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "academic_02_transformed_series.png", dpi=180)
    plt.close()

    rolling = model_df[["INF", "UNRATE", "FEDFUNDS"]].rolling(12).mean()
    ax = rolling.plot(figsize=(12, 5), linewidth=1.5)
    ax.set_title("Twelve-Month Rolling Means: Inflation, Unemployment, Policy Rate")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "academic_03_rolling_means.png", dpi=180)
    plt.close()

    corr = model_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Matrix of Model Variables")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "academic_04_correlation_matrix.png", dpi=180)
    plt.close()

    scatter_matrix(model_df, figsize=(12, 12), diagonal="kde", alpha=0.45)
    plt.suptitle("Scatter Matrix of Transformed Model Variables", y=1.0)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "academic_05_scatter_matrix.png", dpi=180)
    plt.close()

    axes = model_df.hist(figsize=(12, 9), bins=30)
    for row in axes:
        for axis in row:
            axis.grid(alpha=0.2)
    plt.suptitle("Distribution of Stationary Model Variables")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "academic_06_distributions.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plot_acf(model_df["INF"], lags=36, ax=axes[0])
    plot_pacf(model_df["INF"], lags=36, ax=axes[1], method="ywm")
    axes[0].set_title("ACF of Monthly Inflation")
    axes[1].set_title("PACF of Monthly Inflation")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "academic_07_inflation_acf_pacf.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(model_df.index, model_df["INF"], label="Inflation")
    ax.axvspan(pd.Timestamp("2008-09-01"), pd.Timestamp("2009-06-01"), alpha=0.2, color="red", label="2008 crisis")
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), alpha=0.2, color="purple", label="COVID shock")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Structural Break Inspection: Inflation Around Crisis Periods")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "academic_08_structural_break_inspection.png", dpi=180)
    plt.close()


def split_train_test(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return model_df.iloc[:-TEST_MONTHS], model_df.iloc[-TEST_MONTHS:]


def select_and_fit_var(model_df: pd.DataFrame, dummies: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    lag_selection = VAR(model_df, exog=dummies).select_order(maxlags=MAX_VAR_LAGS)
    lag_table = pd.DataFrame(
        {
            "AIC": lag_selection.ics["aic"],
            "BIC": lag_selection.ics["bic"],
            "HQIC": lag_selection.ics["hqic"],
            "FPE": lag_selection.ics["fpe"],
        }
    )
    lag_table.index.name = "lag"
    lag_table.to_csv(TABLE_DIR / "academic_var_lag_selection.csv")

    selected_lag = int(lag_selection.aic)
    selected_lag = max(selected_lag, 1)
    results = VAR(model_df, exog=dummies).fit(selected_lag, trend="c")

    with open(TABLE_DIR / "academic_var_model_summary.txt", "w", encoding="utf-8") as file:
        file.write(f"Selected AIC lag order: {selected_lag}\n\n")
        file.write(str(results.summary()))

    return results, lag_table


def fit_varx(model_df: pd.DataFrame, dummies: pd.DataFrame, p_lags: int) -> object:
    endog = model_df[VARX_ENDOG]
    exog = pd.concat([model_df[["FEDFUNDS", "SENTIMENT_CHANGE"]], dummies], axis=1)
    results = VAR(endog, exog=exog).fit(p_lags, trend="c")

    with open(TABLE_DIR / "academic_varx_model_summary.txt", "w", encoding="utf-8") as file:
        file.write("VARX specification\n")
        file.write(f"Endogenous variables: {VARX_ENDOG}\n")
        file.write(f"Exogenous variables: {VARX_EXOG}\n\n")
        file.write(str(results.summary()))

    return results


def select_varx_lag_order(model_df: pd.DataFrame, dummies: pd.DataFrame) -> pd.DataFrame:
    endog = model_df[VARX_ENDOG]
    exog = pd.concat([model_df[["FEDFUNDS", "SENTIMENT_CHANGE"]], dummies], axis=1)
    lag_selection = VAR(endog, exog=exog).select_order(maxlags=MAX_VAR_LAGS)
    lag_table = pd.DataFrame(
        {
            "AIC": lag_selection.ics["aic"],
            "BIC": lag_selection.ics["bic"],
            "HQIC": lag_selection.ics["hqic"],
            "FPE": lag_selection.ics["fpe"],
        }
    )
    lag_table.index.name = "lag"
    lag_table.to_csv(TABLE_DIR / "academic_varx_lag_selection.csv")
    return lag_table


def save_model_architecture(p_lags: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    architecture_rows = [
        {
            "component": "Primary reduced-form model",
            "model": "VAR",
            "endogenous_variables": ", ".join(VAR_COLUMNS),
            "exogenous_variables": "D_2008, D_COVID",
            "lag_order": p_lags,
            "selection_or_role": "Lag order selected by minimum AIC over candidate lags 0-12; BIC/HQIC and residual robustness are reported as checks.",
            "purpose": "Main system model for dynamic interactions, Granger causality, residual diagnostics, IRF, and FEVD.",
        },
        {
            "component": "Conditional forecasting model",
            "model": "VARX",
            "endogenous_variables": ", ".join(VARX_ENDOG),
            "exogenous_variables": ", ".join(VARX_EXOG),
            "lag_order": p_lags,
            "selection_or_role": "Uses the same lag order as the baseline VAR for comparability.",
            "purpose": "Conditional forecast model where policy/sentiment/crisis controls are treated as externally conditioned paths.",
        },
        {
            "component": "Structural interpretation layer",
            "model": "Recursive SVAR via Cholesky decomposition of VAR residual covariance",
            "endogenous_variables": ", ".join(VAR_COLUMNS),
            "exogenous_variables": "D_2008, D_COVID",
            "lag_order": p_lags,
            "selection_or_role": "Uses the reduced-form VAR ordering for Cholesky identification.",
            "purpose": "Impulse response and FEVD analysis under explicit contemporaneous recursive restrictions.",
        },
    ]
    architecture = pd.DataFrame(architecture_rows)
    architecture.to_csv(TABLE_DIR / "academic_final_model_architecture.csv", index=False)

    ordering_rows = [
        {
            "order": 1,
            "variable": "FEDFUNDS",
            "contemporaneous_assumption": "Policy rate can contemporaneously affect all later variables in the recursive system.",
            "economic_motivation": "Monetary policy is a fast-moving financial/policy variable and Granger results show strong predictive content for inflation, unemployment, production, and money growth.",
        },
        {
            "order": 2,
            "variable": "INF",
            "contemporaneous_assumption": "Inflation reacts within the month to policy shocks but contemporaneously affects variables ordered after it.",
            "economic_motivation": "Prices are central to the policy objective; inflation shocks can affect real activity, money demand, and confidence with short delays.",
        },
        {
            "order": 3,
            "variable": "UNRATE",
            "contemporaneous_assumption": "Labor-market conditions react contemporaneously to policy and inflation shocks, while affecting slower real and monetary aggregates in the same period.",
            "economic_motivation": "Unemployment adjusts more slowly than financial variables but is a key state variable in Phillips-curve and business-cycle dynamics.",
        },
        {
            "order": 4,
            "variable": "INDPRO_GROWTH",
            "contemporaneous_assumption": "Industrial production growth can react to policy, price, and labor-market shocks within the period.",
            "economic_motivation": "Production is a real-activity measure that responds to demand, financing conditions, and labor-market slack.",
        },
        {
            "order": 5,
            "variable": "M2_GROWTH",
            "contemporaneous_assumption": "Money growth reacts contemporaneously to the variables ordered before it but does not contemporaneously move them under this identification.",
            "economic_motivation": "Broad money adjusts through banking, portfolio, and liquidity channels and is placed after policy and real-activity indicators.",
        },
        {
            "order": 6,
            "variable": "SENTIMENT_CHANGE",
            "contemporaneous_assumption": "Sentiment is allowed to react contemporaneously to all macro shocks ordered before it.",
            "economic_motivation": "Survey sentiment is fast-moving but often reflects news about policy, prices, labor markets, output, and liquidity; placing it last is conservative for macro shock interpretation.",
        },
    ]
    ordering = pd.DataFrame(ordering_rows)
    ordering.to_csv(TABLE_DIR / "academic_cholesky_ordering.csv", index=False)
    return architecture, ordering


def save_parameter_significance(results: object, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for parameter in results.params.index:
        for equation in results.params.columns:
            coefficient = results.params.loc[parameter, equation]
            std_error = results.stderr.loc[parameter, equation]
            rows.append(
                {
                    "equation": equation,
                    "parameter": parameter,
                    "coefficient": coefficient,
                    "std_error": std_error,
                    "t_stat": results.tvalues.loc[parameter, equation],
                    "p_value": results.pvalues.loc[parameter, equation],
                    "lower_95": coefficient - 1.96 * std_error,
                    "upper_95": coefficient + 1.96 * std_error,
                    "significant_at_5pct": results.pvalues.loc[parameter, equation] < 0.05,
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / f"{prefix}_parameter_significance.csv", index=False)

    summary = (
        table.groupby("equation")
        .agg(
            total_parameters=("parameter", "count"),
            significant_at_5pct=("significant_at_5pct", "sum"),
            min_p_value=("p_value", "min"),
        )
        .reset_index()
    )
    summary["share_significant_at_5pct"] = (
        summary["significant_at_5pct"] / summary["total_parameters"]
    )
    summary.to_csv(TABLE_DIR / f"{prefix}_parameter_significance_summary.csv", index=False)
    return table, summary


def residual_autocorrelation_visuals(
    residuals: pd.DataFrame, prefix: str, max_lag: int = 24
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_vars = len(residuals.columns)
    n_cols = 2
    n_rows = int(np.ceil(n_vars / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    confidence_bound = 1.96 / np.sqrt(len(residuals))
    summary_rows = []

    for idx, column in enumerate(residuals.columns):
        plot_acf(residuals[column], lags=max_lag, alpha=0.05, ax=axes[idx], zero=False)
        axes[idx].axhline(confidence_bound, color="red", linestyle="--", linewidth=1)
        axes[idx].axhline(-confidence_bound, color="red", linestyle="--", linewidth=1)
        axes[idx].set_title(f"Residual ACF: {column}")
        acf_values = [residuals[column].autocorr(lag=lag) for lag in range(1, max_lag + 1)]
        outside = [value for value in acf_values if abs(value) > confidence_bound]
        summary_rows.append(
            {
                "equation": column,
                "max_abs_acf_lag_1_to_24": np.nanmax(np.abs(acf_values)),
                "confidence_bound_approx": confidence_bound,
                "num_lags_outside_bound": len(outside),
                "any_lag_outside_bound": len(outside) > 0,
            }
        )

    for axis in axes[n_vars:]:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{prefix}_residual_acf.png", dpi=180)
    plt.close()

    ccf_rows = []
    columns = list(residuals.columns)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(2.8 * n_vars, 2.4 * n_vars))
    lags = np.arange(1, max_lag + 1)

    for row_idx, target in enumerate(columns):
        for col_idx, source in enumerate(columns):
            axis = axes[row_idx, col_idx]
            if source == target:
                values = [residuals[target].autocorr(lag=lag) for lag in lags]
                axis.set_title(f"ACF: {target}", fontsize=8)
            else:
                values = [
                    np.corrcoef(
                        residuals[source].iloc[:-lag],
                        residuals[target].iloc[lag:],
                    )[0, 1]
                    for lag in lags
                ]
                axis.set_title(f"{source} leads {target}", fontsize=8)
                outside = [value for value in values if abs(value) > confidence_bound]
                ccf_rows.append(
                    {
                        "source_residual": source,
                        "target_residual": target,
                        "max_abs_ccf_lag_1_to_24": np.nanmax(np.abs(values)),
                        "confidence_bound_approx": confidence_bound,
                        "num_lags_outside_bound": len(outside),
                        "any_lag_outside_bound": len(outside) > 0,
                    }
                )

            axis.bar(lags, values, color="#4C78A8")
            axis.axhline(0, color="black", linewidth=0.8)
            axis.axhline(confidence_bound, color="red", linestyle="--", linewidth=0.8)
            axis.axhline(-confidence_bound, color="red", linestyle="--", linewidth=0.8)
            axis.set_ylim(-0.45, 0.45)
            axis.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{prefix}_residual_acf_ccf_matrix.png", dpi=180)
    plt.close()

    acf_summary = pd.DataFrame(summary_rows)
    ccf_summary = pd.DataFrame(ccf_rows)
    acf_summary.to_csv(TABLE_DIR / f"{prefix}_residual_acf_summary.csv", index=False)
    ccf_summary.to_csv(TABLE_DIR / f"{prefix}_residual_ccf_summary.csv", index=False)
    return acf_summary, ccf_summary


def residual_arch_tests(residuals: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    for column in residuals.columns:
        lm_stat, lm_p_value, f_stat, f_p_value = het_arch(residuals[column].dropna(), nlags=12)
        rows.append(
            {
                "equation": column,
                "arch_lm_stat_lag12": lm_stat,
                "arch_lm_p_value_lag12": lm_p_value,
                "arch_f_stat_lag12": f_stat,
                "arch_f_p_value_lag12": f_p_value,
                "arch_effects_at_5pct": lm_p_value < 0.05,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / f"{prefix}_arch_tests.csv", index=False)
    return table


def lag_residual_autocorrelation_robustness(
    model_df: pd.DataFrame, dummies: pd.DataFrame, max_lag: int = 8
) -> pd.DataFrame:
    rows = []
    for p_lag in range(1, max_lag + 1):
        fitted = VAR(model_df, exog=dummies).fit(p_lag, trend="c")
        residuals = fitted.resid
        confidence_bound = 1.96 / np.sqrt(len(residuals))
        exceedances = 0
        for column in residuals.columns:
            acf_values = [residuals[column].autocorr(lag=lag) for lag in range(1, 25)]
            exceedances += sum(abs(value) > confidence_bound for value in acf_values)
        try:
            whiteness_p_value = fitted.test_whiteness(nlags=12).pvalue
        except Exception:
            whiteness_p_value = np.nan
        rows.append(
            {
                "lag_order": p_lag,
                "aic": fitted.aic,
                "bic": fitted.bic,
                "total_acf_lag_exceedances": exceedances,
                "stable_system": fitted.is_stable(),
                "multivariate_whiteness_p_value": whiteness_p_value,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "academic_var_lag_residual_autocorr_robustness.csv", index=False)
    return table


def residual_diagnostics(results: object, prefix: str) -> pd.DataFrame:
    residuals = results.resid
    dw_values = durbin_watson(residuals)
    rows = []

    for idx, column in enumerate(residuals.columns):
        lb = acorr_ljungbox(residuals[column], lags=[12], return_df=True)
        rows.append(
            {
                "variable": column,
                "durbin_watson": dw_values[idx],
                "ljung_box_p_value_lag12": lb["lb_pvalue"].iloc[0],
                "white_noise_at_5pct": lb["lb_pvalue"].iloc[0] > 0.05,
            }
        )

    diagnostics = pd.DataFrame(rows)
    arch = residual_arch_tests(residuals, prefix)
    diagnostics = diagnostics.merge(
        arch[["equation", "arch_lm_p_value_lag12", "arch_effects_at_5pct"]],
        left_on="variable",
        right_on="equation",
        how="left",
    ).drop(columns=["equation"])
    diagnostics["stable_system"] = results.is_stable()
    try:
        diagnostics["multivariate_whiteness_p_value"] = results.test_whiteness(nlags=12).pvalue
    except Exception:
        diagnostics["multivariate_whiteness_p_value"] = np.nan
    diagnostics.to_csv(TABLE_DIR / f"{prefix}_residual_diagnostics.csv", index=False)
    residual_autocorrelation_visuals(residuals, prefix)

    residuals.plot(subplots=True, figsize=(12, 10), linewidth=1.1, title=f"{prefix.upper()} Residuals")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{prefix}_residual_plots.png", dpi=180)
    plt.close()

    corr = residuals.corr()
    corr.to_csv(TABLE_DIR / f"{prefix}_residual_correlation_matrix.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{prefix.upper()} Residual Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{prefix}_residual_correlation_heatmap.png", dpi=180)
    plt.close()

    eigenvalues = 1 / results.roots
    fig, ax = plt.subplots(figsize=(7, 7))
    circle = plt.Circle((0, 0), 1, fill=False, linestyle="--", color="black")
    ax.add_artist(circle)
    ax.scatter(eigenvalues.real, eigenvalues.imag, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_title(f"{prefix.upper()} Stability: Inverse Roots")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{prefix}_stability_roots.png", dpi=180)
    plt.close()

    return diagnostics


def granger_map(model_df: pd.DataFrame, maxlag: int) -> pd.DataFrame:
    rows = []
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="verbose is deprecated", category=FutureWarning
        )
        for target in model_df.columns:
            for source in model_df.columns:
                if source == target:
                    continue
                result = grangercausalitytests(
                    model_df[[target, source]], maxlag=maxlag, verbose=False
                )
                p_value = result[maxlag][0]["ssr_ftest"][1]
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "lag": maxlag,
                        "p_value": p_value,
                        "significant_at_5pct": p_value < 0.05,
                    }
                )

    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "academic_granger_causality_map.csv", index=False)

    pivot = table.pivot(index="source", columns="target", values="p_value")
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(pivot, cmap="viridis_r", vmin=0, vmax=0.1)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="p-value")
    ax.set_title(f"Pairwise Granger Causality P-Values at Lag {maxlag}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "academic_09_granger_causality_heatmap.png", dpi=180)
    plt.close()

    return table


def save_irf_fevd(results: object, horizon: int = 24) -> pd.DataFrame:
    irf = results.irf(horizon)
    irf.plot(orth=True, signif=0.05, figsize=(15, 13))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "academic_10_structural_irf.png", dpi=180)
    plt.close()

    responses = irf.orth_irfs
    irf_path_rows = []
    irf_rows = []
    for response_idx, response in enumerate(results.names):
        for shock_idx, shock in enumerate(results.names):
            path = responses[:, response_idx, shock_idx]
            for h, value in enumerate(path):
                irf_path_rows.append(
                    {
                        "response": response,
                        "shock": shock,
                        "horizon": h,
                        "orthogonalized_response": value,
                    }
                )
            peak_horizon = int(np.nanargmax(np.abs(path)))
            irf_rows.append(
                {
                    "response": response,
                    "shock": shock,
                    "impact_h0": path[0],
                    "response_h3": path[3],
                    "response_h6": path[6],
                    "response_h12": path[12],
                    "response_h24": path[24],
                    "cumulative_response_h0_to_h24": path.sum(),
                    "peak_abs_response": path[peak_horizon],
                    "peak_abs_response_horizon": peak_horizon,
                    "persistence_ratio_h12_to_impact": (
                        path[12] / path[0] if abs(path[0]) > 1e-12 else np.nan
                    ),
                }
            )

    irf_paths = pd.DataFrame(irf_path_rows)
    irf_paths.to_csv(TABLE_DIR / "academic_irf_paths.csv", index=False)
    irf_table = pd.DataFrame(irf_rows)
    irf_table.to_csv(TABLE_DIR / "academic_irf_interpretation_table.csv", index=False)
    irf_table.loc[irf_table["shock"] == "FEDFUNDS"].to_csv(
        TABLE_DIR / "academic_irf_monetary_policy_shock.csv", index=False
    )

    fevd = results.fevd(horizon)
    rows = []
    full_rows = []
    for response_idx, response in enumerate(results.names):
        for h in range(1, horizon + 1):
            for shock_idx, shock in enumerate(results.names):
                full_rows.append(
                    {
                        "response": response,
                        "horizon": h,
                        "shock": shock,
                        "variance_share": fevd.decomp[response_idx, h - 1, shock_idx],
                    }
                )
        for h in [1, 3, 6, 12, 24]:
            for shock_idx, shock in enumerate(results.names):
                rows.append(
                    {
                        "response": response,
                        "horizon": h,
                        "shock": shock,
                        "variance_share": fevd.decomp[response_idx, h - 1, shock_idx],
                    }
                )

    pd.DataFrame(full_rows).to_csv(TABLE_DIR / "academic_fevd_full.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "academic_fevd_selected_horizons.csv", index=False)
    dominant = (
        table.sort_values("variance_share", ascending=False)
        .groupby(["response", "horizon"])
        .head(1)
        .sort_values(["response", "horizon"])
        .reset_index(drop=True)
    )
    dominant.to_csv(TABLE_DIR / "academic_fevd_dominant_shocks.csv", index=False)

    fig, axes = plt.subplots(len(results.names), 1, figsize=(13, 14), sharex=True)
    horizons = np.arange(1, horizon + 1)
    for response_idx, response in enumerate(results.names):
        bottom = np.zeros(horizon)
        for shock_idx, shock in enumerate(results.names):
            values = fevd.decomp[response_idx, :, shock_idx]
            axes[response_idx].bar(horizons, values, bottom=bottom, label=shock)
            bottom += values
        axes[response_idx].set_title(response)
        axes[response_idx].set_ylim(0, 1)
        axes[response_idx].grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Horizon in months")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "academic_11_fevd.png", dpi=180)
    plt.close()

    return table


def econometric_forecasts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dummies: pd.DataFrame,
    p_lags: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dummies = dummies.loc[train.index]
    test_dummies = dummies.loc[test.index]

    var = VAR(train, exog=train_dummies).fit(p_lags, trend="c")
    var_forecast = var.forecast(train.values[-p_lags:], steps=len(test), exog_future=test_dummies.values)
    var_forecast = pd.DataFrame(var_forecast, index=test.index, columns=train.columns)["INF"]

    train_varx_exog = pd.concat([train[["FEDFUNDS", "SENTIMENT_CHANGE"]], train_dummies], axis=1)
    test_varx_exog = pd.concat([test[["FEDFUNDS", "SENTIMENT_CHANGE"]], test_dummies], axis=1)
    varx = VAR(train[VARX_ENDOG], exog=train_varx_exog).fit(p_lags, trend="c")
    varx_forecast = varx.forecast(
        train[VARX_ENDOG].values[-p_lags:], steps=len(test), exog_future=test_varx_exog.values
    )
    varx_forecast = pd.DataFrame(varx_forecast, index=test.index, columns=VARX_ENDOG)["INF"]

    random_walk = pd.Series(train["INF"].iloc[-1], index=test.index, name="Random Walk")

    forecasts = pd.DataFrame(
        {
            "actual_INF": test["INF"],
            "VAR": var_forecast,
            "VARX": varx_forecast,
            "Random Walk": random_walk,
        }
    )
    forecasts.to_csv(TABLE_DIR / "academic_econometric_inflation_forecasts.csv", index_label="date")

    rows = []
    for model in ["VAR", "VARX", "Random Walk"]:
        rows.append(
            {
                "model": "Random Walk (direct 36-month)" if model == "Random Walk" else model,
                "target": "INF",
                "forecast_design": "final_36_months",
                "rmse": rmse(forecasts["actual_INF"], forecasts[model]),
                "mae": mean_absolute_error(forecasts["actual_INF"], forecasts[model]),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLE_DIR / "academic_econometric_forecast_metrics.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(forecasts.index, forecasts["actual_INF"], label="Actual INF", color="black", linewidth=1.8)
    for model in ["VAR", "VARX", "Random Walk"]:
        plt.plot(forecasts.index, forecasts[model], label=model, linestyle="--")
    plt.title("Out-of-Sample Inflation Forecasts: Econometric Models")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "academic_12_econometric_forecast_comparison.png", dpi=180)
    plt.close()

    return forecasts, metrics


def rolling_varx_forecasts(model_df: pd.DataFrame, dummies: pd.DataFrame, p_lags: int) -> pd.DataFrame:
    rows = []
    origins = [
        model_df.index.get_loc(date)
        for date in model_df.loc[model_df.index >= ROLLING_TEST_START].index
        if model_df.index.get_loc(date) + ROLLING_HORIZON < len(model_df)
    ]

    for origin_pos in origins:
        history = model_df.iloc[: origin_pos + 1]
        future = model_df.iloc[origin_pos + 1 : origin_pos + 1 + ROLLING_HORIZON]
        h_exog = pd.concat(
            [history[["FEDFUNDS", "SENTIMENT_CHANGE"]], dummies.loc[history.index]], axis=1
        )
        f_exog = pd.concat(
            [future[["FEDFUNDS", "SENTIMENT_CHANGE"]], dummies.loc[future.index]], axis=1
        )
        fitted = VAR(history[VARX_ENDOG], exog=h_exog).fit(p_lags, trend="c")
        forecast_values = fitted.forecast(
            history[VARX_ENDOG].values[-p_lags:], steps=ROLLING_HORIZON, exog_future=f_exog.values
        )
        forecast = pd.DataFrame(forecast_values, index=future.index, columns=VARX_ENDOG)
        random_walk = pd.DataFrame(
            np.tile(history[VARX_ENDOG].iloc[-1].values, (ROLLING_HORIZON, 1)),
            index=future.index,
            columns=VARX_ENDOG,
        )

        for column in VARX_ENDOG:
            rows.append(
                {
                    "origin": model_df.index[origin_pos],
                    "variable": column,
                    "varx_rmse": rmse(future[column], forecast[column]),
                    "random_walk_rmse": rmse(future[column], random_walk[column]),
                }
            )

    detail = pd.DataFrame(rows)
    aggregate = detail.groupby("variable")[["varx_rmse", "random_walk_rmse"]].mean()
    aggregate["winner"] = np.where(
        aggregate["varx_rmse"] < aggregate["random_walk_rmse"], "VARX", "Random Walk"
    )
    aggregate.to_csv(TABLE_DIR / "academic_rolling_3month_varx_rmse.csv")

    ax = aggregate[["varx_rmse", "random_walk_rmse"]].plot(kind="bar", figsize=(10, 5))
    ax.set_title("Average Rolling 3-Month Forecast RMSE: VARX vs Random Walk")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "academic_13_rolling_varx_rmse.png", dpi=180)
    plt.close()

    return aggregate


def make_lagged_supervised(model_df: pd.DataFrame, lags: int, target: str) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    index = []
    for i in range(lags, len(model_df)):
        features = {}
        for lag in range(1, lags + 1):
            for column in model_df.columns:
                features[f"{column}_lag{lag}"] = model_df[column].iloc[i - lag]
        rows.append(features)
        index.append(model_df.index[i])
    x = pd.DataFrame(rows, index=index)
    y = model_df[target].iloc[lags:]
    y.index = x.index
    return x, y


def machine_learning_benchmarks(model_df: pd.DataFrame, p_lags: int) -> pd.DataFrame:
    x, y = make_lagged_supervised(model_df, p_lags, "INF")
    x_train, x_test = x.iloc[:-TEST_MONTHS], x.iloc[-TEST_MONTHS:]
    y_train, y_test = y.iloc[:-TEST_MONTHS], y.iloc[-TEST_MONTHS:]

    models = {
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=400, random_state=42, min_samples_leaf=3
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, max_depth=2),
    }

    predictions = pd.DataFrame({"actual_INF": y_test})
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = pd.Series(model.predict(x_test), index=y_test.index)
        predictions[name] = pred
        rows.append(
            {
                "model": name,
                "target": "INF",
                "forecast_design": "one_step_lagged_features_final_36_months",
                "rmse": rmse(y_test, pred),
                "mae": mean_absolute_error(y_test, pred),
            }
        )

    random_walk = model_df["INF"].shift(1).loc[y_test.index]
    predictions["Random Walk"] = random_walk
    rows.append(
        {
            "model": "Random Walk (one-step)",
            "target": "INF",
            "forecast_design": "one_step_lagged_features_final_36_months",
            "rmse": rmse(y_test, random_walk),
            "mae": mean_absolute_error(y_test, random_walk),
        }
    )

    metrics = pd.DataFrame(rows).sort_values("rmse")
    metrics.to_csv(TABLE_DIR / "academic_ml_forecast_metrics.csv", index=False)
    predictions.to_csv(TABLE_DIR / "academic_ml_inflation_forecasts.csv", index_label="date")

    plt.figure(figsize=(12, 5))
    plt.plot(predictions.index, predictions["actual_INF"], label="Actual INF", color="black", linewidth=1.8)
    for column in predictions.columns.drop("actual_INF"):
        plt.plot(predictions.index, predictions[column], label=column, linestyle="--")
    plt.title("Machine Learning Inflation Forecast Comparison")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "academic_14_ml_forecast_comparison.png", dpi=180)
    plt.close()

    return metrics


def combined_model_ranking(econometric_metrics: pd.DataFrame, ml_metrics: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([econometric_metrics, ml_metrics], ignore_index=True)
    combined = combined.sort_values("rmse").reset_index(drop=True)
    combined.to_csv(TABLE_DIR / "academic_all_model_forecast_ranking.csv", index=False)
    return combined


def main() -> None:
    ensure_directories()
    raw = load_raw_data()
    save_variable_dictionary()

    raw_adf = adf_table(raw, "academic_raw_adf_tests.csv")
    cointegration = run_pairwise_cointegration(raw, raw_adf)
    model_df = transform_for_models(raw)
    dummies = make_break_dummies(model_df.index)
    transformed_adf = adf_table(model_df, "academic_transformed_adf_tests.csv")

    save_eda_tables(raw, model_df)
    save_eda_figures(raw, model_df)

    var_results, lag_table = select_and_fit_var(model_df, dummies)
    p_lags = var_results.k_ar
    save_model_architecture(p_lags)
    select_varx_lag_order(model_df, dummies)
    varx_results = fit_varx(model_df, dummies, p_lags)
    save_parameter_significance(var_results, "academic_var")
    save_parameter_significance(varx_results, "academic_varx")
    lag_residual_autocorrelation_robustness(model_df, dummies)
    var_diagnostics = residual_diagnostics(var_results, "academic_var")
    varx_diagnostics = residual_diagnostics(varx_results, "academic_varx")
    granger = granger_map(model_df, p_lags)
    save_irf_fevd(var_results)

    train, test = split_train_test(model_df)
    econ_forecasts, econ_metrics = econometric_forecasts(train, test, dummies, p_lags)
    rolling_metrics = rolling_varx_forecasts(model_df, dummies, p_lags)
    ml_metrics = machine_learning_benchmarks(model_df, p_lags)
    combined = combined_model_ranking(econ_metrics, ml_metrics)

    print("Academic time-series pipeline completed successfully.")
    print(f"Raw data shape: {raw.shape}")
    print(f"Model data shape: {model_df.shape}")
    print(f"Selected VAR lag order: {p_lags}")
    print("\nADF tests after transformation:")
    print(transformed_adf[["variable", "p_value", "stationary_at_5pct"]])
    print("\nCointegration tests:")
    print(cointegration)
    print("\nVAR diagnostics:")
    print(var_diagnostics)
    print("\nVARX diagnostics:")
    print(varx_diagnostics)
    print("\nSignificant Granger relationships:")
    print(granger.loc[granger["significant_at_5pct"], ["source", "target", "p_value"]])
    print("\nRolling VARX 3-month forecast RMSE:")
    print(rolling_metrics)
    print("\nCombined forecast ranking:")
    print(combined)


if __name__ == "__main__":
    main()
