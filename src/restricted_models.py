from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.api import VAR

from src.dashboard_helpers import DATA_DIR, TABLE_DIR, round_numeric
from src.diagnostics import (
    acf_values,
    residual_cross_correlation_summary,
    residual_normality_table,
    residual_test_table,
)
from src.forecasting import directional_accuracy, rmse


TEST_MONTHS = 36
ACF_MAX_LAG = 12
WEAK_BLOCK_PVALUE = 0.05

VAR_ENDOG = ["INF", "FEDFUNDS", "UNRATE", "INDPRO_GROWTH", "SENTIMENT_CHANGE"]
VAR_LAG = 5
VARX_ENDOG = ["INF", "UNRATE", "INDPRO_GROWTH", "M2_GROWTH"]
VARX_EXOG = ["FEDFUNDS", "SENTIMENT_CHANGE"]
VARX_LAG = 4

RESTRICTION_OUTPUTS = {
    "VAR": {
        "restrictions": TABLE_DIR / "restricted_var_restrictions.csv",
        "metrics": TABLE_DIR / "restricted_var_metrics.csv",
        "diagnostics": TABLE_DIR / "restricted_var_residual_diagnostics.csv",
        "forecasts": TABLE_DIR / "restricted_var_forecast_comparison.csv",
    },
    "VARX": {
        "restrictions": TABLE_DIR / "restricted_varx_restrictions.csv",
        "metrics": TABLE_DIR / "restricted_varx_metrics.csv",
        "diagnostics": TABLE_DIR / "restricted_varx_residual_diagnostics.csv",
        "forecasts": TABLE_DIR / "restricted_varx_forecast_comparison.csv",
    },
}


@dataclass(frozen=True)
class RestrictedSpec:
    model_type: str
    model_name: str
    endog: list[str]
    exog: list[str]
    lag_order: int


def load_modeling_frame() -> pd.DataFrame:
    model = pd.read_csv(DATA_DIR / "academic_model_data.csv", parse_dates=["date"], index_col="date")
    dummies = pd.read_csv(DATA_DIR / "academic_break_dummies.csv", parse_dates=["date"], index_col="date")
    return pd.concat([model, dummies], axis=1)


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data.iloc[:-TEST_MONTHS].copy()
    test = data.iloc[-TEST_MONTHS:].copy()
    return train, test


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def compact_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def lagged_design(
    endog: pd.DataFrame,
    lag_order: int,
    exog: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for position in range(lag_order, len(endog)):
        row = {"const": 1.0}
        if exog is not None:
            for column in exog.columns:
                row[column] = exog.iloc[position][column]
        for lag in range(1, lag_order + 1):
            for column in endog.columns:
                row[f"L{lag}.{column}"] = endog.iloc[position - lag][column]
        rows.append(row)
    x = pd.DataFrame(rows, index=endog.index[lag_order:])
    y = endog.iloc[lag_order:].copy()
    return x, y


def fit_full_equations(x: pd.DataFrame, y: pd.DataFrame, lag_order: int) -> dict[str, dict[str, object]]:
    models: dict[str, dict[str, object]] = {}
    hac_lags = min(12, max(1, lag_order))
    for equation in y.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = sm.OLS(y[equation], x).fit()
            hc3 = fitted.get_robustcov_results(cov_type="HC3")
            hac = fitted.get_robustcov_results(cov_type="HAC", maxlags=hac_lags)
        models[equation] = {"classical": fitted, "hc3": hc3, "hac": hac}
    return models


def block_f_pvalue(fitted: object, x_columns: pd.Index, block_columns: list[str]) -> float:
    indices = [x_columns.get_loc(column) for column in block_columns if column in x_columns]
    if not indices:
        return np.nan
    restriction = np.zeros((len(indices), len(x_columns)))
    for row_idx, col_idx in enumerate(indices):
        restriction[row_idx, col_idx] = 1.0
    try:
        return safe_float(fitted.f_test(restriction).pvalue)
    except Exception:
        return np.nan


def robust_block_min_pvalues(model_bundle: dict[str, object], x_columns: pd.Index, block_columns: list[str]) -> tuple[float, float]:
    indices = [x_columns.get_loc(column) for column in block_columns if column in x_columns]
    if not indices:
        return np.nan, np.nan
    hc3 = model_bundle["hc3"]
    hac = model_bundle["hac"]
    hc3_min = float(np.nanmin([safe_float(hc3.pvalues[index]) for index in indices]))
    hac_min = float(np.nanmin([safe_float(hac.pvalues[index]) for index in indices]))
    return hc3_min, hac_min


def granger_pvalues(train_endog: pd.DataFrame, lag_order: int, model_type: str) -> pd.DataFrame:
    rows = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = VAR(train_endog).fit(lag_order, trend="c")
        for target in train_endog.columns:
            for source in train_endog.columns:
                if source == target:
                    continue
                try:
                    result = fitted.test_causality(target, [source], kind="f")
                    p_value = safe_float(result.pvalue)
                    stat_value = safe_float(result.test_statistic)
                except Exception:
                    p_value = np.nan
                    stat_value = np.nan
                rows.append(
                    {
                        "model_type": model_type,
                        "source": source,
                        "target": target,
                        "granger_p_value": p_value,
                        "granger_test_statistic": stat_value,
                    }
                )
    except Exception:
        for target in train_endog.columns:
            for source in train_endog.columns:
                if source != target:
                    rows.append(
                        {
                            "model_type": model_type,
                            "source": source,
                            "target": target,
                            "granger_p_value": np.nan,
                            "granger_test_statistic": np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def protected_block(model_type: str, target: str, source: str) -> tuple[bool, str]:
    if source == target:
        return True, "own lags are retained to preserve persistence and dynamic adjustment"
    if model_type == "VAR" and source == "FEDFUNDS":
        return True, "FEDFUNDS lag block is retained as a theoretically central monetary-policy channel"
    if model_type == "VAR" and target == "FEDFUNDS" and source == "INF":
        return True, "INF -> FEDFUNDS is retained because it represents a plausible policy-reaction channel"
    if model_type == "VAR" and {source, target} == {"UNRATE", "INDPRO_GROWTH"}:
        return True, "UNRATE and industrial production are retained as a core real-side transmission pair"
    return False, ""


def choose_restrictions(
    spec: RestrictedSpec,
    x: pd.DataFrame,
    y: pd.DataFrame,
    full_models: dict[str, dict[str, object]],
    granger: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for target in spec.endog:
        for source in spec.endog:
            block_columns = [f"L{lag}.{source}" for lag in range(1, spec.lag_order + 1)]
            protected, protection_reason = protected_block(spec.model_type, target, source)
            model_bundle = full_models[target]
            fitted = model_bundle["classical"]
            block_p = block_f_pvalue(fitted, x.columns, block_columns)
            hc3_min, hac_min = robust_block_min_pvalues(model_bundle, x.columns, block_columns)
            granger_row = granger.loc[(granger["source"] == source) & (granger["target"] == target)]
            granger_p = safe_float(granger_row["granger_p_value"].iloc[0]) if not granger_row.empty else np.nan
            weak_predictive = pd.isna(granger_p) or granger_p >= WEAK_BLOCK_PVALUE
            weak_joint = pd.isna(block_p) or block_p >= WEAK_BLOCK_PVALUE
            weak_robust = (
                (pd.isna(hc3_min) or hc3_min >= WEAK_BLOCK_PVALUE)
                and (pd.isna(hac_min) or hac_min >= WEAK_BLOCK_PVALUE)
            )
            impose = bool((not protected) and weak_predictive and weak_joint and weak_robust)
            if impose:
                reason = (
                    "removed because the source block does not show Granger/block evidence "
                    "and robust lag-level evidence is weak"
                )
            elif protected:
                reason = protection_reason
            else:
                reason = "retained because predictive, joint, robust, or economic evidence is not weak enough"
            rows.append(
                {
                    "model_type": spec.model_type,
                    "restricted_model": spec.model_name,
                    "equation": target,
                    "source_variable": source,
                    "restriction_type": "drop all lagged coefficients from source block" if impose else "retain source block",
                    "removed_parameters": spec.lag_order if impose else 0,
                    "remaining_block_parameters": 0 if impose else spec.lag_order,
                    "granger_p_value": granger_p,
                    "classical_block_f_p_value": block_p,
                    "min_hc3_p_value_in_block": hc3_min,
                    "min_hac_p_value_in_block": hac_min,
                    "economic_override": protected,
                    "imposed": impose,
                    "decision_rule": (
                        f"drop only if not protected and Granger, block F, HC3, and HAC evidence are all weak at p>={WEAK_BLOCK_PVALUE:.2f}"
                    ),
                    "reason": reason,
                    "Acceptable if": "restrictions are defensible only when forecast/residual diagnostics do not deteriorate materially",
                }
            )
    return pd.DataFrame(rows)


def restricted_column_map(restrictions: pd.DataFrame, lag_order: int) -> dict[str, list[str]]:
    excluded: dict[str, list[str]] = {}
    imposed = restrictions.loc[restrictions["imposed"].astype(bool)]
    for equation, group in imposed.groupby("equation"):
        columns: list[str] = []
        for source in group["source_variable"]:
            columns.extend([f"L{lag}.{source}" for lag in range(1, lag_order + 1)])
        excluded[equation] = columns
    return excluded


def estimate_restricted_equations(
    x: pd.DataFrame,
    y: pd.DataFrame,
    restrictions: pd.DataFrame,
    lag_order: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    excluded_by_equation = restricted_column_map(restrictions, lag_order)
    equations: dict[str, object] = {}
    residuals = pd.DataFrame(index=y.index)
    fitted_values = pd.DataFrame(index=y.index)
    for equation in y.columns:
        excluded = set(excluded_by_equation.get(equation, []))
        columns = [column for column in x.columns if column not in excluded]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = sm.OLS(y[equation], x[columns]).fit()
        equations[equation] = fitted
        residuals[equation] = fitted.resid
        fitted_values[equation] = fitted.fittedvalues
    return equations, residuals, fitted_values


def coefficient_matrices(equations: dict[str, object], endog: list[str], lag_order: int) -> np.ndarray:
    coefs = np.zeros((lag_order, len(endog), len(endog)))
    for eq_idx, equation in enumerate(endog):
        params = equations[equation].params
        for lag in range(1, lag_order + 1):
            for src_idx, source in enumerate(endog):
                coefs[lag - 1, eq_idx, src_idx] = safe_float(params.get(f"L{lag}.{source}", 0.0))
    return coefs


def companion_stability(coefs: np.ndarray) -> tuple[bool, float, pd.DataFrame]:
    lag_order, k_endog, _ = coefs.shape
    companion = np.zeros((k_endog * lag_order, k_endog * lag_order))
    companion[:k_endog, : k_endog * lag_order] = np.hstack([coefs[lag] for lag in range(lag_order)])
    if lag_order > 1:
        companion[k_endog:, :-k_endog] = np.eye(k_endog * (lag_order - 1))
    eigenvalues = np.linalg.eigvals(companion)
    roots = pd.DataFrame(
        {
            "real": eigenvalues.real,
            "imag": eigenvalues.imag,
            "modulus": np.abs(eigenvalues),
            "Acceptable if": "maximum companion eigenvalue modulus is below 1",
        }
    )
    max_modulus = float(np.max(np.abs(eigenvalues))) if len(eigenvalues) else np.nan
    return bool(max_modulus < 1), max_modulus, roots


def recursive_forecast(
    equations: dict[str, object],
    train_endog: pd.DataFrame,
    test_endog: pd.DataFrame,
    lag_order: int,
    exog_future: pd.DataFrame | None = None,
) -> pd.DataFrame:
    history = train_endog.copy()
    rows = []
    for step, date in enumerate(test_endog.index):
        base_row = {"const": 1.0}
        if exog_future is not None:
            for column in exog_future.columns:
                base_row[column] = exog_future.iloc[step][column]
        for lag in range(1, lag_order + 1):
            lag_values = history.iloc[-lag]
            for column in train_endog.columns:
                base_row[f"L{lag}.{column}"] = lag_values[column]
        pred = {}
        for equation, fitted in equations.items():
            value = 0.0
            for parameter, coefficient in fitted.params.items():
                value += coefficient * base_row.get(parameter, 0.0)
            pred[equation] = value
        rows.append(pred)
        history = pd.concat([history, pd.DataFrame([pred], index=[date])])
    return pd.DataFrame(rows, index=test_endog.index)


def forecast_metrics_table(train_endog: pd.DataFrame, test_endog: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in forecasts.columns:
        actual = test_endog[variable]
        predicted = forecasts[variable]
        naive = pd.Series(train_endog[variable].iloc[-1], index=actual.index)
        previous_actual = actual.shift(1)
        previous_actual.iloc[0] = train_endog[variable].iloc[-1]
        rows.append(
            {
                "variable": variable,
                "RMSE": rmse(actual, predicted),
                "MAE": float(np.mean(np.abs(actual - predicted))),
                "naive_RMSE": rmse(actual, naive),
                "relative_RMSE_vs_no_leak_naive": rmse(actual, predicted) / rmse(actual, naive) if rmse(actual, naive) else np.nan,
                "directional_accuracy": directional_accuracy(actual, predicted, previous_actual),
                "Acceptable if": "lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better",
            }
        )
    return pd.DataFrame(rows)


def equation_fit_metrics(y: pd.DataFrame, fitted_values: pd.DataFrame, residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for equation in y.columns:
        sse = float(np.square(residuals[equation]).sum())
        centered = y[equation] - y[equation].mean()
        sst = float(np.square(centered).sum())
        r2 = 1 - sse / sst if sst else np.nan
        rows.append(
            {
                "equation": equation,
                "r_squared": r2,
                "residual_std_error": float(residuals[equation].std()),
                "Acceptable if": "higher R-squared is better but residual diagnostics and forecast performance are more important",
            }
        )
    return pd.DataFrame(rows)


def residual_acf_summary(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for equation in residuals.columns:
        acf_df = acf_values(residuals[equation], max_lag=ACF_MAX_LAG)
        rows.append(
            {
                "equation": equation,
                "acf_exceedance_count": int(acf_df["outside_bound"].sum()) if not acf_df.empty else 0,
                "acf_exceedance_share": float(acf_df["outside_bound"].mean()) if not acf_df.empty else np.nan,
                "max_abs_acf_lag_1_to_12": float(acf_df["acf"].abs().max()) if not acf_df.empty else np.nan,
                "lag_of_max_abs_acf": int(acf_df.loc[acf_df["acf"].abs().idxmax(), "lag"]) if not acf_df.empty else np.nan,
                "Acceptable if": "positive-lag residual ACF bars should mostly stay inside +/-1.96/sqrt(T); lag 0 is excluded",
            }
        )
    return pd.DataFrame(rows)


def approximate_portmanteau(residuals: pd.DataFrame, max_lag: int = ACF_MAX_LAG) -> tuple[float, float, int]:
    clean = residuals.dropna()
    n_obs = len(clean)
    k_endog = clean.shape[1]
    max_lag_used = min(max_lag, n_obs - 2)
    if max_lag_used <= 0:
        return np.nan, np.nan, 0
    values = clean.to_numpy()
    centered = values - values.mean(axis=0, keepdims=True)
    gamma_0 = centered.T @ centered / n_obs
    inv_gamma_0 = np.linalg.pinv(gamma_0)
    q_stat = 0.0
    for lag in range(1, max_lag_used + 1):
        left = centered[lag:]
        right = centered[:-lag]
        gamma_h = left.T @ right / n_obs
        q_stat += np.trace(gamma_h.T @ inv_gamma_0 @ gamma_h @ inv_gamma_0) / max(n_obs - lag, 1)
    q_stat *= n_obs * n_obs
    dof = max(1, k_endog * k_endog * max_lag_used)
    p_value = float(stats.chi2.sf(q_stat, dof))
    return float(q_stat), p_value, dof


def residual_diagnostics_table(
    spec: RestrictedSpec,
    residuals: pd.DataFrame,
    fit_metrics: pd.DataFrame,
) -> pd.DataFrame:
    tests = residual_test_table(residuals)
    normality = residual_normality_table(residuals)
    acf_summary = residual_acf_summary(residuals)
    q_stat, portmanteau_p, portmanteau_df = approximate_portmanteau(residuals)
    rows = []
    for equation in residuals.columns:
        test_row = tests.loc[tests["equation"] == equation].iloc[0]
        norm_row = normality.loc[normality["equation"] == equation].iloc[0]
        acf_row = acf_summary.loc[acf_summary["equation"] == equation].iloc[0]
        fit_row = fit_metrics.loc[fit_metrics["equation"] == equation].iloc[0]
        rows.append(
            {
                "model_type": spec.model_type,
                "restricted_model": spec.model_name,
                "diagnostic_type": "equation_residual",
                "equation": equation,
                "source_residual": "",
                "target_residual": "",
                "lag_of_max_abs_ccf": np.nan,
                "max_abs_ccf": np.nan,
                "durbin_watson": test_row["durbin_watson"],
                "ljung_box_p_value": test_row["ljung_box_p_value"],
                "arch_lm_p_value": test_row["arch_lm_p_value"],
                "jarque_bera_p_value": norm_row["jarque_bera_p_value"],
                "skewness": norm_row["skewness"],
                "kurtosis_pearson": norm_row["kurtosis_pearson"],
                "acf_exceedance_count_lags_1_to_12": acf_row["acf_exceedance_count"],
                "acf_exceedance_share_lags_1_to_12": acf_row["acf_exceedance_share"],
                "max_abs_acf_lag_1_to_12": acf_row["max_abs_acf_lag_1_to_12"],
                "r_squared": fit_row["r_squared"],
                "portmanteau_statistic_approx": q_stat,
                "portmanteau_df_approx": portmanteau_df,
                "portmanteau_p_value_approx": portmanteau_p,
                "Acceptable if": "LB/JB/ARCH p-values > 0.05, most positive-lag ACF inside bounds, and approximate Portmanteau p-value > 0.05",
            }
        )
    ccf = residual_cross_correlation_summary(residuals, max_lag=ACF_MAX_LAG)
    if not ccf.empty:
        for _, ccf_row in ccf.iterrows():
            rows.append(
                {
                    "model_type": spec.model_type,
                    "restricted_model": spec.model_name,
                    "diagnostic_type": "lagged_cross_correlation",
                    "equation": "",
                    "source_residual": ccf_row["source_residual"],
                    "target_residual": ccf_row["target_residual"],
                    "lag_of_max_abs_ccf": ccf_row["lag_of_max_abs_ccf"],
                    "max_abs_ccf": ccf_row["max_abs_ccf"],
                    "ccf_at_max_abs_lag": ccf_row["ccf_at_max_abs_lag"],
                    "durbin_watson": np.nan,
                    "ljung_box_p_value": np.nan,
                    "arch_lm_p_value": np.nan,
                    "jarque_bera_p_value": np.nan,
                    "skewness": np.nan,
                    "kurtosis_pearson": np.nan,
                    "acf_exceedance_count_lags_1_to_12": np.nan,
                    "acf_exceedance_share_lags_1_to_12": np.nan,
                    "max_abs_acf_lag_1_to_12": np.nan,
                    "r_squared": np.nan,
                    "portmanteau_statistic_approx": q_stat,
                    "portmanteau_df_approx": portmanteau_df,
                    "portmanteau_p_value_approx": portmanteau_p,
                    "Acceptable if": "cross-equation residual CCF includes lag 0; smaller absolute values indicate weaker remaining residual dependence",
                }
            )
    return pd.DataFrame(rows)


def total_parameters(equations: dict[str, object]) -> int:
    return int(sum(len(fitted.params) for fitted in equations.values()))


def baseline_metrics(model_type: str) -> pd.DataFrame:
    path = TABLE_DIR / ("optimized_final_var_metrics.csv" if model_type == "VAR" else "optimized_final_varx_metrics.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def baseline_model_specs(model_type: str) -> pd.Series | None:
    path = TABLE_DIR / "optimized_final_model_specs.csv"
    if not path.exists():
        return None
    specs = pd.read_csv(path)
    subset = specs.loc[specs["model_type"] == model_type]
    return subset.iloc[0] if not subset.empty else None


def forecast_comparison_table(
    spec: RestrictedSpec,
    restricted_metrics: pd.DataFrame,
) -> pd.DataFrame:
    base = baseline_metrics(spec.model_type)
    rows = []
    for _, restricted_row in restricted_metrics.iterrows():
        variable = restricted_row["variable"]
        base_row = base.loc[base["variable"] == variable] if not base.empty and "variable" in base else pd.DataFrame()
        baseline_rmse = safe_float(base_row["RMSE"].iloc[0]) if not base_row.empty else np.nan
        baseline_mae = safe_float(base_row["MAE"].iloc[0]) if not base_row.empty else np.nan
        rows.append(
            {
                "model_type": spec.model_type,
                "restricted_model": spec.model_name,
                "variable": variable,
                "baseline_RMSE": baseline_rmse,
                "restricted_RMSE": restricted_row["RMSE"],
                "RMSE_change_restricted_minus_baseline": restricted_row["RMSE"] - baseline_rmse,
                "restricted_better_RMSE": restricted_row["RMSE"] < baseline_rmse if pd.notna(baseline_rmse) else np.nan,
                "baseline_MAE": baseline_mae,
                "restricted_MAE": restricted_row["MAE"],
                "MAE_change_restricted_minus_baseline": restricted_row["MAE"] - baseline_mae,
                "restricted_better_MAE": restricted_row["MAE"] < baseline_mae if pd.notna(baseline_mae) else np.nan,
                "restricted_relative_RMSE_vs_naive": restricted_row["relative_RMSE_vs_no_leak_naive"],
                "restricted_directional_accuracy": restricted_row["directional_accuracy"],
                "Acceptable if": "restricted model should preserve or improve RMSE/MAE with fewer parameters",
            }
        )
    return pd.DataFrame(rows)


def metrics_table(
    spec: RestrictedSpec,
    restrictions: pd.DataFrame,
    equations: dict[str, object],
    residuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    stable: bool,
    max_eigenvalue_modulus: float,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    baseline_spec = baseline_model_specs(spec.model_type)
    baseline_params = safe_float(baseline_spec["total_parameters"]) if baseline_spec is not None else np.nan
    restricted_params = total_parameters(equations)
    removed_params = int(restrictions["removed_parameters"].sum())
    params_per_equation = {
        equation: int(len(fitted.params))
        for equation, fitted in equations.items()
    }
    equation_diag = diagnostics.loc[diagnostics["diagnostic_type"] == "equation_residual"]
    cross_diag = diagnostics.loc[diagnostics["diagnostic_type"] == "lagged_cross_correlation"]
    inflation_metrics = forecast_metrics.loc[forecast_metrics["variable"] == "INF"]
    inflation_rmse = safe_float(inflation_metrics["RMSE"].iloc[0]) if not inflation_metrics.empty else np.nan
    inflation_mae = safe_float(inflation_metrics["MAE"].iloc[0]) if not inflation_metrics.empty else np.nan
    baseline_inflation_rmse = safe_float(baseline_spec["inflation_RMSE"]) if baseline_spec is not None else np.nan
    rmse_change = inflation_rmse - baseline_inflation_rmse if pd.notna(baseline_inflation_rmse) else np.nan
    interpretation = (
        "useful parsimonious alternative"
        if removed_params > 0 and pd.notna(rmse_change) and rmse_change <= 0.02 and stable
        else "robustness check only; unrestricted baseline remains preferred unless diagnostics/forecasting clearly improve"
    )
    return pd.DataFrame(
        [
            {
                "model_type": spec.model_type,
                "restricted_model": spec.model_name,
                "baseline_model": baseline_spec["candidate_name"] if baseline_spec is not None else "",
                "endogenous_variables": compact_list(spec.endog),
                "exogenous_variables": compact_list(spec.exog),
                "lag_order": spec.lag_order,
                "n_train_effective": len(residuals),
                "n_test": len(forecasts),
                "k_endogenous": len(spec.endog),
                "k_exogenous": len(spec.exog),
                "baseline_total_parameters": baseline_params,
                "restricted_total_parameters": restricted_params,
                "parameters_removed": removed_params,
                "parameter_reduction_share": removed_params / baseline_params if baseline_params else np.nan,
                "remaining_parameters_per_equation": "; ".join([f"{key}: {value}" for key, value in params_per_equation.items()]),
                "stable": stable,
                "max_companion_eigenvalue_modulus": max_eigenvalue_modulus,
                "inflation_RMSE": inflation_rmse,
                "inflation_MAE": inflation_mae,
                "baseline_inflation_RMSE": baseline_inflation_rmse,
                "inflation_RMSE_change_restricted_minus_baseline": rmse_change,
                "mean_RMSE": safe_float(forecast_metrics["RMSE"].mean()),
                "mean_MAE": safe_float(forecast_metrics["MAE"].mean()),
                "min_ljung_box_p_value": safe_float(equation_diag["ljung_box_p_value"].min()) if not equation_diag.empty else np.nan,
                "baseline_min_ljung_box_p_value": safe_float(baseline_spec["min_ljung_box_p_value"]) if baseline_spec is not None else np.nan,
                "mean_ljung_box_p_value": safe_float(equation_diag["ljung_box_p_value"].mean()) if not equation_diag.empty else np.nan,
                "max_acf_exceedance_share": safe_float(equation_diag["acf_exceedance_share_lags_1_to_12"].max()) if not equation_diag.empty else np.nan,
                "baseline_acf_exceedance_share": safe_float(baseline_spec["acf_exceedance_share"]) if baseline_spec is not None else np.nan,
                "max_abs_cross_ccf_including_lag0": safe_float(cross_diag["max_abs_ccf"].max()) if not cross_diag.empty else np.nan,
                "baseline_max_abs_cross_ccf": safe_float(baseline_spec["max_abs_cross_ccf"]) if baseline_spec is not None else np.nan,
                "min_jarque_bera_p_value": safe_float(equation_diag["jarque_bera_p_value"].min()) if not equation_diag.empty else np.nan,
                "baseline_min_jarque_bera_p_value": safe_float(baseline_spec["min_jarque_bera_p_value"]) if baseline_spec is not None else np.nan,
                "min_arch_lm_p_value": safe_float(equation_diag["arch_lm_p_value"].min()) if not equation_diag.empty else np.nan,
                "baseline_min_arch_lm_p_value": safe_float(baseline_spec["min_arch_lm_p_value"]) if baseline_spec is not None else np.nan,
                "portmanteau_p_value_approx": safe_float(equation_diag["portmanteau_p_value_approx"].iloc[0]) if not equation_diag.empty else np.nan,
                "baseline_portmanteau_whiteness_p_value": safe_float(baseline_spec["portmanteau_whiteness_p_value"]) if baseline_spec is not None else np.nan,
                "interpretation": interpretation,
                "Acceptable if": "restricted model is stable, materially more parsimonious, and does not worsen forecasts or residual diagnostics",
            }
        ]
    )


def run_restricted_model(spec: RestrictedSpec, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train, test = split_data(data)
    train_endog = train[spec.endog]
    test_endog = test[spec.endog]
    train_exog = train[spec.exog] if spec.exog else None
    test_exog = test[spec.exog] if spec.exog else None
    x, y = lagged_design(train_endog, spec.lag_order, train_exog)
    full_models = fit_full_equations(x, y, spec.lag_order)
    granger = granger_pvalues(train_endog, spec.lag_order, spec.model_type)
    restrictions = choose_restrictions(spec, x, y, full_models, granger)
    equations, residuals, fitted_values = estimate_restricted_equations(x, y, restrictions, spec.lag_order)
    coefs = coefficient_matrices(equations, spec.endog, spec.lag_order)
    stable, max_modulus, roots = companion_stability(coefs)
    forecasts = recursive_forecast(equations, train_endog, test_endog, spec.lag_order, test_exog)
    forecast_metrics = forecast_metrics_table(train_endog, test_endog, forecasts)
    fit_metrics = equation_fit_metrics(y, fitted_values, residuals)
    diagnostics = residual_diagnostics_table(spec, residuals, fit_metrics)
    metrics = metrics_table(spec, restrictions, equations, residuals, forecasts, forecast_metrics, stable, max_modulus, diagnostics)
    comparison = forecast_comparison_table(spec, forecast_metrics)
    return {
        "restrictions": restrictions,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "forecasts": comparison,
        "roots": roots,
        "restricted_forecast_metrics": forecast_metrics,
    }


def save_outputs(results: dict[str, dict[str, pd.DataFrame]]) -> None:
    for model_type, tables in results.items():
        for key, path in RESTRICTION_OUTPUTS[model_type].items():
            round_numeric(tables[key]).to_csv(path, index=False)


def run_all() -> dict[str, dict[str, pd.DataFrame]]:
    data = load_modeling_frame()
    specs = [
        RestrictedSpec("VAR", "Restricted_VAR_block_parsimony", VAR_ENDOG, [], VAR_LAG),
        RestrictedSpec("VARX", "Restricted_VARX_block_parsimony", VARX_ENDOG, VARX_EXOG, VARX_LAG),
    ]
    results = {spec.model_type: run_restricted_model(spec, data) for spec in specs}
    save_outputs(results)
    return results


def main() -> None:
    run_all()
    print("Restricted VAR/VARX robustness tables written to outputs/tables.")


if __name__ == "__main__":
    main()
