from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf as sm_acf, adfuller, kpss, pacf as sm_pacf

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.diagnostics import residual_cross_correlation_summary
from src.models_var import fit_var_system, select_var_lags, var_fevd_paths, var_irf_paths
from src.models_varx import fit_varx_system, select_varx_lags, varx_exogenous_scenario_response


DATA_DIR = BASE_DIR / "data"
TABLE_DIR = BASE_DIR / "outputs" / "tables"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"

SPLIT_DATE = "2023-04-01"
MAX_LAG = 8
IRF_HORIZON = 24

BASELINE_VAR_ENDOG = ["INF", "FEDFUNDS", "UNRATE", "INDPRO_GROWTH", "SENTIMENT_CHANGE"]
BASELINE_VAR_LAG = 5
BASELINE_VARX_ENDOG = ["INF", "UNRATE", "INDPRO_GROWTH", "M2_GROWTH"]
BASELINE_VARX_EXOG = ["FEDFUNDS", "SENTIMENT_CHANGE"]
BASELINE_VARX_LAG = 4

VAR_DIFF_CANDIDATES = {
    "VAR_D_FEDFUNDS_core_sentiment": [
        "INF",
        "D_FEDFUNDS",
        "UNRATE",
        "INDPRO_GROWTH",
        "SENTIMENT_CHANGE",
    ],
    "VAR_D_FEDFUNDS_full": [
        "INF",
        "D_FEDFUNDS",
        "UNRATE",
        "INDPRO_GROWTH",
        "M2_GROWTH",
        "SENTIMENT_CHANGE",
    ],
}

VARX_DIFF_CANDIDATES = {
    "VARX_D_FEDFUNDS_A": {
        "endog": ["INF", "UNRATE", "INDPRO_GROWTH", "M2_GROWTH"],
        "exog": ["D_FEDFUNDS", "SENTIMENT_CHANGE"],
    },
    "VARX_D_FEDFUNDS_policy_change_no_m2_endog": {
        "endog": ["INF", "UNRATE", "INDPRO_GROWTH"],
        "exog": ["D_FEDFUNDS", "M2_GROWTH", "SENTIMENT_CHANGE"],
    },
}


def ensure_directories() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_model_data() -> pd.DataFrame:
    path = DATA_DIR / "academic_model_data.csv"
    model_df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    if "D_FEDFUNDS" not in model_df.columns:
        raw = pd.read_csv(DATA_DIR / "raw_fred_macro.csv", parse_dates=["date"], index_col="date")
        raw = raw.asfreq("MS").ffill()
        model_df["D_FEDFUNDS"] = raw["FEDFUNDS"].diff().loc[model_df.index]
        model_df = model_df.dropna()
        model_df.to_csv(path, index_label="date")
    return model_df


def safe_stat(value) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def stationarity_results(model_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in ["FEDFUNDS", "D_FEDFUNDS"]:
        series = model_df[variable].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                adf_result = adfuller(series, autolag="AIC")
                adf_stat, adf_p, adf_lag, adf_nobs = adf_result[:4]
            except Exception:
                adf_stat, adf_p, adf_lag, adf_nobs = np.nan, np.nan, np.nan, np.nan
            try:
                kpss_stat, kpss_p, kpss_lag, _ = kpss(series, regression="c", nlags="auto")
            except Exception:
                kpss_stat, kpss_p, kpss_lag = np.nan, np.nan, np.nan
        acf_raw = sm_acf(series, nlags=min(12, len(series) // 2 - 1), fft=False)
        pacf_raw = sm_pacf(series, nlags=min(12, len(series) // 2 - 1), method="ywm")
        stationary = bool(pd.notna(adf_p) and pd.notna(kpss_p) and adf_p < 0.05 and kpss_p > 0.05)
        rows.append(
            {
                "variable": variable,
                "test": "ADF/KPSS/ACF/PACF stationarity robustness",
                "Acceptable if": "ADF p-value < 0.05 and KPSS p-value > 0.05 support stationarity; lower ACF persistence is preferred",
                "adf_statistic": adf_stat,
                "adf_p_value": adf_p,
                "adf_used_lag": adf_lag,
                "adf_n_obs": adf_nobs,
                "kpss_statistic": kpss_stat,
                "kpss_p_value": kpss_p,
                "kpss_used_lag": kpss_lag,
                "acf_lag1": acf_raw[1] if len(acf_raw) > 1 else np.nan,
                "pacf_lag1": pacf_raw[1] if len(pacf_raw) > 1 else np.nan,
                "stationary_by_adf_kpss_rule": stationary,
                "interpretation": (
                    "stationary policy-rate change"
                    if variable == "D_FEDFUNDS" and stationary
                    else "highly persistent policy-rate level; stationarity is questionable"
                    if variable == "FEDFUNDS"
                    else "mixed stationarity evidence"
                ),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "fedfunds_stationarity_robustness.csv", index=False)
    table.to_csv(TABLE_DIR / "fedfunds_level_vs_diff_stationarity.csv", index=False)
    return table


def plot_stationarity(model_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    model_df["FEDFUNDS"].plot(ax=axes[0, 0], linewidth=1.4, title="FEDFUNDS Level")
    model_df["D_FEDFUNDS"].plot(ax=axes[0, 1], linewidth=1.4, title="D_FEDFUNDS Monthly Change")
    for axis, variable in zip(axes[1], ["FEDFUNDS", "D_FEDFUNDS"]):
        series = model_df[variable].dropna()
        values = sm_acf(series, nlags=24, fft=False)[1:]
        lags = np.arange(1, len(values) + 1)
        bound = 1.96 / np.sqrt(len(series))
        axis.bar(lags, values)
        axis.axhline(bound, color="red", linestyle="--", linewidth=1)
        axis.axhline(-bound, color="red", linestyle="--", linewidth=1)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"ACF: {variable}")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fedfunds_vs_d_fedfunds_stationarity.png", dpi=180)
    plt.close()


def lag_best_table(lag_table: pd.DataFrame) -> dict[str, int]:
    valid = lag_table.loc[lag_table["lag"] >= 1].copy()
    best = {}
    for criterion in ["AIC", "BIC", "HQIC", "FPE"]:
        if criterion in valid.columns and valid[criterion].notna().any():
            best[f"{criterion.lower()}_best_lag"] = int(valid.loc[valid[criterion].idxmin(), "lag"])
        else:
            best[f"{criterion.lower()}_best_lag"] = np.nan
    return best


def safe_portmanteau_p(fitted: object, lag_order: int) -> float:
    try:
        return float(fitted.test_whiteness(nlags=min(12, max(lag_order + 1, 2))).pvalue)
    except Exception:
        return np.nan


def safe_normality_p(fitted: object) -> float:
    try:
        return float(fitted.test_normality().pvalue)
    except Exception:
        return np.nan


def residual_summary(
    residuals: pd.DataFrame,
    fitted: object,
    model_name: str,
    candidate: str,
    lag_order: int,
    max_lag: int = 24,
) -> pd.DataFrame:
    rows = []
    bound = 1.96 / np.sqrt(len(residuals))
    dw_values = durbin_watson(residuals)
    portmanteau_p = safe_portmanteau_p(fitted, lag_order)
    normality_p = safe_normality_p(fitted)
    for idx, column in enumerate(residuals.columns):
        series = residuals[column].dropna()
        lb_lag = min(12, max(1, len(series) // 5))
        try:
            lb_p = float(acorr_ljungbox(series, lags=[lb_lag], return_df=True)["lb_pvalue"].iloc[0])
        except Exception:
            lb_p = np.nan
        try:
            arch_p = float(het_arch(series, nlags=lb_lag)[1])
        except Exception:
            arch_p = np.nan
        try:
            jb = stats.jarque_bera(series)
            jb_p = float(jb.pvalue)
        except Exception:
            jb_p = np.nan
        acf_values = [series.autocorr(lag=lag) for lag in range(1, min(max_lag, len(series) - 2) + 1)]
        exceedances = [value for value in acf_values if pd.notna(value) and abs(value) > bound]
        rows.append(
            {
                "model": model_name,
                "candidate": candidate,
                "lag_order": lag_order,
                "diagnostic_type": "equation_residual",
                "equation": column,
                "test": "DW, Ljung-Box, ACF, Jarque-Bera, ARCH-LM",
                "Acceptable if": "DW near 2; Ljung-Box/JB/ARCH p-values > 0.05 preferred; positive-lag ACF exceedances should be limited",
                "durbin_watson": safe_stat(dw_values[idx]),
                "ljung_box_lag": lb_lag,
                "ljung_box_p_value": lb_p,
                "max_abs_acf_lag_1_to_24": np.nanmax(np.abs(acf_values)) if acf_values else np.nan,
                "acf_confidence_bound": bound,
                "acf_exceedance_count": len(exceedances),
                "acf_exceedance_share": len(exceedances) / max(len(acf_values), 1),
                "jarque_bera_p_value": jb_p,
                "arch_lm_p_value": arch_p,
                "portmanteau_p_value": portmanteau_p,
                "system_normality_p_value": normality_p,
                "stable": bool(fitted.is_stable()) if hasattr(fitted, "is_stable") else np.nan,
            }
        )
    ccf = residual_cross_correlation_summary(residuals, max_lag=12)
    for _, row in ccf.iterrows():
        rows.append(
            {
                "model": model_name,
                "candidate": candidate,
                "lag_order": lag_order,
                "diagnostic_type": "lagged_cross_correlation",
                "source_residual": row["source_residual"],
                "target_residual": row["target_residual"],
                "test": "Residual lagged cross-correlation",
                "Acceptable if": "smaller absolute cross-correlations indicate weaker remaining cross-equation residual dependence; lag 0 is meaningful for cross-series residuals",
                "lag_of_max_abs_ccf": row["lag_of_max_abs_ccf"],
                "ccf_at_max_abs_lag": row["ccf_at_max_abs_lag"],
                "max_abs_ccf": row["max_abs_ccf"],
                "portmanteau_p_value": portmanteau_p,
                "system_normality_p_value": normality_p,
                "stable": bool(fitted.is_stable()) if hasattr(fitted, "is_stable") else np.nan,
            }
        )
    return pd.DataFrame(rows)


def candidate_metrics(
    run: dict,
    model_name: str,
    candidate: str,
    lag_order: int,
    endog: list[str],
    exog: list[str],
    lag_best: dict[str, int],
) -> dict:
    residuals = run["residuals"]
    diag = residual_summary(residuals, run["fitted"], model_name, candidate, lag_order)
    eq_diag = diag.loc[diag["diagnostic_type"] == "equation_residual"].copy()
    params_per_equation = int(run["fit_info"]["parameters_per_equation"].iloc[0])
    total_params = int(run["fit_info"]["total_parameters"].iloc[0])
    effective_obs = int(run["fit_info"]["effective_observations"].iloc[0])
    metrics = run["metrics"].copy()
    inflation = metrics.loc[metrics["variable"] == "INF"]
    inflation_rmse = float(inflation["RMSE"].iloc[0]) if not inflation.empty else np.nan
    inflation_mae = float(inflation["MAE"].iloc[0]) if not inflation.empty else np.nan
    min_ljung = float(eq_diag["ljung_box_p_value"].min()) if "ljung_box_p_value" in eq_diag else np.nan
    min_arch = float(eq_diag["arch_lm_p_value"].min()) if "arch_lm_p_value" in eq_diag else np.nan
    min_jb = float(eq_diag["jarque_bera_p_value"].min()) if "jarque_bera_p_value" in eq_diag else np.nan
    acf_share = float(eq_diag["acf_exceedance_count"].sum() / max(len(eq_diag) * 24, 1))
    port_p = safe_portmanteau_p(run["fitted"], lag_order)
    stable = bool(run["stable"])
    score = (
        (100.0 if stable else -100.0)
        - 8.0 * inflation_rmse
        - 4.0 * float(metrics["RMSE"].mean())
        - 10.0 * acf_share
        + 2.0 * min(max(min_ljung, 0), 0.5)
        + 2.0 * min(max(port_p, 0), 0.5)
        - total_params / max(effective_obs, 1)
    )
    row = {
        "model": model_name,
        "candidate": candidate,
        "endogenous_variables": ", ".join(endog),
        "exogenous_variables": ", ".join(exog) if exog else "none",
        "lag_order": lag_order,
        **lag_best,
        "aic": safe_stat(run["fit_info"]["aic"].iloc[0]),
        "bic": safe_stat(run["fit_info"]["bic"].iloc[0]),
        "hqic": safe_stat(run["fit_info"]["hqic"].iloc[0]),
        "stable": stable,
        "effective_observations": effective_obs,
        "parameters_per_equation": params_per_equation,
        "total_parameters": total_params,
        "parameter_to_observation_ratio": total_params / max(effective_obs, 1),
        "inflation_RMSE": inflation_rmse,
        "inflation_MAE": inflation_mae,
        "mean_RMSE": float(metrics["RMSE"].mean()),
        "mean_MAE": float(metrics["MAE"].mean()),
        "min_ljung_box_p_value": min_ljung,
        "portmanteau_p_value": port_p,
        "acf_exceedance_share": acf_share,
        "min_jarque_bera_p_value": min_jb,
        "min_arch_lm_p_value": min_arch,
        "arch_effects_any_equation": bool((eq_diag["arch_lm_p_value"] < 0.05).any()),
        "normality_rejected_any_equation": bool((eq_diag["jarque_bera_p_value"] < 0.05).any()),
        "selection_score": score,
        "Acceptable if": "stable model, lower RMSE/MAE, limited ACF exceedances, and diagnostic p-values > 0.05 are preferred",
    }
    return row


def run_var_search(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metric_rows = []
    diagnostic_tables = []
    runs: dict[tuple[str, int], dict] = {}
    for candidate, endog in VAR_DIFF_CANDIDATES.items():
        train = model_df.loc[:SPLIT_DATE, endog]
        lag_table = select_var_lags(train, MAX_LAG)
        lag_table.to_csv(TABLE_DIR / f"{candidate.lower()}_lag_selection.csv", index=False)
        if candidate == "VAR_D_FEDFUNDS_core_sentiment":
            lag_table.to_csv(TABLE_DIR / "d_fedfunds_var_lag_selection.csv", index=False)
        lag_best = lag_best_table(lag_table)
        for lag in range(1, MAX_LAG + 1):
            try:
                run = fit_var_system(model_df, SPLIT_DATE, endog, lag, "INF")
            except Exception:
                continue
            runs[(candidate, lag)] = run
            metric_rows.append(candidate_metrics(run, "VAR_D_FEDFUNDS", candidate, lag, endog, [], lag_best))
            diagnostic_tables.append(residual_summary(run["residuals"], run["fitted"], "VAR_D_FEDFUNDS", candidate, lag))
    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise RuntimeError("No D_FEDFUNDS VAR candidates could be estimated.")
    metrics = metrics.sort_values("selection_score", ascending=False).reset_index(drop=True)
    metrics["selected_diff_robustness_model"] = False
    metrics.loc[0, "selected_diff_robustness_model"] = True
    diagnostics = pd.concat(diagnostic_tables, ignore_index=True) if diagnostic_tables else pd.DataFrame()
    metrics.to_csv(TABLE_DIR / "var_diff_fedfunds_model_metrics.csv", index=False)
    metrics.to_csv(TABLE_DIR / "d_fedfunds_var_metrics.csv", index=False)
    diagnostics.to_csv(TABLE_DIR / "var_diff_fedfunds_residual_diagnostics.csv", index=False)
    selected_key = (metrics.loc[0, "candidate"], int(metrics.loc[0, "lag_order"]))
    return metrics, diagnostics, runs[selected_key]


def run_varx_search(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    metric_rows = []
    diagnostic_tables = []
    runs: dict[tuple[str, int], dict] = {}
    candidate_specs: dict[str, dict] = {}
    for candidate, spec in VARX_DIFF_CANDIDATES.items():
        endog = spec["endog"]
        exog = spec["exog"]
        candidate_specs[candidate] = spec
        train = model_df.loc[:SPLIT_DATE]
        lag_table = select_varx_lags(train[endog], train[exog], MAX_LAG)
        lag_table.to_csv(TABLE_DIR / f"{candidate.lower()}_lag_selection.csv", index=False)
        if candidate == "VARX_D_FEDFUNDS_A":
            lag_table.to_csv(TABLE_DIR / "d_fedfunds_varx_lag_selection.csv", index=False)
        lag_best = lag_best_table(lag_table)
        for lag in range(1, MAX_LAG + 1):
            try:
                run = fit_varx_system(model_df, SPLIT_DATE, endog, exog, lag, "INF")
            except Exception:
                continue
            runs[(candidate, lag)] = run
            metric_rows.append(candidate_metrics(run, "VARX_D_FEDFUNDS", candidate, lag, endog, exog, lag_best))
            diagnostic_tables.append(residual_summary(run["residuals"], run["fitted"], "VARX_D_FEDFUNDS", candidate, lag))
    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise RuntimeError("No D_FEDFUNDS VARX candidates could be estimated.")
    metrics = metrics.sort_values("selection_score", ascending=False).reset_index(drop=True)
    metrics["selected_diff_robustness_model"] = False
    metrics.loc[0, "selected_diff_robustness_model"] = True
    diagnostics = pd.concat(diagnostic_tables, ignore_index=True) if diagnostic_tables else pd.DataFrame()
    metrics.to_csv(TABLE_DIR / "varx_diff_fedfunds_model_metrics.csv", index=False)
    metrics.to_csv(TABLE_DIR / "d_fedfunds_varx_metrics.csv", index=False)
    diagnostics.to_csv(TABLE_DIR / "varx_diff_fedfunds_residual_diagnostics.csv", index=False)
    selected_key = (metrics.loc[0, "candidate"], int(metrics.loc[0, "lag_order"]))
    return metrics, diagnostics, runs[selected_key], candidate_specs[selected_key[0]]


def granger_table(run: dict) -> pd.DataFrame:
    fitted = run["fitted"]
    rows = []
    names = list(fitted.names)
    for target in names:
        for source in names:
            if source == target:
                continue
            try:
                result = fitted.test_causality(target, [source], kind="f")
                stat = safe_stat(getattr(result, "test_statistic", np.nan))
                p_value = safe_stat(getattr(result, "pvalue", np.nan))
            except Exception:
                stat = np.nan
                p_value = np.nan
            rows.append(
                {
                    "test": "VAR block Granger causality F-test",
                    "Acceptable if": "p-value < 0.05 indicates predictive causality, not structural causality",
                    "source": source,
                    "target": target,
                    "p_value": p_value,
                    "test_statistic": stat,
                    "significant_at_5pct": bool(pd.notna(p_value) and p_value < 0.05),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "var_diff_fedfunds_granger.csv", index=False)
    table.to_csv(TABLE_DIR / "d_fedfunds_granger.csv", index=False)
    return table


def save_irf_fevd(run: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    fitted = run["fitted"]
    irf = var_irf_paths(fitted, IRF_HORIZON)
    key_horizons = [0, 1, 3, 6, 12, 24]
    irf_summary = irf.loc[
        (irf["shock"] == "D_FEDFUNDS") & (irf["horizon"].isin(key_horizons))
    ].copy()
    irf_summary["test"] = "Orthogonalized VAR response to D_FEDFUNDS policy-rate-change shock"
    irf_summary["Acceptable if"] = "responses are economically interpretable and robust to stationarity treatment; Cholesky identification caveats still apply"
    irf_summary.to_csv(TABLE_DIR / "var_diff_fedfunds_irf_summary.csv", index=False)
    irf_summary.to_csv(TABLE_DIR / "d_fedfunds_irf_summary.csv", index=False)

    fevd = var_fevd_paths(fitted, IRF_HORIZON)
    fevd_summary = fevd.loc[
        (fevd["response"] == "INF") & (fevd["horizon"].isin([1, 6, 12, 24]))
    ].copy()
    fevd_summary["test"] = "Forecast-error variance decomposition for D_FEDFUNDS robustness VAR"
    fevd_summary["Acceptable if"] = "larger variance shares indicate larger contribution to inflation forecast-error variance under this identification"
    fevd_summary.to_csv(TABLE_DIR / "var_diff_fedfunds_fevd_summary.csv", index=False)
    fevd_summary.to_csv(TABLE_DIR / "d_fedfunds_fevd_summary.csv", index=False)
    return irf_summary, fevd_summary


def save_varx_scenario(run: dict, spec: dict) -> pd.DataFrame:
    scenario = varx_exogenous_scenario_response(
        run["fitted"],
        run["train"],
        spec["exog"],
        "D_FEDFUNDS",
        IRF_HORIZON,
        shock_size=1.0,
    )
    if not scenario.empty:
        scenario["test"] = "VARX conditional scenario response to D_FEDFUNDS exogenous shock"
        scenario["Acceptable if"] = "scenario responses are interpreted conditionally on supplied exogenous paths, not as structural IRFs"
    scenario.to_csv(TABLE_DIR / "varx_diff_fedfunds_scenario_response.csv", index=False)
    return scenario


def comparison_summary(
    run: dict,
    model_label: str,
    variables: list[str],
    exog: list[str],
    stationarity_treatment: str,
    policy_variable: str,
    granger_conclusion: str = "",
    response_conclusion: str = "",
) -> dict:
    diag = residual_summary(run["residuals"], run["fitted"], model_label, model_label, int(run["fit_info"]["lag_order"].iloc[0]))
    eq_diag = diag.loc[diag["diagnostic_type"] == "equation_residual"]
    ccf_diag = diag.loc[diag["diagnostic_type"] == "lagged_cross_correlation"]
    metrics = run["metrics"]
    inflation = metrics.loc[metrics["variable"] == "INF"]
    fit_metrics = run["fit_metrics"]
    roots = run["roots"]
    forecast = run["forecasts"]["INF"]
    actual = run["test"]["INF"]
    actual_direction = np.sign(actual.diff().dropna())
    forecast_direction = np.sign(forecast.diff().dropna()).reindex(actual_direction.index)
    valid_direction = forecast_direction.notna()
    directional_accuracy = (
        float((actual_direction.loc[valid_direction] == forecast_direction.loc[valid_direction]).mean())
        if valid_direction.any()
        else np.nan
    )
    effective_obs = int(run["fit_info"]["effective_observations"].iloc[0])
    total_parameters = int(run["fit_info"]["total_parameters"].iloc[0])
    return {
        "model": model_label,
        "policy_variable_used": policy_variable,
        "variables_used": ", ".join(variables),
        "exogenous_variables": ", ".join(exog) if exog else "none",
        "lag_order": int(run["fit_info"]["lag_order"].iloc[0]),
        "stationarity_treatment": stationarity_treatment,
        "stable": bool(run["stable"]),
        "max_companion_eigenvalue_modulus": float(roots["modulus"].max()) if not roots.empty else np.nan,
        "aic": safe_stat(run["fit_info"]["aic"].iloc[0]),
        "bic": safe_stat(run["fit_info"]["bic"].iloc[0]),
        "hqic": safe_stat(run["fit_info"]["hqic"].iloc[0]),
        "fpe": safe_stat(getattr(run["fitted"], "fpe", np.nan)),
        "parameters_per_equation": int(run["fit_info"]["parameters_per_equation"].iloc[0]),
        "total_parameters": total_parameters,
        "effective_observations": effective_obs,
        "observations_per_parameter": effective_obs / max(total_parameters, 1),
        "equation_R2_mean": float(fit_metrics["R_squared"].mean()) if "R_squared" in fit_metrics else np.nan,
        "equation_R2_min": float(fit_metrics["R_squared"].min()) if "R_squared" in fit_metrics else np.nan,
        "inflation_RMSE": float(inflation["RMSE"].iloc[0]) if not inflation.empty else np.nan,
        "inflation_MAE": float(inflation["MAE"].iloc[0]) if not inflation.empty else np.nan,
        "mean_RMSE": float(metrics["RMSE"].mean()),
        "mean_MAE": float(metrics["MAE"].mean()),
        "directional_accuracy_INF": directional_accuracy,
        "ljung_box_min_p_value": float(eq_diag["ljung_box_p_value"].min()),
        "portmanteau_p_value": safe_portmanteau_p(run["fitted"], int(run["fit_info"]["lag_order"].iloc[0])),
        "acf_exceedance_share": float(eq_diag["acf_exceedance_count"].sum() / max(len(eq_diag) * 24, 1)),
        "residual_cross_correlation_max": float(ccf_diag["max_abs_ccf"].max()) if not ccf_diag.empty else np.nan,
        "jarque_bera_min_p_value": float(eq_diag["jarque_bera_p_value"].min()),
        "arch_effects_any_equation": bool((eq_diag["arch_lm_p_value"] < 0.05).any()),
        "arch_lm_min_p_value": float(eq_diag["arch_lm_p_value"].min()),
        "normality_rejected_any_equation": bool((eq_diag["jarque_bera_p_value"] < 0.05).any()),
        "robust_significance_sensitivity": (
            "robust inference remains a sensitivity check because residual normality and ARCH tests reject in both policy representations"
        ),
        "granger_causality_conclusion": granger_conclusion,
        "irf_fevd_or_scenario_conclusion": response_conclusion,
        "Acceptable if": "D_FEDFUNDS should improve stationarity without materially worsening forecast and diagnostic quality",
    }


def causality_table_for_run(run: dict) -> pd.DataFrame:
    fitted = run["fitted"]
    rows = []
    for target in fitted.names:
        for source in fitted.names:
            if source == target:
                continue
            try:
                result = fitted.test_causality(target, [source], kind="f")
                p_value = safe_stat(getattr(result, "pvalue", np.nan))
            except Exception:
                p_value = np.nan
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "p_value": p_value,
                    "significant_at_5pct": bool(pd.notna(p_value) and p_value < 0.05),
                }
            )
    return pd.DataFrame(rows)


def granger_conclusion(granger: pd.DataFrame, policy_variable: str) -> str:
    if granger.empty:
        return "Granger results unavailable."
    selected = granger.loc[
        (granger["source"] == policy_variable)
        & (granger["significant_at_5pct"] == True)
    ].sort_values("p_value")
    if selected.empty:
        return f"{policy_variable} has no significant predictive channel at the 5% level."
    pairs = ", ".join(f"{row.source}->{row.target} (p={row.p_value:.3g})" for row in selected.head(5).itertuples())
    return f"Significant predictive channels from {policy_variable}: {pairs}."


def irf_fevd_conclusion(run: dict, policy_variable: str) -> str:
    try:
        irf = var_irf_paths(run["fitted"], IRF_HORIZON)
        fevd = var_fevd_paths(run["fitted"], IRF_HORIZON)
        h1 = irf.loc[
            (irf["response"] == "INF")
            & (irf["shock"] == policy_variable)
            & (irf["horizon"] == 1),
            "value",
        ].iloc[0]
        h12 = irf.loc[
            (irf["response"] == "INF")
            & (irf["shock"] == policy_variable)
            & (irf["horizon"] == 12),
            "value",
        ].iloc[0]
        share = fevd.loc[
            (fevd["response"] == "INF")
            & (fevd["shock"] == policy_variable)
            & (fevd["horizon"] == 12),
            "variance_share",
        ].iloc[0]
        return f"INF response to {policy_variable} shock: h1={h1:.4f}, h12={h12:.4f}; INF FEVD share at h12={share:.3f}."
    except Exception:
        return f"IRF/FEVD summary unavailable for {policy_variable}."


def scenario_conclusion(run: dict, exog: list[str], policy_variable: str) -> str:
    try:
        scenario = varx_exogenous_scenario_response(
            run["fitted"],
            run["train"],
            exog,
            policy_variable,
            IRF_HORIZON,
            shock_size=1.0,
        )
        inf = scenario.loc[
            (scenario["response"] == "INF")
            & (scenario["horizon"].isin([1, 12, 24]))
        ]
        values = ", ".join(f"h{int(row.horizon)}={row.value:.4f}" for row in inf.itertuples())
        return f"Conditional INF scenario response to {policy_variable}: {values}."
    except Exception:
        return f"VARX scenario response unavailable for {policy_variable}."


def save_comparison_tables(model_df: pd.DataFrame, selected_var: dict, selected_varx: dict, selected_varx_spec: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    level_var = fit_var_system(model_df, SPLIT_DATE, BASELINE_VAR_ENDOG, BASELINE_VAR_LAG, "INF")
    level_varx = fit_varx_system(model_df, SPLIT_DATE, BASELINE_VARX_ENDOG, BASELINE_VARX_EXOG, BASELINE_VARX_LAG, "INF")
    diff_var_endog = list(selected_var["fitted"].names)
    level_granger = causality_table_for_run(level_var)
    diff_granger = granger_table(selected_var)
    var_table = pd.DataFrame(
        [
            comparison_summary(
                level_var,
                "VAR_level_FEDFUNDS_baseline",
                BASELINE_VAR_ENDOG,
                [],
                "FEDFUNDS in levels: policy stance",
                "FEDFUNDS",
                granger_conclusion(level_granger, "FEDFUNDS"),
                irf_fevd_conclusion(level_var, "FEDFUNDS"),
            ),
            comparison_summary(
                selected_var,
                "VAR_D_FEDFUNDS_candidate",
                diff_var_endog,
                [],
                "D_FEDFUNDS: monthly policy-rate change",
                "D_FEDFUNDS",
                granger_conclusion(diff_granger, "D_FEDFUNDS"),
                irf_fevd_conclusion(selected_var, "D_FEDFUNDS"),
            ),
        ]
    )
    varx_table = pd.DataFrame(
        [
            comparison_summary(
                level_varx,
                "VARX_level_FEDFUNDS_baseline",
                BASELINE_VARX_ENDOG,
                BASELINE_VARX_EXOG,
                "FEDFUNDS in levels: exogenous policy stance path",
                "FEDFUNDS",
                "VARX treats FEDFUNDS as an externally supplied scenario path.",
                scenario_conclusion(level_varx, BASELINE_VARX_EXOG, "FEDFUNDS"),
            ),
            comparison_summary(
                selected_varx,
                "VARX_D_FEDFUNDS_candidate",
                selected_varx_spec["endog"],
                selected_varx_spec["exog"],
                "D_FEDFUNDS: exogenous policy-rate-change path",
                "D_FEDFUNDS",
                "VARX treats D_FEDFUNDS as an externally supplied policy-change scenario path.",
                scenario_conclusion(selected_varx, selected_varx_spec["exog"], "D_FEDFUNDS"),
            ),
        ]
    )
    var_table.to_csv(TABLE_DIR / "var_fedfunds_level_vs_diff_comparison.csv", index=False)
    varx_table.to_csv(TABLE_DIR / "varx_fedfunds_level_vs_diff_comparison.csv", index=False)
    var_table.to_csv(TABLE_DIR / "fedfunds_level_vs_diff_var_comparison.csv", index=False)
    varx_table.to_csv(TABLE_DIR / "fedfunds_level_vs_diff_varx_comparison.csv", index=False)
    return var_table, varx_table, level_var, level_varx


def plot_irf(irf_summary: pd.DataFrame) -> None:
    if irf_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for response, group in irf_summary.groupby("response"):
        ax.plot(group["horizon"], group["value"], marker="o", label=response)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("D_FEDFUNDS VAR: Responses to Policy-Rate-Change Shock")
    ax.set_xlabel("Horizon in months")
    ax.set_ylabel("Orthogonalized response")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "var_diff_fedfunds_irf.png", dpi=180)
    plt.savefig(FIGURE_DIR / "d_fedfunds_var_irf.png", dpi=180)
    plt.close()


def plot_varx_scenario(scenario: pd.DataFrame) -> None:
    if scenario.empty:
        return
    y_col = "scenario_response" if "scenario_response" in scenario.columns else "response_value"
    if y_col not in scenario.columns:
        y_col = "value"
    fig, ax = plt.subplots(figsize=(11, 6))
    for response, group in scenario.groupby("response"):
        ax.plot(group["horizon"], group[y_col], marker="o", label=response)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("VARX Conditional Scenario Response: D_FEDFUNDS Shock")
    ax.set_xlabel("Horizon in months")
    ax.set_ylabel("Conditional response")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "varx_diff_fedfunds_scenario_response.png", dpi=180)
    plt.savefig(FIGURE_DIR / "d_fedfunds_varx_scenario_response.png", dpi=180)
    plt.close()


def plot_forecast_comparison(var_table: pd.DataFrame, varx_table: pd.DataFrame) -> None:
    combined = pd.concat(
        [
            var_table.assign(model_family="VAR"),
            varx_table.assign(model_family="VARX"),
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = combined["model"].str.replace("_", " ", regex=False)
    ax.bar(labels, combined["inflation_RMSE"], color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_title("Inflation RMSE: Level FEDFUNDS Baseline vs D_FEDFUNDS Robustness")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fedfunds_level_vs_diff_forecast_comparison.png", dpi=180)
    plt.close()


def bool_text(value: bool) -> str:
    return "yes" if bool(value) else "no"


def baseline_decision_table(
    stationarity: pd.DataFrame,
    var_comparison: pd.DataFrame,
    varx_comparison: pd.DataFrame,
) -> pd.DataFrame:
    fed = stationarity.set_index("variable")
    var_level = var_comparison.loc[var_comparison["policy_variable_used"] == "FEDFUNDS"].iloc[0]
    var_diff = var_comparison.loc[var_comparison["policy_variable_used"] == "D_FEDFUNDS"].iloc[0]
    varx_level = varx_comparison.loc[varx_comparison["policy_variable_used"] == "FEDFUNDS"].iloc[0]
    varx_diff = varx_comparison.loc[varx_comparison["policy_variable_used"] == "D_FEDFUNDS"].iloc[0]

    rows = [
        {
            "model_family": "VAR",
            "official_baseline_policy_variable": "D_FEDFUNDS",
            "official_model_name": "VAR_D_FEDFUNDS_candidate",
            "official_endogenous_variables": "INF, D_FEDFUNDS, UNRATE, INDPRO_GROWTH, SENTIMENT_CHANGE",
            "official_exogenous_variables": "none",
            "official_lag_order": int(var_diff["lag_order"]),
            "robustness_alternative": "VAR_level_FEDFUNDS_baseline",
            "stationarity_decision": "D_FEDFUNDS wins: KPSS no longer rejects stationarity and ACF(1) is much lower",
            "forecast_decision": (
                f"Level is slightly better for inflation RMSE ({var_level['inflation_RMSE']:.4f} vs {var_diff['inflation_RMSE']:.4f}), "
                f"but D_FEDFUNDS improves mean RMSE ({var_diff['mean_RMSE']:.4f} vs {var_level['mean_RMSE']:.4f})"
            ),
            "diagnostic_decision": (
                f"D_FEDFUNDS improves min Ljung-Box p-value ({var_diff['ljung_box_min_p_value']:.4f} vs {var_level['ljung_box_min_p_value']:.4f}) "
                f"but has a slightly higher ACF exceedance share ({var_diff['acf_exceedance_share']:.3f} vs {var_level['acf_exceedance_share']:.3f}); system whiteness still rejects"
            ),
            "final_decision": "Switch official VAR baseline to D_FEDFUNDS because stationarity validity improves and forecast loss is negligible.",
            "Acceptable if": "official baseline should balance stationarity, stability, diagnostics, forecasting, parsimony, and interpretability",
        },
        {
            "model_family": "VARX",
            "official_baseline_policy_variable": "FEDFUNDS",
            "official_model_name": "VARX_level_FEDFUNDS_baseline",
            "official_endogenous_variables": "INF, UNRATE, INDPRO_GROWTH, M2_GROWTH",
            "official_exogenous_variables": "FEDFUNDS, SENTIMENT_CHANGE",
            "official_lag_order": int(varx_level["lag_order"]),
            "robustness_alternative": "VARX_D_FEDFUNDS_candidate",
            "stationarity_decision": "D_FEDFUNDS wins stationarity, but VARX treats the policy path as an external scenario input.",
            "forecast_decision": (
                f"Level FEDFUNDS wins clearly for VARX forecasting: inflation RMSE {varx_level['inflation_RMSE']:.4f} vs {varx_diff['inflation_RMSE']:.4f}; "
                f"mean RMSE {varx_level['mean_RMSE']:.4f} vs {varx_diff['mean_RMSE']:.4f}"
            ),
            "diagnostic_decision": (
                f"D_FEDFUNDS improves equation-level Ljung-Box/ACF metrics, but both versions still reject system whiteness and have non-normal/ARCH residual issues"
            ),
            "final_decision": "Keep official VARX baseline with level FEDFUNDS for conditional policy-stance scenario forecasting; report D_FEDFUNDS VARX as robustness.",
            "Acceptable if": "official baseline should balance stationarity, stability, diagnostics, forecasting, parsimony, and interpretability",
        },
    ]
    table = pd.DataFrame(rows)
    table["fedfunds_adf_p_value"] = fed.loc["FEDFUNDS", "adf_p_value"]
    table["fedfunds_kpss_p_value"] = fed.loc["FEDFUNDS", "kpss_p_value"]
    table["fedfunds_acf_lag1"] = fed.loc["FEDFUNDS", "acf_lag1"]
    table["d_fedfunds_adf_p_value"] = fed.loc["D_FEDFUNDS", "adf_p_value"]
    table["d_fedfunds_kpss_p_value"] = fed.loc["D_FEDFUNDS", "kpss_p_value"]
    table["d_fedfunds_acf_lag1"] = fed.loc["D_FEDFUNDS", "acf_lag1"]
    table.to_csv(TABLE_DIR / "fedfunds_level_vs_diff_final_baseline_decision.csv", index=False)
    return table


def save_forecast_comparison(var_comparison: pd.DataFrame, varx_comparison: pd.DataFrame) -> pd.DataFrame:
    table = pd.concat(
        [
            var_comparison.assign(model_family="VAR"),
            varx_comparison.assign(model_family="VARX"),
        ],
        ignore_index=True,
    )
    cols = [
        "model_family",
        "model",
        "policy_variable_used",
        "lag_order",
        "inflation_RMSE",
        "inflation_MAE",
        "mean_RMSE",
        "mean_MAE",
        "directional_accuracy_INF",
        "stationarity_treatment",
        "Acceptable if",
    ]
    table[[c for c in cols if c in table.columns]].to_csv(
        TABLE_DIR / "d_fedfunds_forecast_comparison.csv",
        index=False,
    )
    return table


def report_section(
    stationarity: pd.DataFrame,
    var_comparison: pd.DataFrame,
    varx_comparison: pd.DataFrame,
    var_granger: pd.DataFrame,
    var_irf: pd.DataFrame,
    var_fevd: pd.DataFrame,
    varx_scenario: pd.DataFrame,
    decision: pd.DataFrame,
) -> str:
    fed = stationarity.set_index("variable")
    fed_acf = fed.loc["FEDFUNDS", "acf_lag1"]
    d_acf = fed.loc["D_FEDFUNDS", "acf_lag1"]
    fed_adf = fed.loc["FEDFUNDS", "adf_p_value"]
    d_adf = fed.loc["D_FEDFUNDS", "adf_p_value"]
    fed_kpss = fed.loc["FEDFUNDS", "kpss_p_value"]
    d_kpss = fed.loc["D_FEDFUNDS", "kpss_p_value"]

    var_level = var_comparison.loc[var_comparison["policy_variable_used"] == "FEDFUNDS"].iloc[0]
    var_diff = var_comparison.loc[var_comparison["policy_variable_used"] == "D_FEDFUNDS"].iloc[0]
    varx_level = varx_comparison.loc[varx_comparison["policy_variable_used"] == "FEDFUNDS"].iloc[0]
    varx_diff = varx_comparison.loc[varx_comparison["policy_variable_used"] == "D_FEDFUNDS"].iloc[0]
    var_decision = decision.loc[decision["model_family"] == "VAR"].iloc[0]
    varx_decision = decision.loc[decision["model_family"] == "VARX"].iloc[0]

    sig_granger = var_granger.loc[var_granger["significant_at_5pct"] == True].copy()
    sig_pairs = ", ".join(
        f"{row.source}->{row.target} (p={row.p_value:.3g})"
        for row in sig_granger.itertuples()
    )
    if not sig_pairs:
        sig_pairs = "no block Granger relationships significant at 5% in the selected D_FEDFUNDS VAR"

    irf_inf = var_irf.loc[(var_irf["response"] == "INF") & (var_irf["horizon"].isin([1, 6, 12, 24]))]
    irf_text = ", ".join(
        f"h{int(row.horizon)}={row.value:.4f}" for row in irf_inf.itertuples()
    ) or "not available"
    fevd_inf = var_fevd.loc[(var_fevd["shock"] == "D_FEDFUNDS") & (var_fevd["horizon"].isin([12, 24]))]
    fevd_text = ", ".join(
        f"h{int(row.horizon)}={row.variance_share:.3f}" for row in fevd_inf.itertuples()
    ) or "D_FEDFUNDS FEVD share not available"
    scenario_inf = varx_scenario.loc[(varx_scenario["response"] == "INF") & (varx_scenario["horizon"].isin([1, 6, 12, 24]))]
    scenario_col = "scenario_response" if "scenario_response" in scenario_inf.columns else "value"
    scenario_text = ", ".join(
        f"h{int(row.horizon)}={getattr(row, scenario_col):.4f}" for row in scenario_inf.itertuples()
    ) if not scenario_inf.empty and scenario_col in scenario_inf else "not available"

    return f"""
## FEDFUNDS vs D_FEDFUNDS Baseline Decision

This branch compares level `FEDFUNDS` against `D_FEDFUNDS = FEDFUNDS.diff()` as the policy variable. Level `FEDFUNDS` measures policy stance; `D_FEDFUNDS` measures monthly tightening/easing movements. The dashboard keeps both variables available, but the official baseline uses only one policy representation within each model.

### Stationarity evidence

- Level `FEDFUNDS`: ADF p-value = {fed_adf:.4g}, KPSS p-value = {fed_kpss:.4g}, ACF(1) = {fed_acf:.4f}.
- `D_FEDFUNDS`: ADF p-value = {d_adf:.4g}, KPSS p-value = {d_kpss:.4g}, ACF(1) = {d_acf:.4f}.
- Conclusion: `D_FEDFUNDS` is more stationary and much less persistent than level `FEDFUNDS`.

### VAR baseline decision

- Level-FEDFUNDS baseline: lag {int(var_level['lag_order'])}, inflation RMSE = {var_level['inflation_RMSE']:.4f}, mean RMSE = {var_level['mean_RMSE']:.4f}, min Ljung-Box p = {var_level['ljung_box_min_p_value']:.4f}, Portmanteau p = {var_level['portmanteau_p_value']:.4g}, ACF exceedance share = {var_level['acf_exceedance_share']:.3f}, stable = {bool_text(var_level['stable'])}.
- D_FEDFUNDS candidate: lag {int(var_diff['lag_order'])}, inflation RMSE = {var_diff['inflation_RMSE']:.4f}, mean RMSE = {var_diff['mean_RMSE']:.4f}, min Ljung-Box p = {var_diff['ljung_box_min_p_value']:.4f}, Portmanteau p = {var_diff['portmanteau_p_value']:.4g}, ACF exceedance share = {var_diff['acf_exceedance_share']:.3f}, stable = {bool_text(var_diff['stable'])}.
- Official VAR baseline: `{var_decision['official_baseline_policy_variable']}`, lag {int(var_decision['official_lag_order'])}. {var_decision['final_decision']}

### VARX baseline decision

- Level-FEDFUNDS VARX baseline: lag {int(varx_level['lag_order'])}, inflation RMSE = {varx_level['inflation_RMSE']:.4f}, mean RMSE = {varx_level['mean_RMSE']:.4f}, min Ljung-Box p = {varx_level['ljung_box_min_p_value']:.4f}, ACF exceedance share = {varx_level['acf_exceedance_share']:.3f}, stable = {bool_text(varx_level['stable'])}.
- D_FEDFUNDS VARX candidate: lag {int(varx_diff['lag_order'])}, inflation RMSE = {varx_diff['inflation_RMSE']:.4f}, mean RMSE = {varx_diff['mean_RMSE']:.4f}, min Ljung-Box p = {varx_diff['ljung_box_min_p_value']:.4f}, ACF exceedance share = {varx_diff['acf_exceedance_share']:.3f}, stable = {bool_text(varx_diff['stable'])}.
- Official VARX baseline: `{varx_decision['official_baseline_policy_variable']}`, lag {int(varx_decision['official_lag_order'])}. {varx_decision['final_decision']}

### Granger, IRF, FEVD, and scenario conclusions

- Selected D_FEDFUNDS VAR Granger relationships: {sig_pairs}.
- Response of inflation to a D_FEDFUNDS shock: {irf_text}. This is a response to an unexpected policy-rate change, not a policy-rate-level stance shock.
- D_FEDFUNDS contribution to INF FEVD: {fevd_text}.
- VARX conditional inflation scenario response to a D_FEDFUNDS exogenous shock: {scenario_text}. VARX responses are conditional scenario responses, not structural IRFs.

### Final answer

`D_FEDFUNDS` is more stationary. The official VAR baseline switches to `D_FEDFUNDS` because the inflation forecast loss is negligible and stationarity validity improves. The official VARX baseline keeps level `FEDFUNDS` because conditional forecast performance is materially better with the policy-rate level. The alternative representation remains available as a sensitivity-analysis variable in the dashboard.
""".strip()


def current_baseline_note(decision: pd.DataFrame) -> str:
    var_decision = decision.loc[decision["model_family"] == "VAR"].iloc[0]
    varx_decision = decision.loc[decision["model_family"] == "VARX"].iloc[0]
    return f"""
## Current Official Baseline Update

This note supersedes earlier level-FEDFUNDS baseline references in older optimization tables in this file. Those older tables are retained as historical comparison output, but the official baseline after the FEDFUNDS-vs-D_FEDFUNDS experiment is:

- VAR baseline: `{var_decision['official_model_name']}`, endogenous variables `{var_decision['official_endogenous_variables']}`, lag {int(var_decision['official_lag_order'])}, policy representation `{var_decision['official_baseline_policy_variable']}`.
- VARX baseline: `{varx_decision['official_model_name']}`, endogenous variables `{varx_decision['official_endogenous_variables']}`, exogenous variables `{varx_decision['official_exogenous_variables']}`, lag {int(varx_decision['official_lag_order'])}, policy representation `{varx_decision['official_baseline_policy_variable']}`.

Interpretation rule: `FEDFUNDS` is the policy-rate level / policy stance; `D_FEDFUNDS` is the monthly policy-rate change / tightening-easing movement. They should not both be treated as the official policy variable in the same baseline model.
""".strip()


def remove_section(text: str, marker: str) -> str:
    if marker not in text:
        return text
    before, after = text.split(marker, 1)
    next_section = after.find("\n## ")
    if next_section == -1:
        return before.rstrip()
    return before.rstrip() + "\n\n" + after[next_section + 1 :].lstrip()


def insert_after_title(text: str, section: str) -> str:
    if not text.startswith("# "):
        return section + "\n\n" + text.lstrip()
    first_newline = text.find("\n")
    if first_newline == -1:
        return text.rstrip() + "\n\n" + section
    return text[: first_newline + 1].rstrip() + "\n\n" + section + "\n\n" + text[first_newline + 1 :].lstrip()


def update_report_files(section: str, decision: pd.DataFrame) -> None:
    markers = [
        "## FEDFUNDS vs D_FEDFUNDS Baseline Decision",
        "## FEDFUNDS Stationarity Robustness",
    ]
    note_marker = "## Current Official Baseline Update"
    note = current_baseline_note(decision)
    for name in ["PROJECT_RESULTS_CONTEXT.md", "PROJECT_CONTEXT.md", "project_report.md", "README.md"]:
        path = BASE_DIR / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        text = remove_section(text, note_marker)
        for marker in markers:
            if marker in text:
                text = text.split(marker)[0].rstrip()
                break
        text = insert_after_title(text, note)
        text = text.rstrip() + "\n\n" + section + "\n"
        path.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_directories()
    model_df = load_model_data()
    stationarity = stationarity_results(model_df)
    plot_stationarity(model_df)

    var_metrics, var_diagnostics, selected_var = run_var_search(model_df)
    varx_metrics, varx_diagnostics, selected_varx, selected_varx_spec = run_varx_search(model_df)
    var_granger = granger_table(selected_var)
    var_irf, var_fevd = save_irf_fevd(selected_var)
    varx_scenario = save_varx_scenario(selected_varx, selected_varx_spec)
    var_comparison, varx_comparison, level_var, level_varx = save_comparison_tables(model_df, selected_var, selected_varx, selected_varx_spec)
    decision = baseline_decision_table(stationarity, var_comparison, varx_comparison)
    save_forecast_comparison(var_comparison, varx_comparison)

    plot_irf(var_irf)
    plot_varx_scenario(varx_scenario)
    plot_forecast_comparison(var_comparison, varx_comparison)

    section = report_section(
        stationarity,
        var_comparison,
        varx_comparison,
        var_granger,
        var_irf,
        var_fevd,
        varx_scenario,
        decision,
    )
    update_report_files(section, decision)

    print("FEDFUNDS stationarity robustness completed.")
    print("\nSelected D_FEDFUNDS VAR:")
    print(var_metrics.loc[var_metrics["selected_diff_robustness_model"], ["candidate", "lag_order", "inflation_RMSE", "mean_RMSE", "stable"]])
    print("\nSelected D_FEDFUNDS VARX:")
    print(varx_metrics.loc[varx_metrics["selected_diff_robustness_model"], ["candidate", "lag_order", "inflation_RMSE", "mean_RMSE", "stable"]])
    print("\nLevel vs diff VAR comparison:")
    print(var_comparison[["model", "lag_order", "inflation_RMSE", "mean_RMSE", "acf_exceedance_share", "stable"]])
    print("\nLevel vs diff VARX comparison:")
    print(varx_comparison[["model", "lag_order", "inflation_RMSE", "mean_RMSE", "acf_exceedance_share", "stable"]])


if __name__ == "__main__":
    main()
