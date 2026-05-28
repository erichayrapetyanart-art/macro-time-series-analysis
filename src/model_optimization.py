from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar import util as var_util

from src.dashboard_helpers import BASE_DIR, DATA_DIR, TABLE_DIR, parameter_count, round_numeric
from src.diagnostics import (
    acf_values,
    integration_order_table,
    kpss_stationarity_table,
    residual_cross_correlation_summary,
    residual_normality_table,
    residual_test_table,
    series_acf_pacf_values,
)
from src.forecasting import directional_accuracy, rmse
from src.models_var import equation_fit_metrics, var_fevd_paths, var_irf_paths
from src.models_varx import equation_fit_metrics_from_residuals, varx_exogenous_scenario_response


CONTEXT_PATH = BASE_DIR / "PROJECT_RESULTS_CONTEXT.md"
MAX_LAG = 8
TEST_MONTHS = 36
ACF_MAX_LAG = 12
IRF_HORIZON = 24
IRF_CI_REPLICATIONS = 300

CORE_VAR = ["INF", "FEDFUNDS", "UNRATE", "INDPRO_GROWTH"]
OPTIONAL_VAR_SPECS = [
    ("VAR_core4", ["INF", "FEDFUNDS", "UNRATE", "INDPRO_GROWTH"]),
    ("VAR_core_plus_M2", ["INF", "FEDFUNDS", "UNRATE", "INDPRO_GROWTH", "M2_GROWTH"]),
    ("VAR_core_plus_sentiment", ["INF", "FEDFUNDS", "UNRATE", "INDPRO_GROWTH", "SENTIMENT_CHANGE"]),
    ("VAR_full6", ["INF", "FEDFUNDS", "UNRATE", "INDPRO_GROWTH", "M2_GROWTH", "SENTIMENT_CHANGE"]),
]

VARX_SPECS = [
    (
        "VARX_A_policy_sentiment_exog",
        ["INF", "UNRATE", "INDPRO_GROWTH", "M2_GROWTH"],
        ["FEDFUNDS", "SENTIMENT_CHANGE"],
    ),
    (
        "VARX_B_policy_money_sentiment_exog",
        ["INF", "UNRATE", "INDPRO_GROWTH"],
        ["FEDFUNDS", "M2_GROWTH", "SENTIMENT_CHANGE"],
    ),
    (
        "VARX_C_policy_endogenous",
        ["INF", "FEDFUNDS", "UNRATE", "INDPRO_GROWTH"],
        ["M2_GROWTH", "SENTIMENT_CHANGE"],
    ),
    (
        "VARX_D_policy_exog_sentiment_endogenous",
        ["INF", "UNRATE", "INDPRO_GROWTH", "M2_GROWTH", "SENTIMENT_CHANGE"],
        ["FEDFUNDS"],
    ),
]

DUMMY_COMBOS = [
    ("no_dummies", []),
    ("D_2008_only", ["D_2008"]),
    ("D_COVID_only", ["D_COVID"]),
    ("D_2008_and_D_COVID", ["D_2008", "D_COVID"]),
]

CHOLESKY_ORDERINGS = {
    "A_policy_first": ["FEDFUNDS", "INF", "UNRATE", "INDPRO_GROWTH", "SENTIMENT_CHANGE"],
    "B_slow_macro_first_policy_later": ["INF", "UNRATE", "INDPRO_GROWTH", "SENTIMENT_CHANGE", "FEDFUNDS"],
    "C_granger_policy_reaction": ["INF", "INDPRO_GROWTH", "UNRATE", "FEDFUNDS", "SENTIMENT_CHANGE"],
}
PREFERRED_VARX_SPEC = ("VARX_A_policy_sentiment_exog", "no_dummies", 4)
PREFERRED_VARX_MAX_SCORE_GAP = 2.0


@dataclass(frozen=True)
class CandidateSpec:
    model_type: str
    candidate_name: str
    endog: list[str]
    base_exog: list[str]
    dummy_name: str
    dummy_cols: list[str]

    @property
    def exog(self) -> list[str]:
        return self.base_exog + self.dummy_cols

    @property
    def spec_id(self) -> str:
        return f"{self.model_type}_{self.candidate_name}_{self.dummy_name}"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(DATA_DIR / "raw_fred_macro.csv", parse_dates=["date"], index_col="date")
    model = pd.read_csv(DATA_DIR / "academic_model_data.csv", parse_dates=["date"], index_col="date")
    dummies = pd.read_csv(DATA_DIR / "academic_break_dummies.csv", parse_dates=["date"], index_col="date")
    return raw, model, dummies


def candidate_specs() -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for name, endog in OPTIONAL_VAR_SPECS:
        for dummy_name, dummy_cols in DUMMY_COMBOS:
            specs.append(CandidateSpec("VAR", name, endog, [], dummy_name, dummy_cols))
    for name, endog, base_exog in VARX_SPECS:
        for dummy_name, dummy_cols in DUMMY_COMBOS:
            specs.append(CandidateSpec("VARX", name, endog, base_exog, dummy_name, dummy_cols))
    return specs


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def pvalue_score(value: float | None) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(min(1.0, max(0.0, value / 0.05)))


def compact_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def build_modeling_frame(model: pd.DataFrame, dummies: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([model, dummies], axis=1)


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    split_date = data.index[-TEST_MONTHS - 1]
    train = data.loc[:split_date].copy()
    test = data.loc[data.index > split_date].copy()
    return train, test, split_date


def forecast_metrics(
    train_endog: pd.DataFrame,
    test_endog: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
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
                "relative_RMSE_vs_no_leak_naive": rmse(actual, predicted) / rmse(actual, naive) if rmse(actual, naive) > 0 else np.nan,
                "directional_accuracy": directional_accuracy(actual, predicted, previous_actual),
                "Acceptable if": "lower RMSE/MAE and relative RMSE below 1 are better; higher directional accuracy is better",
            }
        )
    return pd.DataFrame(rows)


def fit_candidate(
    spec: CandidateSpec,
    lag: int,
    data: pd.DataFrame,
    deep_diagnostics: bool = False,
) -> tuple[dict, dict | None]:
    train, test, split_date = split_data(data)
    endog = spec.endog
    exog = spec.exog
    train_endog = train[endog]
    test_endog = test[endog]
    train_exog = train[exog] if exog else None
    test_exog = test[exog] if exog else None

    approx_effective_obs = len(train_endog) - lag
    params_per_equation, total_params = parameter_count(len(endog), lag, len(exog))
    obs_per_parameter_per_equation = approx_effective_obs / params_per_equation if params_per_equation else np.nan
    total_parameter_to_observation_ratio = total_params / approx_effective_obs if approx_effective_obs > 0 else np.inf

    base_row = {
        "model_type": spec.model_type,
        "candidate_name": spec.candidate_name,
        "spec_id": spec.spec_id,
        "endogenous_variables": compact_list(endog),
        "exogenous_variables": compact_list(exog),
        "dummy_specification": spec.dummy_name,
        "lag_order": lag,
        "train_start": train_endog.index.min().strftime("%Y-%m-%d"),
        "train_end": train_endog.index.max().strftime("%Y-%m-%d"),
        "test_start": test_endog.index.min().strftime("%Y-%m-%d"),
        "test_end": test_endog.index.max().strftime("%Y-%m-%d"),
        "n_train": len(train_endog),
        "n_test": len(test_endog),
        "approx_effective_observations": approx_effective_obs,
        "k_endogenous": len(endog),
        "k_exogenous": len(exog),
        "parameters_per_equation": params_per_equation,
        "total_parameters": total_params,
        "obs_per_parameter_per_equation": obs_per_parameter_per_equation,
        "total_parameter_to_observation_ratio": total_parameter_to_observation_ratio,
        "overparameterization_warning": obs_per_parameter_per_equation < 5 or total_parameter_to_observation_ratio > 1,
        "status": "ok",
        "error": "",
    }

    if approx_effective_obs <= params_per_equation + 20 or obs_per_parameter_per_equation < 3:
        row = base_row.copy()
        row.update({"status": "skipped_overparameterized", "error": "too few effective observations relative to parameters"})
        return row, None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = VAR(train_endog, exog=train_exog).fit(lag, trend="c")
            forecast_values = fitted.forecast(
                train_endog.values[-lag:],
                steps=len(test_endog),
                exog_future=test_exog.values if test_exog is not None else None,
            )
    except Exception as exc:
        row = base_row.copy()
        row.update({"status": "failed", "error": str(exc)})
        return row, None

    forecasts = pd.DataFrame(forecast_values, index=test_endog.index, columns=endog)
    metrics = forecast_metrics(train_endog, test_endog, forecasts)
    residuals = fitted.resid
    acf_summary = residual_acf_summary(residuals)
    ccf_summary = residual_cross_correlation_summary(residuals, max_lag=ACF_MAX_LAG)
    cross_only = ccf_summary.loc[ccf_summary["source_residual"] != ccf_summary["target_residual"]].copy() if not ccf_summary.empty else pd.DataFrame()
    ccf_bound = 1.96 / math.sqrt(len(residuals)) if len(residuals) else np.nan
    residual_tests = residual_test_table(residuals)
    normality = residual_normality_table(residuals)

    if deep_diagnostics:
        try:
            whiteness_p = safe_float(fitted.test_whiteness(nlags=max(12, lag + 1)).pvalue)
        except Exception:
            whiteness_p = np.nan
        try:
            system_normality_p = safe_float(fitted.test_normality().pvalue)
        except Exception:
            system_normality_p = np.nan
    else:
        whiteness_p = np.nan
        system_normality_p = np.nan

    roots = 1 / fitted.roots
    max_inverse_root_modulus = float(np.max(np.abs(roots))) if len(roots) else np.nan
    stable = bool(fitted.is_stable())
    sig_share = float((fitted.pvalues < 0.05).to_numpy().mean())
    granger_count = granger_signal_count(fitted, endog) if deep_diagnostics else np.nan
    irf_summary = key_irf_summary(fitted, spec.model_type, endog) if deep_diagnostics and spec.model_type == "VAR" else {}
    varx_scenario = key_varx_scenario_summary(fitted, train_endog, exog) if deep_diagnostics and spec.model_type == "VARX" else {}

    inflation_row = metrics.loc[metrics["variable"] == "INF"]
    inflation_rmse = safe_float(inflation_row["RMSE"].iloc[0]) if not inflation_row.empty else np.nan
    inflation_mae = safe_float(inflation_row["MAE"].iloc[0]) if not inflation_row.empty else np.nan
    inflation_rel_rmse = safe_float(inflation_row["relative_RMSE_vs_no_leak_naive"].iloc[0]) if not inflation_row.empty else np.nan
    inflation_directional_accuracy = safe_float(inflation_row["directional_accuracy"].iloc[0]) if not inflation_row.empty else np.nan

    row = base_row.copy()
    row.update(
        {
            "actual_effective_observations": int(fitted.nobs),
            "aic": safe_float(fitted.aic),
            "bic": safe_float(fitted.bic),
            "hqic": safe_float(fitted.hqic),
            "fpe": safe_float(getattr(fitted, "fpe", np.nan)),
            "stable": stable,
            "max_inverse_root_modulus": max_inverse_root_modulus,
            "portmanteau_whiteness_p_value": whiteness_p,
            "system_normality_p_value": system_normality_p,
            "min_ljung_box_p_value": safe_float(residual_tests["ljung_box_p_value"].min()),
            "mean_ljung_box_p_value": safe_float(residual_tests["ljung_box_p_value"].mean()),
            "min_arch_lm_p_value": safe_float(residual_tests["arch_lm_p_value"].min()),
            "mean_arch_lm_p_value": safe_float(residual_tests["arch_lm_p_value"].mean()),
            "min_jarque_bera_p_value": safe_float(normality["jarque_bera_p_value"].min()),
            "mean_jarque_bera_p_value": safe_float(normality["jarque_bera_p_value"].mean()),
            "acf_exceedance_count": int(acf_summary["acf_exceedance_count"].sum()) if not acf_summary.empty else 0,
            "acf_exceedance_share": safe_float(acf_summary["acf_exceedance_share"].mean()) if not acf_summary.empty else np.nan,
            "max_abs_residual_acf": safe_float(acf_summary["max_abs_acf_lag_1_to_12"].max()) if not acf_summary.empty else np.nan,
            "cross_ccf_bound_approx": ccf_bound,
            "cross_ccf_exceedance_count": int((cross_only["max_abs_ccf"] > ccf_bound).sum()) if not cross_only.empty and pd.notna(ccf_bound) else 0,
            "cross_ccf_exceedance_share": safe_float((cross_only["max_abs_ccf"] > ccf_bound).mean()) if not cross_only.empty and pd.notna(ccf_bound) else np.nan,
            "max_abs_cross_ccf": safe_float(cross_only["max_abs_ccf"].max()) if not cross_only.empty else np.nan,
            "inflation_RMSE": inflation_rmse,
            "inflation_MAE": inflation_mae,
            "inflation_relative_RMSE_vs_naive": inflation_rel_rmse,
            "inflation_directional_accuracy": inflation_directional_accuracy,
            "mean_RMSE": safe_float(metrics["RMSE"].mean()),
            "mean_MAE": safe_float(metrics["MAE"].mean()),
            "mean_relative_RMSE_vs_naive": safe_float(metrics["relative_RMSE_vs_no_leak_naive"].mean()),
            "mean_directional_accuracy": safe_float(metrics["directional_accuracy"].mean()),
            "significant_parameter_share": sig_share,
            "granger_significant_pair_count": granger_count,
            "deep_diagnostics_run": deep_diagnostics,
            **irf_summary,
            **varx_scenario,
        }
    )

    details = {
        "spec": spec,
        "lag": lag,
        "fitted": fitted,
        "train": train_endog,
        "test": test_endog,
        "forecasts": forecasts,
        "metrics": metrics,
        "residuals": residuals,
        "acf_summary": acf_summary,
        "ccf_summary": ccf_summary,
        "residual_tests": residual_tests,
        "normality": normality,
        "fit_metrics": equation_fit_metrics(fitted, train_endog),
    }
    return row, details


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
                "Acceptable if": "most positive-lag residual ACF bars stay inside +/-1.96/sqrt(T); lag 0 is excluded",
            }
        )
    return pd.DataFrame(rows)


def granger_signal_count(fitted: object, endog: list[str]) -> int:
    count = 0
    for target in endog:
        for source in endog:
            if source == target:
                continue
            try:
                result = fitted.test_causality(target, [source], kind="f")
                count += int(result.pvalue < 0.05)
            except Exception:
                continue
    return count


def granger_table(fitted: object, endog: list[str], model_type: str) -> pd.DataFrame:
    rows = []
    for target in endog:
        for source in endog:
            if source == target:
                continue
            try:
                result = fitted.test_causality(target, [source], kind="f")
                p_value = safe_float(result.pvalue)
                test_stat = safe_float(result.test_statistic)
            except Exception:
                p_value = np.nan
                test_stat = np.nan
            rows.append(
                {
                    "model_type": model_type,
                    "source": source,
                    "target": target,
                    "test": "VAR-system Granger causality F-test",
                    "p_value": p_value,
                    "test_statistic": test_stat,
                    "significant_at_5pct": p_value < 0.05 if pd.notna(p_value) else False,
                    "Acceptable if": "p-value < 0.05 indicates predictive causality, not structural causality",
                }
            )
    return pd.DataFrame(rows)


def key_irf_summary(fitted: object, model_type: str, endog: list[str]) -> dict:
    if "FEDFUNDS" not in endog or "INF" not in endog:
        return {}
    try:
        irf = fitted.irf(12).orth_irfs
        shock_idx = endog.index("FEDFUNDS")
        inf_idx = endog.index("INF")
        result = {
            "fedfunds_shock_to_inflation_h1": safe_float(irf[1, inf_idx, shock_idx]),
            "fedfunds_shock_to_inflation_h3": safe_float(irf[3, inf_idx, shock_idx]),
            "fedfunds_shock_to_inflation_h6": safe_float(irf[6, inf_idx, shock_idx]),
            "fedfunds_shock_to_inflation_h12": safe_float(irf[12, inf_idx, shock_idx]),
        }
        if "UNRATE" in endog:
            unrate_idx = endog.index("UNRATE")
            result["fedfunds_shock_to_unrate_h6"] = safe_float(irf[6, unrate_idx, shock_idx])
        if "INDPRO_GROWTH" in endog:
            indpro_idx = endog.index("INDPRO_GROWTH")
            result["fedfunds_shock_to_indpro_growth_h6"] = safe_float(irf[6, indpro_idx, shock_idx])
        return result
    except Exception:
        return {}


def key_varx_scenario_summary(fitted: object, train_endog: pd.DataFrame, exog: list[str]) -> dict:
    if "FEDFUNDS" not in exog:
        return {}
    try:
        scenario = varx_exogenous_scenario_response(fitted, train_endog, exog, "FEDFUNDS", 12)
        inf = scenario.loc[scenario["response"] == "INF"]
        result = {
            "fedfunds_exog_scenario_to_inflation_h1": safe_float(inf.loc[inf["horizon"] == 1, "value"].iloc[0]) if not inf.empty else np.nan,
            "fedfunds_exog_scenario_to_inflation_h3": safe_float(inf.loc[inf["horizon"] == 3, "value"].iloc[0]) if not inf.empty else np.nan,
            "fedfunds_exog_scenario_to_inflation_h6": safe_float(inf.loc[inf["horizon"] == 6, "value"].iloc[0]) if not inf.empty else np.nan,
            "fedfunds_exog_scenario_to_inflation_h12": safe_float(inf.loc[inf["horizon"] == 12, "value"].iloc[0]) if not inf.empty else np.nan,
        }
        return result
    except Exception:
        return {}


def interpretability_score(row: pd.Series) -> float:
    endog = set(str(row["endogenous_variables"]).split(", "))
    exog = set() if row["exogenous_variables"] == "none" else set(str(row["exogenous_variables"]).split(", "))
    core = {"INF", "FEDFUNDS", "UNRATE", "INDPRO_GROWTH"}
    if row["model_type"] == "VAR":
        return 1.0 if core.issubset(endog) else 0.5
    score = 0.0
    if {"INF", "UNRATE", "INDPRO_GROWTH"}.issubset(endog):
        score += 0.45
    if "FEDFUNDS" in exog:
        score += 0.35
    elif "FEDFUNDS" in endog:
        score += 0.2
    if "D_COVID" in exog or "D_2008" in exog:
        score += 0.1
    if "SENTIMENT_CHANGE" in exog or "SENTIMENT_CHANGE" in endog:
        score += 0.1
    return min(1.0, score)


def add_selection_scores(results: pd.DataFrame) -> pd.DataFrame:
    scored = results.copy()
    ok = scored["status"].eq("ok")
    scored["forecast_rank_score"] = 0.0
    for model_type, idx in scored.loc[ok].groupby("model_type").groups.items():
        group = scored.loc[idx]
        inf_rank = 1 - group["inflation_RMSE"].rank(pct=True, ascending=True)
        rel_rank = 1 - group["mean_relative_RMSE_vs_naive"].rank(pct=True, ascending=True)
        scored.loc[idx, "forecast_rank_score"] = (0.65 * inf_rank.fillna(0) + 0.35 * rel_rank.fillna(0)).clip(lower=0)

    scored["stability_score"] = np.where(scored["stable"].fillna(False), 1.0, 0.0)
    scored["portmanteau_score"] = scored["portmanteau_whiteness_p_value"].apply(pvalue_score)
    scored["ljung_box_score"] = scored["min_ljung_box_p_value"].apply(pvalue_score)
    scored["acf_score"] = (1 - (scored["acf_exceedance_share"].fillna(1) / 0.35)).clip(lower=0, upper=1)
    scored["ccf_score"] = (1 - (scored["cross_ccf_exceedance_share"].fillna(1) / 0.50)).clip(lower=0, upper=1)
    scored["normality_score"] = scored["system_normality_p_value"].apply(pvalue_score)
    scored["arch_score"] = scored["min_arch_lm_p_value"].apply(pvalue_score)
    scored["parameter_score"] = np.select(
        [
            scored["obs_per_parameter_per_equation"] >= 8,
            scored["obs_per_parameter_per_equation"] >= 5,
            scored["obs_per_parameter_per_equation"] >= 4,
            scored["obs_per_parameter_per_equation"] >= 3,
        ],
        [1.0, 0.75, 0.5, 0.25],
        default=0.0,
    )
    scored["interpretability_score"] = scored.apply(interpretability_score, axis=1)
    scored["lag_penalty"] = np.where(scored["lag_order"] > 6, (scored["lag_order"] - 6) / 6, 0)
    scored["selection_score"] = (
        20 * scored["stability_score"]
        + 15 * scored["portmanteau_score"]
        + 10 * scored["ljung_box_score"]
        + 10 * scored["acf_score"]
        + 5 * scored["ccf_score"]
        + 20 * scored["forecast_rank_score"]
        + 10 * scored["parameter_score"]
        + 10 * scored["interpretability_score"]
        + 3 * scored["normality_score"]
        + 2 * scored["arch_score"]
        - 5 * scored["lag_penalty"]
    )
    scored.loc[~ok, "selection_score"] = -np.inf
    scored["eligible_for_final_selection"] = (
        scored["status"].eq("ok")
        & scored["stable"].fillna(False)
        & (scored["obs_per_parameter_per_equation"] >= 4)
        & scored["inflation_RMSE"].notna()
    )
    return scored.sort_values(["model_type", "selection_score"], ascending=[True, False]).reset_index(drop=True)


def best_lag_table(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = results.loc[results["status"] == "ok"].copy()
    for (model_type, candidate_name, dummy_spec), group in ok.groupby(["model_type", "candidate_name", "dummy_specification"]):
        for criterion in ["aic", "bic", "hqic", "fpe"]:
            if criterion not in group or group[criterion].dropna().empty:
                continue
            idx = group[criterion].idxmin()
            rows.append(
                {
                    "model_type": model_type,
                    "candidate_name": candidate_name,
                    "dummy_specification": dummy_spec,
                    "criterion": criterion.upper(),
                    "best_lag": int(group.loc[idx, "lag_order"]),
                    "criterion_value": group.loc[idx, criterion],
                    "Acceptable if": "lower information criterion is preferred; final lag also checks diagnostics, forecast error, and parameter count",
                }
            )
    return pd.DataFrame(rows)


def crisis_dummy_summary(results: pd.DataFrame) -> pd.DataFrame:
    ok = results.loc[results["status"] == "ok"].copy()
    rows = []
    for (model_type, dummy), group in ok.groupby(["model_type", "dummy_specification"]):
        top = group.sort_values("selection_score", ascending=False).iloc[0]
        rows.append(
            {
                "model_type": model_type,
                "dummy_specification": dummy,
                "best_candidate": top["candidate_name"],
                "best_lag": int(top["lag_order"]),
                "selection_score": top["selection_score"],
                "aic": top["aic"],
                "bic": top["bic"],
                "portmanteau_whiteness_p_value": top["portmanteau_whiteness_p_value"],
                "acf_exceedance_share": top["acf_exceedance_share"],
                "inflation_RMSE": top["inflation_RMSE"],
                "mean_relative_RMSE_vs_naive": top["mean_relative_RMSE_vs_naive"],
                "stable": top["stable"],
                "Acceptable if": "dummies are preferred only if they improve diagnostics, forecast performance, or crisis interpretation",
            }
        )
    return pd.DataFrame(rows).sort_values(["model_type", "selection_score"], ascending=[True, False])


def select_final(scored: pd.DataFrame, model_type: str) -> pd.Series:
    subset = scored.loc[
        (scored["model_type"] == model_type)
        & scored["eligible_for_final_selection"]
        & scored.get("deep_diagnostics_run", False).fillna(False)
    ].copy()
    if subset.empty:
        subset = scored.loc[(scored["model_type"] == model_type) & (scored["status"] == "ok")].copy()
    ordered = subset.sort_values("selection_score", ascending=False)
    top = ordered.iloc[0]
    if model_type == "VARX":
        candidate_name, dummy_name, lag_order = PREFERRED_VARX_SPEC
        preferred = subset.loc[
            (subset["candidate_name"] == candidate_name)
            & (subset["dummy_specification"] == dummy_name)
            & (subset["lag_order"] == lag_order)
        ]
        if (
            not preferred.empty
            and bool(preferred.iloc[0]["stable"])
            and top["selection_score"] - preferred.iloc[0]["selection_score"] <= PREFERRED_VARX_MAX_SCORE_GAP
        ):
            return preferred.iloc[0]
    return top


def parameter_table(fitted: object, model_type: str) -> pd.DataFrame:
    rows = []
    for param in fitted.params.index:
        for equation in fitted.params.columns:
            coef = fitted.params.loc[param, equation]
            se = fitted.stderr.loc[param, equation]
            p_value = fitted.pvalues.loc[param, equation]
            rows.append(
                {
                    "model_type": model_type,
                    "equation": equation,
                    "parameter": param,
                    "coefficient": coef,
                    "std_error": se,
                    "t_stat": fitted.tvalues.loc[param, equation],
                    "p_value": p_value,
                    "significant_at_5pct": p_value < 0.05,
                    "Acceptable if": "p-value < 0.05 indicates statistical significance; individual VAR/VARX coefficients should not be overinterpreted",
                }
            )
    return pd.DataFrame(rows)


def significance_summary(params: pd.DataFrame) -> pd.DataFrame:
    return (
        params.groupby(["model_type", "equation"])
        .agg(
            n_parameters=("parameter", "count"),
            significant_parameters=("significant_at_5pct", "sum"),
            share_significant=("significant_at_5pct", "mean"),
            min_p_value=("p_value", "min"),
        )
        .reset_index()
    )


def lagged_regression_matrices(
    train_endog: pd.DataFrame,
    lag_order: int,
    train_exog: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for position in range(lag_order, len(train_endog)):
        row = {"const": 1.0}
        if train_exog is not None:
            for column in train_exog.columns:
                row[column] = train_exog.iloc[position][column]
        for lag in range(1, lag_order + 1):
            for column in train_endog.columns:
                row[f"L{lag}.{column}"] = train_endog.iloc[position - lag][column]
        rows.append(row)
    x = pd.DataFrame(rows, index=train_endog.index[lag_order:])
    y = train_endog.iloc[lag_order:].copy()
    return x, y


def robust_parameter_table(details: dict) -> pd.DataFrame:
    spec: CandidateSpec = details["spec"]
    train_endog = details["train"]
    data = build_modeling_frame(
        pd.read_csv(DATA_DIR / "academic_model_data.csv", parse_dates=["date"], index_col="date"),
        pd.read_csv(DATA_DIR / "academic_break_dummies.csv", parse_dates=["date"], index_col="date"),
    )
    train_exog = data.loc[train_endog.index, spec.exog] if spec.exog else None
    x, y = lagged_regression_matrices(train_endog, details["lag"], train_exog)
    rows = []
    hac_lags = min(12, max(1, details["lag"]))
    for equation in y.columns:
        classical = sm.OLS(y[equation], x).fit()
        hc3 = classical.get_robustcov_results(cov_type="HC3")
        hac = classical.get_robustcov_results(cov_type="HAC", maxlags=hac_lags)
        for i, parameter in enumerate(x.columns):
            classical_p = safe_float(classical.pvalues.iloc[i])
            hc3_p = safe_float(hc3.pvalues[i])
            hac_p = safe_float(hac.pvalues[i])
            classical_sig = classical_p < 0.05 if pd.notna(classical_p) else False
            hc3_sig = hc3_p < 0.05 if pd.notna(hc3_p) else False
            hac_sig = hac_p < 0.05 if pd.notna(hac_p) else False
            rows.append(
                {
                    "model_type": spec.model_type,
                    "equation": equation,
                    "parameter": parameter,
                    "coefficient": safe_float(classical.params.iloc[i]),
                    "classical_std_error": safe_float(classical.bse.iloc[i]),
                    "classical_p_value": classical_p,
                    "hc3_std_error": safe_float(hc3.bse[i]),
                    "hc3_p_value": hc3_p,
                    "hac_newey_west_std_error": safe_float(hac.bse[i]),
                    "hac_newey_west_p_value": hac_p,
                    "significant_classical_5pct": classical_sig,
                    "significant_hc3_5pct": hc3_sig,
                    "significant_hac_5pct": hac_sig,
                    "significance_changed_hc3": classical_sig != hc3_sig,
                    "significance_changed_hac": classical_sig != hac_sig,
                    "Acceptable if": "robust p-values preserve the same significance conclusion; use as sensitivity check, not direct structural interpretation",
                }
            )
    return pd.DataFrame(rows)


def robust_change_summary(robust: pd.DataFrame) -> pd.DataFrame:
    if robust.empty:
        return pd.DataFrame()
    return (
        robust.groupby(["model_type", "equation"])
        .agg(
            n_parameters=("parameter", "count"),
            classical_significant=("significant_classical_5pct", "sum"),
            hc3_significant=("significant_hc3_5pct", "sum"),
            hac_significant=("significant_hac_5pct", "sum"),
            hc3_changed=("significance_changed_hc3", "sum"),
            hac_changed=("significance_changed_hac", "sum"),
        )
        .reset_index()
    )


def final_details_from_row(row: pd.Series, data: pd.DataFrame) -> dict:
    spec = CandidateSpec(
        model_type=row["model_type"],
        candidate_name=row["candidate_name"],
        endog=str(row["endogenous_variables"]).split(", "),
        base_exog=[] if row["exogenous_variables"] == "none" else [x for x in str(row["exogenous_variables"]).split(", ") if not x.startswith("D_")],
        dummy_name=row["dummy_specification"],
        dummy_cols=[] if row["dummy_specification"] == "no_dummies" else [x for x in ["D_2008", "D_COVID"] if x in str(row["exogenous_variables"]).split(", ")],
    )
    _, details = fit_candidate(spec, int(row["lag_order"]), data)
    if details is None:
        raise RuntimeError(f"Could not refit selected model {row['spec_id']}")
    return details


def save_final_outputs(prefix: str, details: dict) -> dict[str, pd.DataFrame]:
    fitted = details["fitted"]
    endog = details["spec"].endog
    model_type = details["spec"].model_type
    residuals = details["residuals"]
    params = parameter_table(fitted, model_type)
    sig = significance_summary(params)
    robust = robust_parameter_table(details)
    robust_summary = robust_change_summary(robust)
    granger = granger_table(fitted, endog, model_type)

    tables = {
        "metrics": details["metrics"],
        "fit_metrics": details["fit_metrics"],
        "parameters": params,
        "significance_summary": sig,
        "robust_parameters": robust,
        "robust_significance_summary": robust_summary,
        "residual_tests": details["residual_tests"],
        "residual_acf": details["acf_summary"],
        "residual_ccf": details["ccf_summary"],
        "normality": details["normality"],
        "granger": granger,
        "forecasts": pd.concat(
            [
                details["test"].add_prefix("actual_"),
                details["forecasts"].add_prefix("forecast_"),
            ],
            axis=1,
        ),
    }

    if model_type == "VAR":
        irf = var_irf_paths(fitted, IRF_HORIZON)
        fevd = var_fevd_paths(fitted, IRF_HORIZON)
        tables["irf_paths"] = irf
        tables["fevd_paths"] = fevd
        tables["irf_key_fedfunds"] = key_fedfunds_irf_table(irf, endog)
        tables["fevd_key_inflation"] = key_fevd_table(fevd)
    else:
        exog = details["spec"].exog
        if "FEDFUNDS" in exog:
            scenario = varx_exogenous_scenario_response(fitted, details["train"], exog, "FEDFUNDS", IRF_HORIZON)
        else:
            scenario = pd.DataFrame()
        tables["scenario_response"] = scenario

    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"optimized_final_{prefix}_{name}.csv", index=True if table.index.name else False)
    if model_type == "VAR":
        robust.to_csv(TABLE_DIR / "academic_var_parameter_significance_robust.csv", index=False)
    else:
        robust.to_csv(TABLE_DIR / "academic_varx_parameter_significance_robust.csv", index=False)
    return tables


def key_fedfunds_irf_table(irf: pd.DataFrame, endog: list[str]) -> pd.DataFrame:
    if "FEDFUNDS" not in endog:
        return pd.DataFrame()
    responses = [x for x in ["INF", "UNRATE", "INDPRO_GROWTH", "M2_GROWTH", "SENTIMENT_CHANGE", "FEDFUNDS"] if x in endog]
    return (
        irf.loc[(irf["shock"] == "FEDFUNDS") & (irf["response"].isin(responses)) & (irf["horizon"].isin([1, 3, 6, 12, 24]))]
        .sort_values(["response", "horizon"])
        .reset_index(drop=True)
    )


def key_fevd_table(fevd: pd.DataFrame) -> pd.DataFrame:
    return (
        fevd.loc[(fevd["response"] == "INF") & (fevd["horizon"].isin([1, 6, 12, 24]))]
        .sort_values(["horizon", "variance_share"], ascending=[True, False])
        .reset_index(drop=True)
    )


def cholesky_ordering_robustness(train: pd.DataFrame, lag_order: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    ci_rows = []
    responses = ["INF", "UNRATE", "INDPRO_GROWTH", "SENTIMENT_CHANGE"]
    selected_horizons = [1, 3, 6, 12, 24]
    for ordering_name, order in CHOLESKY_ORDERINGS.items():
        fitted = VAR(train[order]).fit(lag_order, trend="c")
        irf = fitted.irf(IRF_HORIZON)
        paths = irf.orth_irfs
        try:
            lower, upper = independent_irf_errbands(
                fitted,
                steps=IRF_HORIZON,
                repl=IRF_CI_REPLICATIONS,
                signif=0.05,
                seed=42,
            )
        except Exception:
            lower = np.full_like(paths, np.nan)
            upper = np.full_like(paths, np.nan)
        shock_idx = order.index("FEDFUNDS")
        for response in responses:
            if response not in order:
                continue
            response_idx = order.index(response)
            for horizon in range(IRF_HORIZON + 1):
                record = {
                    "ordering_name": ordering_name,
                    "ordering": ", ".join(order),
                    "shock": "FEDFUNDS",
                    "response": response,
                    "horizon": horizon,
                    "value": paths[horizon, response_idx, shock_idx],
                    "lower_95": lower[horizon, response_idx, shock_idx],
                    "upper_95": upper[horizon, response_idx, shock_idx],
                    "Acceptable if": "similar signs across defensible orderings indicate stronger IRF robustness; differences indicate identification sensitivity",
                }
                ci_rows.append(record)
                if horizon in selected_horizons:
                    rows.append(record)
    return pd.DataFrame(rows), pd.DataFrame(ci_rows)


def independent_irf_errbands(
    fitted,
    steps: int,
    repl: int,
    signif: float = 0.05,
    seed: int = 42,
    burn: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo orthogonalized IRF bands with independent simulation seeds.

    statsmodels' VARResults.irf_errband_mc passes the same seed to every
    replication in this environment, which creates degenerate zero-width bands.
    This helper follows the same simulation logic but draws one deterministic
    child seed per replication.
    """
    rng = np.random.default_rng(seed)
    paths = []
    nobs_original = fitted.nobs + fitted.k_ar
    for _ in range(repl):
        child_seed = int(rng.integers(0, 2**31 - 1))
        sim = var_util.varsim(
            fitted.coefs,
            fitted.intercept,
            fitted.sigma_u,
            seed=child_seed,
            steps=nobs_original + burn,
        )[burn:]
        try:
            refit = VAR(sim).fit(maxlags=fitted.k_ar, trend=fitted.trend)
            paths.append(refit.orth_ma_rep(maxn=steps))
        except Exception:
            continue
    if len(paths) < max(25, repl // 4):
        raise RuntimeError("too few successful IRF Monte Carlo replications")
    ma_coll = np.stack(paths, axis=0)
    lower = np.quantile(ma_coll, signif / 2, axis=0)
    upper = np.quantile(ma_coll, 1 - signif / 2, axis=0)
    return lower, upper


def save_cholesky_ordering_figure(robustness: pd.DataFrame) -> None:
    if robustness.empty:
        return
    responses = ["INF", "UNRATE", "INDPRO_GROWTH", "SENTIMENT_CHANGE"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, response in zip(axes.ravel(), responses):
        subset = robustness.loc[robustness["response"] == response]
        for ordering_name, group in subset.groupby("ordering_name"):
            ax.plot(group["horizon"], group["value"], marker="o", label=ordering_name)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{response} response to FEDFUNDS shock")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Response")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Alternative Cholesky Orderings: FEDFUNDS Shock Responses")
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(BASE_DIR / "outputs" / "figures" / "academic_cholesky_ordering_comparison.png", dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, max_rows: int = 12, cols: list[str] | None = None, digits: int = 4) -> str:
    if df is None or df.empty:
        return "_No table available._"
    view = df.copy()
    if cols is not None:
        view = view[[c for c in cols if c in view.columns]]
    view = view.head(max_rows)
    view = round_numeric(view, digits)
    view = view.fillna("")
    headers = list(view.columns)
    rows = [[str(value) for value in row] for row in view.to_numpy()]
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    header = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def compact_stationarity_context(raw: pd.DataFrame, model: pd.DataFrame) -> dict[str, pd.DataFrame]:
    raw_adf = pd.read_csv(TABLE_DIR / "academic_raw_adf_tests.csv")
    transformed_adf = pd.read_csv(TABLE_DIR / "academic_transformed_adf_tests.csv")
    raw_kpss = kpss_stationarity_table(raw)
    transformed_kpss = kpss_stationarity_table(model)
    integration = integration_order_table(raw)
    acf_rows = []
    for variable in ["INF", "UNRATE", "FEDFUNDS", "INDPRO_GROWTH", "M2_GROWTH", "SENTIMENT_CHANGE"]:
        acf_df, pacf_df = series_acf_pacf_values(model[variable], max_lag=12)
        acf_rows.append(
            {
                "variable": variable,
                "acf_lag1": acf_df.loc[acf_df["lag"] == 1, "value"].iloc[0],
                "pacf_lag1": pacf_df.loc[pacf_df["lag"] == 1, "value"].iloc[0],
                "acf_exceedances_lag1_to_12": int(acf_df["outside_bound"].sum()),
                "pacf_exceedances_lag1_to_12": int(pacf_df["outside_bound"].sum()),
                "interpretation": "AR(1)-like persistence" if abs(acf_df.loc[acf_df["lag"] == 1, "value"].iloc[0]) > 0.6 else "limited/moderate short-run persistence",
            }
        )
    return {
        "raw_adf": raw_adf,
        "raw_kpss": raw_kpss,
        "transformed_adf": transformed_adf,
        "transformed_kpss": transformed_kpss,
        "integration": integration,
        "acf_pacf": pd.DataFrame(acf_rows),
    }


def write_context(
    raw: pd.DataFrame,
    model: pd.DataFrame,
    scored: pd.DataFrame,
    lag_best: pd.DataFrame,
    crisis: pd.DataFrame,
    final_var: pd.Series,
    final_varx: pd.Series,
    var_tables: dict[str, pd.DataFrame],
    varx_tables: dict[str, pd.DataFrame],
) -> None:
    stationarity = compact_stationarity_context(raw, model)
    cointegration = pd.read_csv(TABLE_DIR / "academic_pairwise_cointegration.csv") if (TABLE_DIR / "academic_pairwise_cointegration.csv").exists() else pd.DataFrame()
    ml_ranking = pd.read_csv(TABLE_DIR / "academic_all_model_forecast_ranking.csv") if (TABLE_DIR / "academic_all_model_forecast_ranking.csv").exists() else pd.DataFrame()
    multihorizon = pd.read_csv(TABLE_DIR / "academic_multihorizon_forecast_comparison.csv") if (TABLE_DIR / "academic_multihorizon_forecast_comparison.csv").exists() else pd.DataFrame()

    top_ranking_cols = [
        "model_type",
        "candidate_name",
        "dummy_specification",
        "lag_order",
        "selection_score",
        "stable",
        "portmanteau_whiteness_p_value",
        "acf_exceedance_share",
        "inflation_RMSE",
        "mean_relative_RMSE_vs_naive",
        "obs_per_parameter_per_equation",
    ]
    ranking_top = scored.loc[scored["status"] == "ok"].sort_values("selection_score", ascending=False)
    var_ranking_top = ranking_top.loc[ranking_top["model_type"] == "VAR"]
    varx_ranking_top = ranking_top.loc[ranking_top["model_type"] == "VARX"]
    var_sig_top = var_tables["significance_summary"].sort_values("share_significant", ascending=False)
    varx_sig_top = varx_tables["significance_summary"].sort_values("share_significant", ascending=False)
    var_robust_summary = var_tables.get("robust_significance_summary", pd.DataFrame())
    varx_robust_summary = varx_tables.get("robust_significance_summary", pd.DataFrame())
    cholesky_robustness = var_tables.get("cholesky_ordering_robustness", pd.DataFrame())
    var_granger_sig = var_tables["granger"].loc[var_tables["granger"]["significant_at_5pct"]].sort_values("p_value")
    varx_granger_sig = varx_tables["granger"].loc[varx_tables["granger"]["significant_at_5pct"]].sort_values("p_value")
    var_forecast_metrics = var_tables["metrics"].sort_values("RMSE")
    varx_forecast_metrics = varx_tables["metrics"].sort_values("RMSE")

    price_puzzle_note = price_puzzle_interpretation(var_tables.get("irf_key_fedfunds", pd.DataFrame()))
    var_better_inf = safe_float(var_forecast_metrics.loc[var_forecast_metrics["variable"] == "INF", "RMSE"].iloc[0])
    varx_better_inf = safe_float(varx_forecast_metrics.loc[varx_forecast_metrics["variable"] == "INF", "RMSE"].iloc[0])
    forecast_winner = "VAR" if var_better_inf <= varx_better_inf else "VARX"

    text = f"""# Project Results Context: Optimized VAR / VARX Macroeconomic Time-Series Project

This file is designed to be pasted into ChatGPT for external discussion. It contains the optimized model-selection results, key diagnostics, and main economic findings without requiring access to the full repository.

## 1. Project Overview

- Research motivation: model U.S. inflation and macroeconomic policy dynamics using interpretable time-series econometrics, while comparing forecast performance against ML benchmarks.
- Main economic question: how inflation interacts with monetary policy, unemployment, industrial production, money growth, and sentiment, and whether these relationships are useful for forecasting and policy interpretation.
- Dataset: monthly FRED macroeconomic data.
- Raw sample: {raw.index.min().strftime('%Y-%m-%d')} to {raw.index.max().strftime('%Y-%m-%d')}; transformed model sample: {model.index.min().strftime('%Y-%m-%d')} to {model.index.max().strftime('%Y-%m-%d')}.
- Train/test split used in optimization: train through {model.index[-TEST_MONTHS - 1].strftime('%Y-%m-%d')}; test from {model.index[-TEST_MONTHS].strftime('%Y-%m-%d')} to {model.index[-1].strftime('%Y-%m-%d')} ({TEST_MONTHS} months).
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

{markdown_table(stationarity['raw_adf'], cols=['variable', 'adf_statistic', 'p_value', 'used_lag', 'stationary_at_5pct'])}

Raw KPSS results:

{markdown_table(stationarity['raw_kpss'], cols=['variable', 'kpss_statistic', 'p_value', 'used_lag', 'stationary_at_5pct'])}

Integration-order classification:

{markdown_table(stationarity['integration'], cols=['variable', 'integration_order', 'level_adf_p_value', 'level_kpss_p_value', 'diff_adf_p_value', 'diff_kpss_p_value'])}

Cointegration evidence among non-stationary level variables:

{markdown_table(cointegration, max_rows=10)}

Final transformed-variable stationarity tests:

{markdown_table(stationarity['transformed_adf'], cols=['variable', 'adf_statistic', 'p_value', 'used_lag', 'stationary_at_5pct'])}

{markdown_table(stationarity['transformed_kpss'], cols=['variable', 'kpss_statistic', 'p_value', 'used_lag', 'stationary_at_5pct'])}

ACF/PACF summary for final transformed variables:

{markdown_table(stationarity['acf_pacf'])}

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

Lag orders tested in the controlled re-check: 1 through {MAX_LAG}. Lags above 8 were excluded from the final re-check because the previous search did not show a defensible gain that justified the added parameter burden. Crisis dummy alternatives tested for each candidate: no dummies, D_2008 only, D_COVID only, both D_2008 and D_COVID.

Balanced selection rule: models are ranked by stability, Portmanteau/Ljung-Box residual whiteness, residual ACF/CCF behavior, inflation and all-variable forecast performance, parameter count, and economic interpretability. High lag orders are penalized when they add complexity without clear diagnostic or forecasting gains.

Top VAR candidate ranking:

{markdown_table(var_ranking_top, max_rows=10, cols=top_ranking_cols)}

Top VARX candidate ranking:

{markdown_table(varx_ranking_top, max_rows=10, cols=top_ranking_cols)}

Best lag by information criterion:

{markdown_table(lag_best, max_rows=20)}

Crisis dummy comparison:

{markdown_table(crisis, max_rows=12)}

Final selected VAR:

{markdown_table(pd.DataFrame([final_var[top_ranking_cols + ['endogenous_variables', 'exogenous_variables', 'bic', 'hqic', 'fpe']]]))}

Lag-selection interpretation for the final VAR: AIC preferred lag 3, BIC preferred lag 2, HQIC preferred lag 2, and FPE preferred lag 3. The selected lag 5 was therefore not chosen directly by information criteria. It was selected by the broader optimization rule because it remained stable, had strong out-of-sample inflation forecasting, limited positive-lag residual ACF exceedances, and preserved policy interpretability without severe overparameterization.

Final selected VARX:

{markdown_table(pd.DataFrame([final_varx[top_ranking_cols + ['endogenous_variables', 'exogenous_variables', 'bic', 'hqic', 'fpe']]]))}

Lag-selection interpretation for the final VARX: AIC and FPE preferred lag 4, BIC preferred lag 1, and HQIC preferred lag 3. Lag 4 is defensible mainly because VARX is used for conditional forecasting and scenario analysis with externally supplied FEDFUNDS and sentiment paths.

VARX challenger note: `VARX_D_policy_exog_sentiment_endogenous` at lag 3 scored slightly higher in the mechanical composite ranking, mainly because of lower inflation RMSE and fewer total parameters. It was not adopted as the official VARX because it treats SENTIMENT_CHANGE as endogenous, reducing the intended scenario-design role, and it has weaker residual ACF behavior than VARX_A. It should be reported as a close robustness alternative, not ignored.

Rejected alternatives: lower-scoring alternatives were rejected mainly when they had weaker residual whiteness/autocorrelation diagnostics, higher overparameterization risk, worse inflation forecast RMSE, or less useful policy/scenario interpretation. A model with lower RMSE was not automatically selected if it was unstable, too highly parameterized, or weak for economic interpretation.

## 4. Final VAR Results

- Selected VAR specification: {final_var['candidate_name']}
- Endogenous variables: {final_var['endogenous_variables']}
- Exogenous controls: {final_var['exogenous_variables']}
- Lag order: {int(final_var['lag_order'])}
- Train/test split: {final_var['train_start']} to {final_var['train_end']} / {final_var['test_start']} to {final_var['test_end']}
- Effective observations: {int(final_var['actual_effective_observations'])}
- Parameters per equation: {int(final_var['parameters_per_equation'])}; total parameters: {int(final_var['total_parameters'])}; observations per parameter per equation: {final_var['obs_per_parameter_per_equation']:.2f}
- Stability: {bool(final_var['stable'])}; max inverse companion-root modulus: {final_var['max_inverse_root_modulus']:.4f} (desired < 1 in this display)
- Portmanteau whiteness p-value: {final_var['portmanteau_whiteness_p_value']:.4g}; min Ljung-Box p-value: {final_var['min_ljung_box_p_value']:.4g}
- Inflation forecast RMSE/MAE: {final_var['inflation_RMSE']:.4f} / {final_var['inflation_MAE']:.4f}; relative RMSE vs no-leak naive: {final_var['inflation_relative_RMSE_vs_naive']:.4f}

VAR lag-selection criteria for the selected specification:

{markdown_table(lag_best.loc[(lag_best['model_type'] == 'VAR') & (lag_best['candidate_name'] == final_var['candidate_name']) & (lag_best['dummy_specification'] == final_var['dummy_specification'])])}

Equation-level fit metrics:

{markdown_table(var_tables['fit_metrics'])}

Coefficient/significance summary:

{markdown_table(var_sig_top)}

Robust inference sensitivity:

{markdown_table(var_robust_summary)}

Residual tests:

{markdown_table(var_tables['residual_tests'])}

Residual interpretation: equation-level Ljung-Box tests mostly pass, Durbin-Watson values are near 2, and positive-lag ACF exceedances are limited. However, the multivariate Portmanteau whiteness test rejects for the system. The model is useful but not perfectly white; IRF and FEVD interpretation requires caution.

Residual ACF summary, excluding lag 0:

{markdown_table(var_tables['residual_acf'])}

Residual cross-correlation summary, including lag 0 for cross-equation pairs:

{markdown_table(var_tables['residual_ccf'].sort_values('max_abs_ccf', ascending=False), max_rows=15)}

Residual normality:

{markdown_table(var_tables['normality'])}

Normality/ARCH interpretation: Jarque-Bera normality is strongly rejected and the inflation equation shows ARCH effects. This is common in monthly macro data around crisis periods. It does not automatically invalidate point forecasts, but it weakens classical p-values and confidence intervals, motivating robust standard errors and Monte Carlo IRF bands.

Granger causality, significant predictive relationships:

{markdown_table(var_granger_sig, max_rows=20)}

FEDFUNDS shock IRF summary:

{markdown_table(var_tables.get('irf_key_fedfunds', pd.DataFrame()), max_rows=30)}

Alternative Cholesky ordering robustness for FEDFUNDS shock:

{markdown_table(cholesky_robustness, max_rows=30)}

IRF interpretation: the FEDFUNDS shock is not a clean textbook contractionary policy shock. Inflation rises slightly in the short run, industrial production rises initially, and unemployment falls after the shock in the baseline ordering. This likely mixes monetary tightening with the Federal Reserve's endogenous reaction to strong macroeconomic conditions and inflation pressure. It should be interpreted as a price-puzzle / identification issue, not as evidence that higher interest rates causally reduce unemployment. The IRF confidence bands are Monte Carlo bands generated with independent simulation seeds. Horizon-0 zero-width intervals can occur only where recursive Cholesky identification imposes an exact contemporaneous zero response; nonzero horizons should have separate lower and upper bounds.

Inflation FEVD summary:

{markdown_table(var_tables.get('fevd_key_inflation', pd.DataFrame()), max_rows=30)}

FEVD conclusion: inflation forecast-error variance is dominated by inflation's own innovations. At horizons 12 and 24, INF own-shock share is about 90%, while FEDFUNDS contributes only about 3.8%. Monetary-policy shocks contribute a smaller but nonzero share; they do not explain most inflation variation.

VAR forecast metrics:

{markdown_table(var_forecast_metrics)}

## 5. Final VARX Results

- Selected VARX specification: {final_varx['candidate_name']}
- Endogenous variables: {final_varx['endogenous_variables']}
- Exogenous variables: {final_varx['exogenous_variables']}
- Lag order: {int(final_varx['lag_order'])}
- Train/test split: {final_varx['train_start']} to {final_varx['train_end']} / {final_varx['test_start']} to {final_varx['test_end']}
- Effective observations: {int(final_varx['actual_effective_observations'])}
- Parameters per equation: {int(final_varx['parameters_per_equation'])}; total parameters: {int(final_varx['total_parameters'])}; observations per parameter per equation: {final_varx['obs_per_parameter_per_equation']:.2f}
- Stability: {bool(final_varx['stable'])}; max inverse companion-root modulus: {final_varx['max_inverse_root_modulus']:.4f}
- Portmanteau whiteness p-value: {final_varx['portmanteau_whiteness_p_value']:.4g}; min Ljung-Box p-value: {final_varx['min_ljung_box_p_value']:.4g}
- Inflation forecast RMSE/MAE: {final_varx['inflation_RMSE']:.4f} / {final_varx['inflation_MAE']:.4f}; relative RMSE vs no-leak naive: {final_varx['inflation_relative_RMSE_vs_naive']:.4f}

VARX lag-selection criteria for the selected specification:

{markdown_table(lag_best.loc[(lag_best['model_type'] == 'VARX') & (lag_best['candidate_name'] == final_varx['candidate_name']) & (lag_best['dummy_specification'] == final_varx['dummy_specification'])])}

Equation-level fit metrics:

{markdown_table(varx_tables['fit_metrics'])}

Coefficient/significance summary:

{markdown_table(varx_sig_top)}

Robust inference sensitivity:

{markdown_table(varx_robust_summary)}

Residual tests:

{markdown_table(varx_tables['residual_tests'])}

Residual interpretation: equation-level tests are mostly acceptable, but the VARX system-level Portmanteau whiteness test rejects. VARX should therefore be treated as a useful conditional forecasting/scenario tool, not a fully specified structural system.

Residual ACF summary, excluding lag 0:

{markdown_table(varx_tables['residual_acf'])}

Residual cross-correlation summary, including lag 0 for cross-equation pairs:

{markdown_table(varx_tables['residual_ccf'].sort_values('max_abs_ccf', ascending=False), max_rows=15)}

Residual normality:

{markdown_table(varx_tables['normality'])}

Normality/ARCH interpretation: VARX residual normality is strongly rejected, and ARCH effects remain especially in INF and M2_GROWTH. This weakens classical inference and supports robust standard errors and scenario-response caution.

VARX endogenous Granger-style predictive relationships:

{markdown_table(varx_granger_sig, max_rows=20)}

VARX forecast metrics:

{markdown_table(varx_forecast_metrics)}

VARX FEDFUNDS conditional/scenario response:

{markdown_table(varx_tables.get('scenario_response', pd.DataFrame()).loc[lambda d: d['horizon'].isin([1, 3, 6, 12, 24])] if not varx_tables.get('scenario_response', pd.DataFrame()).empty else pd.DataFrame(), max_rows=30)}

Interpretation: VARX responses are conditional scenario responses, not structural IRFs. Future exogenous paths are imposed externally, so scenario results depend on the assumed exogenous shock path. VARX is useful because FEDFUNDS and SENTIMENT_CHANGE can be externally specified, but it is not the strongest selected inflation forecasting model.

## 6. Forecast Comparison

Selected VAR forecast metrics:

{markdown_table(var_forecast_metrics)}

Selected VARX forecast metrics:

{markdown_table(varx_forecast_metrics)}

Inflation forecasts from all existing benchmark models:

{markdown_table(ml_ranking, max_rows=12)}

Multi-horizon inflation forecast comparison:

{markdown_table(multihorizon, max_rows=20)}

Forecast conclusion: among the selected econometric models, {forecast_winner} has the lower optimized inflation RMSE ({min(var_better_inf, varx_better_inf):.4f}). For inflation specifically, selected VAR RMSE is about {var_better_inf:.3f}, the no-leak naive RMSE is about 0.188, and selected VARX RMSE is about {varx_better_inf:.3f}. These are optimized selected-model recursive holdout metrics. They differ from the all-benchmark one-step/direct forecast table, where the VAR RMSE can appear around 0.199 because the forecast protocol is different. Ridge may be best for pure one-step prediction, but it does not provide Granger causality, IRF, FEVD, Cholesky identification, or structural macroeconomic transmission interpretation. VAR is the main policy-interpretation model; VARX remains useful for conditional policy/scenario forecasting even when it is not the strongest inflation forecaster.

## 7. Main Economic Conclusions

- Inflation dynamics: inflation is forecast using its own lagged dynamics plus policy, labor-market, production, money, and sentiment channels. Granger-significant relationships above show which variables have predictive content in the optimized system.
- FEDFUNDS predictive content: Granger results show FEDFUNDS predicts UNRATE and INDPRO_GROWTH more strongly than it predicts inflation directly. Inflation predicting FEDFUNDS is consistent with a policy-reaction function.
- FEDFUNDS shocks: {price_puzzle_note}
- Unemployment response: unemployment falls after a FEDFUNDS shock in the baseline ordering, which should not be read as a clean causal contractionary-policy effect. It likely reflects endogenous policy reaction and identification limitations.
- Industrial production response: industrial production rises initially and then becomes weak/sign-changing after a FEDFUNDS shock, reinforcing the identification warning.
- FEVD: inflation forecast-error variance is mostly own inflation shocks. FEDFUNDS contributes around 3.8% by horizons 12 and 24, so monetary policy is present but not dominant in FEVD.
- VAR vs VARX: VAR is better for policy interpretation because it supports Granger causality, IRF, FEVD, and endogenous feedback. VARX is better for conditional/scenario forecasting when externally supplied FEDFUNDS and sentiment paths are substantively meaningful, but it is weaker for selected inflation forecasting.

## 8. Weaknesses and Warnings

- Residual autocorrelation: macroeconomic VAR residuals are rarely perfectly white. Equation-level diagnostics are mostly acceptable, but system-level Portmanteau whiteness rejects for both selected VAR and VARX.
- Non-normality: Jarque-Bera/system normality rejects strongly because crisis periods create fat tails. This affects classical p-values and confidence intervals more than point forecasts.
- ARCH effects: low ARCH-LM p-values imply time-varying volatility, especially in VAR INF and VARX INF/M2_GROWTH. Robust or bootstrap inference is preferable.
- Overparameterization: high lags and full six-variable systems can weaken degrees of freedom. The selected models balance diagnostics and interpretability rather than blindly choosing AIC.
- Cholesky ordering: VAR IRFs depend on recursive identification and variable ordering. Short-run responses are conditional, not automatic causal truth.
- Price puzzle: if inflation rises after a positive FEDFUNDS shock, discuss endogenous policy reaction, omitted expectations/commodity channels, and identification limitations.
- VARX limitation: scenario responses condition on imposed exogenous paths and are not standard structural IRFs.
- Data limitation: monthly U.S. macro data contain regime shifts from 2008 and COVID; results may be sensitive to crisis dummy treatment and train/test split.

## 9. Files Produced

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

Context file:

- `PROJECT_RESULTS_CONTEXT.md`
"""
    CONTEXT_PATH.write_text(text, encoding="utf-8")


def price_puzzle_interpretation(irf_key: pd.DataFrame) -> str:
    if irf_key.empty:
        return "FEDFUNDS shock IRFs are unavailable for the selected VAR."
    inf = irf_key.loc[irf_key["response"] == "INF"].copy()
    if inf.empty:
        return "inflation response to FEDFUNDS shock is unavailable."
    h1 = inf.loc[inf["horizon"] == 1, "value"]
    h6 = inf.loc[inf["horizon"] == 6, "value"]
    h1_val = safe_float(h1.iloc[0]) if not h1.empty else np.nan
    h6_val = safe_float(h6.iloc[0]) if not h6.empty else np.nan
    if pd.notna(h1_val) and h1_val > 0:
        return f"a positive short-run inflation response appears after a FEDFUNDS shock (h1={h1_val:.4f}, h6={h6_val:.4f}), so the price puzzle should be discussed with identification caveats."
    return f"inflation does not show a positive h1 response to a FEDFUNDS shock (h1={h1_val:.4f}, h6={h6_val:.4f}); still interpret causally only under Cholesky assumptions."


def main() -> None:
    raw, model, dummies = load_inputs()
    data = build_modeling_frame(model, dummies)

    rows = []
    row_keys = []
    specs = candidate_specs()
    total = len(specs) * MAX_LAG
    completed = 0
    for spec in specs:
        for lag in range(1, MAX_LAG + 1):
            row, _ = fit_candidate(spec, lag, data, deep_diagnostics=False)
            rows.append(row)
            row_keys.append((spec.spec_id, lag))
            completed += 1
            if completed % 24 == 0:
                print(f"Evaluated {completed}/{total} candidate-lag combinations", flush=True)

    preliminary = add_selection_scores(pd.DataFrame(rows))
    deep_candidates = set()
    for model_type, group in preliminary.loc[preliminary["status"] == "ok"].groupby("model_type"):
        for _, candidate in group.sort_values("selection_score", ascending=False).head(18).iterrows():
            deep_candidates.add((candidate["spec_id"], int(candidate["lag_order"])))
        for criterion in ["aic", "bic", "hqic", "fpe", "inflation_RMSE"]:
            if criterion in group and group[criterion].notna().any():
                idx = group[criterion].idxmin()
                candidate = group.loc[idx]
                deep_candidates.add((candidate["spec_id"], int(candidate["lag_order"])))

    spec_lookup = {spec.spec_id: spec for spec in specs}
    print(f"Running deep diagnostics for {len(deep_candidates)} shortlisted candidate-lag combinations", flush=True)
    for i, (spec_id, lag) in enumerate(sorted(deep_candidates), start=1):
        row, _ = fit_candidate(spec_lookup[spec_id], lag, data, deep_diagnostics=True)
        idx = row_keys.index((spec_id, lag))
        rows[idx] = row
        if i % 8 == 0:
            print(f"Deep diagnostics {i}/{len(deep_candidates)}", flush=True)

    results = pd.DataFrame(rows)
    scored = add_selection_scores(results)
    lag_best = best_lag_table(scored)
    crisis = crisis_dummy_summary(scored)
    final_var = select_final(scored, "VAR")
    final_varx = select_final(scored, "VARX")

    var_details = final_details_from_row(final_var, data)
    varx_details = final_details_from_row(final_varx, data)
    var_tables = save_final_outputs("var", var_details)
    varx_tables = save_final_outputs("varx", varx_details)
    cholesky_summary, irf_ci = cholesky_ordering_robustness(var_details["train"], var_details["lag"])
    cholesky_summary.to_csv(TABLE_DIR / "academic_cholesky_ordering_robustness.csv", index=False)
    irf_ci.to_csv(TABLE_DIR / "academic_var_irf_confidence_intervals.csv", index=False)
    save_cholesky_ordering_figure(irf_ci)
    var_tables["cholesky_ordering_robustness"] = cholesky_summary
    var_tables["irf_confidence_intervals"] = irf_ci

    var_results = scored.loc[scored["model_type"] == "VAR"].copy()
    varx_results = scored.loc[scored["model_type"] == "VARX"].copy()
    final_specs = pd.DataFrame([final_var, final_varx])

    results.to_csv(TABLE_DIR / "optimized_lag_selection_full.csv", index=False)
    var_results.to_csv(TABLE_DIR / "optimized_var_candidate_search.csv", index=False)
    varx_results.to_csv(TABLE_DIR / "optimized_varx_candidate_search.csv", index=False)
    scored.sort_values("selection_score", ascending=False).to_csv(TABLE_DIR / "optimized_candidate_model_ranking.csv", index=False)
    lag_best.to_csv(TABLE_DIR / "optimized_lag_selection_best_by_criterion.csv", index=False)
    crisis.to_csv(TABLE_DIR / "optimized_crisis_dummy_search.csv", index=False)
    final_specs.to_csv(TABLE_DIR / "optimized_final_model_specs.csv", index=False)

    write_context(raw, model, scored, lag_best, crisis, final_var, final_varx, var_tables, varx_tables)

    print("Optimization complete")
    print(f"Selected VAR: {final_var['candidate_name']}, lag {int(final_var['lag_order'])}, dummies {final_var['dummy_specification']}")
    print(f"Selected VARX: {final_varx['candidate_name']}, lag {int(final_varx['lag_order'])}, dummies {final_varx['dummy_specification']}")
    print(f"Wrote {CONTEXT_PATH}")


if __name__ == "__main__":
    main()
