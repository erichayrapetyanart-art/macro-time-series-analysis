from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf as sm_acf, adfuller, kpss, pacf as sm_pacf
import warnings


def safe_arch_pvalue(series: pd.Series) -> float:
    try:
        return float(het_arch(series.dropna(), nlags=min(12, max(1, len(series) // 5)))[1])
    except Exception:
        return np.nan


def acf_values(series: pd.Series, max_lag: int = 24) -> pd.DataFrame:
    lags = range(1, min(max_lag, len(series) - 2) + 1)
    values = [series.autocorr(lag=lag) for lag in lags]
    bound = 1.96 / np.sqrt(len(series))
    return pd.DataFrame(
        {
            "lag": list(lags),
            "acf": values,
            "upper": bound,
            "lower": -bound,
            "outside_bound": [abs(value) > bound for value in values],
        }
    )


def series_acf_pacf_values(series: pd.Series, max_lag: int = 36) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = series.dropna()
    nlags = min(max_lag, max(1, len(clean) // 2 - 1))
    bound = 1.96 / np.sqrt(len(clean))
    acf_raw = sm_acf(clean, nlags=nlags, fft=False)
    pacf_raw = sm_pacf(clean, nlags=nlags, method="ywm")
    lags = np.arange(1, nlags + 1)
    acf_df = pd.DataFrame(
        {
            "lag": lags,
            "value": acf_raw[1:],
            "upper": bound,
            "lower": -bound,
            "outside_bound": np.abs(acf_raw[1:]) > bound,
            "type": "ACF",
        }
    )
    pacf_df = pd.DataFrame(
        {
            "lag": lags,
            "value": pacf_raw[1:],
            "upper": bound,
            "lower": -bound,
            "outside_bound": np.abs(pacf_raw[1:]) > bound,
            "type": "PACF",
        }
    )
    return acf_df, pacf_df


def residual_normality_table(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in residuals.columns:
        series = residuals[column].dropna()
        jb = stats.jarque_bera(series)
        rows.append(
            {
                "equation": column,
                "test": "Jarque-Bera",
                "Acceptable if": "p-value > 0.05 for approximate normal residuals",
                "jarque_bera_stat": jb.statistic,
                "jarque_bera_p_value": jb.pvalue,
                "skewness": stats.skew(series, bias=False),
                "kurtosis_pearson": stats.kurtosis(series, fisher=False, bias=False),
                "reject_normality_at_5pct": jb.pvalue < 0.05,
            }
        )
    return pd.DataFrame(rows)


def residual_ccf_matrix(residuals: pd.DataFrame, max_lag: int = 12) -> pd.DataFrame:
    cols = list(residuals.columns)
    matrix = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for source in cols:
        for target in cols:
            max_lag_used = min(max_lag, len(residuals) - 2)
            if max_lag_used <= 0:
                matrix.loc[source, target] = np.nan
                continue
            if source == target:
                values = [residuals[target].autocorr(lag=lag) for lag in range(1, max_lag_used + 1)]
            else:
                values = [lagged_cross_correlation(residuals[source], residuals[target], lag) for lag in range(-max_lag_used, max_lag_used + 1)]
            matrix.loc[source, target] = np.nanmax(np.abs(values)) if values else np.nan
    return matrix


def residual_cross_correlation_summary(residuals: pd.DataFrame, max_lag: int = 12) -> pd.DataFrame:
    rows = []
    cols = list(residuals.columns)
    max_lag_used = min(max_lag, len(residuals) - 2)
    if max_lag_used <= 0:
        return pd.DataFrame()
    for source in cols:
        for target in cols:
            values = []
            lags = range(1, max_lag_used + 1) if source == target else range(-max_lag_used, max_lag_used + 1)
            for lag in lags:
                if source == target and lag > 0:
                    value = residuals[target].autocorr(lag=lag)
                else:
                    value = lagged_cross_correlation(residuals[source], residuals[target], lag)
                values.append(value)
            if not values or np.all(pd.isna(values)):
                continue
            values = np.asarray(values, dtype=float)
            lag_values = list(lags)
            max_pos = int(np.nanargmax(np.abs(values)))
            rows.append(
                {
                    "source_residual": source,
                    "target_residual": target,
                    "lag_of_max_abs_ccf": lag_values[max_pos],
                    "ccf_at_max_abs_lag": values[max_pos],
                    "max_abs_ccf": abs(values[max_pos]),
                    "Acceptable if": "cross-series CCF may include lag 0; autocorrelation diagnostics exclude lag 0",
                }
            )
    return pd.DataFrame(rows)


def lagged_cross_correlation(source: pd.Series, target: pd.Series, lag: int) -> float:
    if lag == 0:
        aligned = pd.concat([source, target], axis=1).dropna()
        return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    if lag > 0:
        left = source.iloc[:-lag]
        right = target.iloc[lag:]
    else:
        k = abs(lag)
        left = source.iloc[k:]
        right = target.iloc[:-k]
    return float(np.corrcoef(left, right)[0, 1])


def residual_cross_correlation_values(
    residuals: pd.DataFrame,
    source: str,
    target: str,
    max_lag: int = 12,
) -> pd.DataFrame:
    rows = []
    max_lag_used = min(max_lag, len(residuals) - 2)
    lags = range(1, max_lag_used + 1) if source == target else range(-max_lag_used, max_lag_used + 1)
    bound = 1.96 / np.sqrt(len(residuals))
    for lag in lags:
        if source == target and lag > 0:
            value = residuals[target].autocorr(lag=lag)
        else:
            value = lagged_cross_correlation(residuals[source], residuals[target], lag)
        rows.append(
            {
                "lag": lag,
                "cross_correlation": value,
                "upper": bound,
                "lower": -bound,
                "outside_bound": abs(value) > bound,
                "source_residual": source,
                "target_residual": target,
            }
        )
    return pd.DataFrame(rows)


def signed_ccf_heatmap(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return summary.pivot(
        index="source_residual",
        columns="target_residual",
        values="ccf_at_max_abs_lag",
    )


def kpss_stationarity_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in data.columns:
        series = data[column].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stat, p_value, used_lag, _ = kpss(series, regression="c", nlags="auto")
            except Exception:
                stat, p_value, used_lag = np.nan, np.nan, np.nan
        rows.append(
            {
                "variable": column,
                "test": "KPSS",
                "Acceptable if": "p-value > 0.05 supports stationarity",
                "kpss_statistic": stat,
                "p_value": p_value,
                "used_lag": used_lag,
                "stationary_at_5pct": p_value > 0.05 if pd.notna(p_value) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def integration_order_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in raw.columns:
        level = raw[column].dropna()
        diff = level.diff().dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                level_adf_p = adfuller(level, autolag="AIC")[1]
            except Exception:
                level_adf_p = np.nan
            try:
                level_kpss_p = kpss(level, regression="c", nlags="auto")[1]
            except Exception:
                level_kpss_p = np.nan
            try:
                diff_adf_p = adfuller(diff, autolag="AIC")[1]
            except Exception:
                diff_adf_p = np.nan
            try:
                diff_kpss_p = kpss(diff, regression="c", nlags="auto")[1]
            except Exception:
                diff_kpss_p = np.nan

        level_stationary = pd.notna(level_adf_p) and pd.notna(level_kpss_p) and level_adf_p < 0.05 and level_kpss_p > 0.05
        diff_stationary = pd.notna(diff_adf_p) and pd.notna(diff_kpss_p) and diff_adf_p < 0.05 and diff_kpss_p > 0.05
        if level_stationary:
            order = "I(0)"
        elif diff_stationary:
            order = "I(1)"
        else:
            order = "mixed/problematic"
        rows.append(
            {
                "variable": column,
                "integration_order": order,
                "level_adf_p_value": level_adf_p,
                "level_kpss_p_value": level_kpss_p,
                "diff_adf_p_value": diff_adf_p,
                "diff_kpss_p_value": diff_kpss_p,
                "Acceptable if": "I(0): level ADF p<0.05 and KPSS p>0.05; I(1): first difference satisfies both rules",
            }
        )
    return pd.DataFrame(rows)


def residual_test_table(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in residuals.columns:
        series = residuals[col].dropna()
        lb = acorr_ljungbox(series, lags=[min(12, max(1, len(series) // 5))], return_df=True)
        rows.append(
            {
                "equation": col,
                "test": "Durbin-Watson / Ljung-Box / ARCH-LM",
                "Acceptable if": "DW near 2; Ljung-Box p-value > 0.05; ARCH-LM p-value > 0.05",
                "durbin_watson": durbin_watson(series),
                "ljung_box_p_value": lb["lb_pvalue"].iloc[0],
                "arch_lm_p_value": safe_arch_pvalue(series),
            }
        )
    return pd.DataFrame(rows)
