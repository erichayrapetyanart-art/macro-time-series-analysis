from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
TABLE_DIR = BASE_DIR / "outputs" / "tables"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"

MAX_UI_LAG = 8
MIN_TEST_OBS = 3


@dataclass
class ModelRun:
    model_type: str
    lag_order: int
    target: str
    train: pd.DataFrame
    test: pd.DataFrame
    endog: list[str]
    exog: list[str]
    residuals: pd.DataFrame
    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    fit_metrics: pd.DataFrame
    fit_info: pd.DataFrame
    parameter_table: pd.DataFrame
    stable: bool | None
    roots: pd.DataFrame
    whiteness_p_value: float | None
    normality_p_value: float | None
    warning: str


def read_indexed_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def dataframe_to_json(df: pd.DataFrame) -> str:
    return df.to_json(date_format="iso", orient="split")


def dataframe_from_json(payload: str) -> pd.DataFrame:
    df = pd.read_json(StringIO(payload), orient="split")
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df


def dynamic_max_lag(n_train: int, k_endog: int, k_exog: int) -> int:
    raw = max(1, (n_train - k_exog - 10) // max(2, k_endog * 4))
    return int(max(1, min(MAX_UI_LAG, raw)))


def parameter_count(
    k_endog: int,
    p_lag: int,
    k_exog: int,
    include_const: bool = True,
) -> tuple[int, int]:
    per_equation = k_endog * p_lag + k_exog + (1 if include_const else 0)
    return per_equation, per_equation * k_endog


def make_lagged_features(
    data: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    lag_order: int,
) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    idx = []
    for i in range(lag_order, len(data)):
        row = {}
        for lag in range(1, lag_order + 1):
            for col in feature_cols:
                row[f"{col}_lag{lag}"] = data[col].iloc[i - lag]
        rows.append(row)
        idx.append(data.index[i])
    x = pd.DataFrame(rows, index=idx)
    y = data[target].iloc[lag_order:].copy()
    y.index = x.index
    return x, y


def add_rule_column(table: pd.DataFrame, rule: str, position: int = 1) -> pd.DataFrame:
    if table.empty:
        return table
    result = table.copy()
    if "Acceptable if" not in result.columns:
        result.insert(min(position, len(result.columns)), "Acceptable if", rule)
    return result


def round_numeric(table: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    result = table.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].round(digits)
    return result


def lag_selection_summary(lag_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for criterion in ["AIC", "BIC", "HQIC", "FPE"]:
        if criterion not in lag_table.columns or lag_table[criterion].dropna().empty:
            continue
        idx = lag_table[criterion].idxmin()
        rows.append(
            {
                "criterion": criterion,
                "best_lag": int(lag_table.loc[idx, "lag"]),
                "criterion_value": lag_table.loc[idx, criterion],
                "Acceptable if": "lower information criterion is preferred; compare criteria against degrees-of-freedom limits",
            }
        )
    return pd.DataFrame(rows)
