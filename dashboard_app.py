from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TABLE_DIR = BASE_DIR / "outputs" / "tables"

PLOT_TEMPLATE = "plotly_white"
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
    fitted: object
    residuals: pd.DataFrame
    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    parameter_table: pd.DataFrame
    stable: bool | None
    roots: pd.DataFrame
    warning: str


st.set_page_config(
    page_title="VAR / VARX Macro Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.25rem; padding-bottom: 2.5rem;}
    div[data-testid="stMetric"] {background:#f7f9fb; border:1px solid #e6eaf0; padding:0.7rem; border-radius:8px;}
    .note {border-left:4px solid #315c7c; padding:0.75rem 0.95rem; background:#f6f8fa; margin:0.7rem 0 1rem 0;}
    .warn {border-left:4px solid #b45309; padding:0.75rem 0.95rem; background:#fff7ed; margin:0.7rem 0 1rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def read_indexed_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


@st.cache_data
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


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


def residual_ccf_matrix(residuals: pd.DataFrame, max_lag: int = 12) -> pd.DataFrame:
    cols = list(residuals.columns)
    matrix = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for source in cols:
        for target in cols:
            if source == target:
                values = [residuals[target].autocorr(lag=lag) for lag in range(1, min(max_lag, len(residuals) - 2) + 1)]
            else:
                values = []
                for lag in range(1, min(max_lag, len(residuals) - 2) + 1):
                    values.append(np.corrcoef(residuals[source].iloc[:-lag], residuals[target].iloc[lag:])[0, 1])
            matrix.loc[source, target] = np.nanmax(np.abs(values)) if values else np.nan
    return matrix


def parameter_count(k_endog: int, p_lag: int, k_exog: int, include_const: bool = True) -> tuple[int, int]:
    per_equation = k_endog * p_lag + k_exog + (1 if include_const else 0)
    return per_equation, per_equation * k_endog


def make_lagged_features(data: pd.DataFrame, feature_cols: list[str], target: str, lag_order: int) -> tuple[pd.DataFrame, pd.Series]:
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


def dynamic_max_lag(n_train: int, k_endog: int, k_exog: int) -> int:
    # Keep real-time experiments responsive and avoid extremely weak degrees of freedom.
    raw = max(1, (n_train - k_exog - 10) // max(2, k_endog * 4))
    return int(max(1, min(MAX_UI_LAG, raw)))


@st.cache_data(show_spinner=False)
def select_lags_cached(
    data_json: str,
    split_date: str,
    model_type: str,
    endog: tuple[str, ...],
    exog: tuple[str, ...],
    max_lag: int,
) -> pd.DataFrame:
    data = pd.read_json(StringIO(data_json), orient="split")
    data.index = pd.to_datetime(data.index)
    train = data.loc[:split_date]
    if model_type == "VAR":
        lag_sel = VAR(train[list(endog)]).select_order(maxlags=max_lag)
        return pd.DataFrame(
            {
                "lag": list(range(len(lag_sel.ics["aic"]))),
                "AIC": lag_sel.ics["aic"],
                "BIC": lag_sel.ics["bic"],
                "HQIC": lag_sel.ics["hqic"],
                "FPE": lag_sel.ics["fpe"],
            }
        )

    train_endog = train[list(endog)]
    train_exog = train[list(exog)] if exog else None
    if len(endog) >= 2:
        lag_sel = VAR(train_endog, exog=train_exog).select_order(maxlags=max_lag)
        return pd.DataFrame(
            {
                "lag": list(range(len(lag_sel.ics["aic"]))),
                "AIC": lag_sel.ics["aic"],
                "BIC": lag_sel.ics["bic"],
                "HQIC": lag_sel.ics["hqic"],
                "FPE": lag_sel.ics["fpe"],
            }
        )

    rows = []
    y = train_endog.iloc[:, 0]
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
                }
            )
        except Exception:
            rows.append({"lag": lag, "AIC": np.nan, "BIC": np.nan, "HQIC": np.nan, "FPE": np.nan})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def fit_model_cached(
    data_json: str,
    split_date: str,
    model_type: str,
    endog: tuple[str, ...],
    exog: tuple[str, ...],
    lag_order: int,
    target: str,
) -> dict:
    data = pd.read_json(StringIO(data_json), orient="split")
    data.index = pd.to_datetime(data.index)
    train = data.loc[:split_date].copy()
    test = data.loc[data.index > pd.Timestamp(split_date)].copy()

    if len(test) < MIN_TEST_OBS:
        raise ValueError("The test period is too short. Choose an earlier split date.")
    if lag_order >= len(train) - 5:
        raise ValueError("Lag order is too high for the available training sample.")

    warning = ""
    stable = None
    roots_df = pd.DataFrame()

    if model_type == "VAR":
        fitted = VAR(train[list(endog)]).fit(lag_order, trend="c")
        points, lower, upper = fitted.forecast_interval(train[list(endog)].values[-lag_order:], steps=len(test), alpha=0.05)
        columns = list(endog)
        forecasts = pd.DataFrame(points, index=test.index, columns=columns)
        lower_df = pd.DataFrame(lower, index=test.index, columns=[f"{col}_lower_95" for col in columns])
        upper_df = pd.DataFrame(upper, index=test.index, columns=[f"{col}_upper_95" for col in columns])
        residuals = fitted.resid
        stable = bool(fitted.is_stable())
        roots = 1 / fitted.roots
        roots_df = pd.DataFrame({"real": roots.real, "imag": roots.imag, "modulus": np.abs(roots)})
        params = fitted.params
        stderr = fitted.stderr
        pvalues = fitted.pvalues
        tvalues = fitted.tvalues
    else:
        if exog:
            train_exog = train[list(exog)]
            test_exog = test[list(exog)]
        else:
            train_exog = None
            test_exog = None

        if len(endog) >= 2:
            fitted = VAR(train[list(endog)], exog=train_exog).fit(lag_order, trend="c")
            points, lower, upper = fitted.forecast_interval(
                train[list(endog)].values[-lag_order:],
                steps=len(test),
                alpha=0.05,
                exog_future=test_exog.values if test_exog is not None else None,
            )
            columns = list(endog)
            forecasts = pd.DataFrame(points, index=test.index, columns=columns)
            lower_df = pd.DataFrame(lower, index=test.index, columns=[f"{col}_lower_95" for col in columns])
            upper_df = pd.DataFrame(upper, index=test.index, columns=[f"{col}_upper_95" for col in columns])
            residuals = fitted.resid
            stable = bool(fitted.is_stable())
            roots = 1 / fitted.roots
            roots_df = pd.DataFrame({"real": roots.real, "imag": roots.imag, "modulus": np.abs(roots)})
            params = fitted.params
            stderr = fitted.stderr
            pvalues = fitted.pvalues
            tvalues = fitted.tvalues
        else:
            y = train[list(endog)[0]]
            fitted = SARIMAX(
                y,
                exog=train_exog,
                order=(lag_order, 0, 0),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            pred = fitted.get_forecast(steps=len(test), exog=test_exog)
            conf = pred.conf_int()
            col = list(endog)[0]
            forecasts = pd.DataFrame({col: pred.predicted_mean}, index=test.index)
            lower_df = pd.DataFrame({f"{col}_lower_95": conf.iloc[:, 0].values}, index=test.index)
            upper_df = pd.DataFrame({f"{col}_upper_95": conf.iloc[:, 1].values}, index=test.index)
            residuals = pd.DataFrame({col: fitted.resid.dropna()})
            stable = bool(np.all(np.abs(1 / fitted.arroots) < 1)) if len(fitted.arroots) else None
            roots = 1 / fitted.arroots if len(fitted.arroots) else np.array([])
            roots_df = pd.DataFrame({"real": roots.real, "imag": roots.imag, "modulus": np.abs(roots)})
            params = pd.DataFrame({col: fitted.params})
            stderr = pd.DataFrame({col: fitted.bse})
            pvalues = pd.DataFrame({col: fitted.pvalues})
            tvalues = pd.DataFrame({col: fitted.tvalues})
            warning = "Single-endogenous VARX is estimated as an ARX/SARIMAX-style conditional equation; IRF/FEVD are not available."

    forecasts = pd.concat([forecasts, lower_df, upper_df], axis=1)
    rows = []
    for col in endog:
        if col in forecasts and col in test:
            rows.append(
                {
                    "variable": col,
                    "rmse": rmse(test[col], forecasts[col]),
                    "mae": float(mean_absolute_error(test[col], forecasts[col])),
                }
            )
    metrics = pd.DataFrame(rows)

    parameter_rows = []
    for param in params.index:
        for equation in params.columns:
            coef = params.loc[param, equation]
            se = stderr.loc[param, equation]
            parameter_rows.append(
                {
                    "equation": equation,
                    "parameter": param,
                    "coefficient": coef,
                    "std_error": se,
                    "t_stat": tvalues.loc[param, equation],
                    "p_value": pvalues.loc[param, equation],
                    "lower_95": coef - 1.96 * se,
                    "upper_95": coef + 1.96 * se,
                    "significant_at_5pct": pvalues.loc[param, equation] < 0.05,
                }
            )
    parameter_table = pd.DataFrame(parameter_rows)

    return {
        "train": train.to_json(date_format="iso", orient="split"),
        "test": test.to_json(date_format="iso", orient="split"),
        "forecasts": forecasts.to_json(date_format="iso", orient="split"),
        "residuals": residuals.to_json(date_format="iso", orient="split"),
        "metrics": metrics.to_json(orient="split"),
        "parameter_table": parameter_table.to_json(orient="split"),
        "stable": stable,
        "roots": roots_df.to_json(orient="split"),
        "warning": warning,
    }


def unpack_model_run(payload: dict, model_type: str, lag_order: int, target: str, endog: list[str], exog: list[str]) -> ModelRun:
    return ModelRun(
        model_type=model_type,
        lag_order=lag_order,
        target=target,
        train=pd.read_json(StringIO(payload["train"]), orient="split"),
        test=pd.read_json(StringIO(payload["test"]), orient="split"),
        endog=endog,
        exog=exog,
        fitted=None,
        residuals=pd.read_json(StringIO(payload["residuals"]), orient="split"),
        forecasts=pd.read_json(StringIO(payload["forecasts"]), orient="split"),
        metrics=pd.read_json(StringIO(payload["metrics"]), orient="split"),
        parameter_table=pd.read_json(StringIO(payload["parameter_table"]), orient="split"),
        stable=payload["stable"],
        roots=pd.read_json(StringIO(payload["roots"]), orient="split"),
        warning=payload["warning"],
    )


def plot_time_series(df: pd.DataFrame, columns: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for col in columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    fig.update_layout(title=title, template=PLOT_TEMPLATE, height=480, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_forecasts(run: ModelRun) -> go.Figure:
    target = run.target
    fig = go.Figure()
    history = run.train[target].tail(60)
    fig.add_trace(go.Scatter(x=history.index, y=history, mode="lines", name="Training history", line=dict(color="#6b7280")))
    fig.add_trace(go.Scatter(x=run.test.index, y=run.test[target], mode="lines+markers", name="Actual test", line=dict(color="black", width=2.5)))
    fig.add_trace(go.Scatter(x=run.forecasts.index, y=run.forecasts[target], mode="lines+markers", name=f"{run.model_type} forecast"))
    lower = f"{target}_lower_95"
    upper = f"{target}_upper_95"
    if lower in run.forecasts and upper in run.forecasts:
        fig.add_trace(go.Scatter(x=run.forecasts.index, y=run.forecasts[upper], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=run.forecasts.index, y=run.forecasts[lower], mode="lines", fill="tonexty", fillcolor="rgba(49, 92, 124, 0.18)", line=dict(width=0), name="95% forecast interval", hoverinfo="skip"))
    fig.update_layout(title=f"Out-of-Sample Forecast for {target}", template=PLOT_TEMPLATE, height=520, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_lag_table(lag_table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in ["AIC", "BIC", "HQIC"]:
        if col in lag_table:
            fig.add_trace(go.Scatter(x=lag_table["lag"], y=lag_table[col], mode="lines+markers", name=col))
    fig.update_layout(title="Lag Selection Criteria", template=PLOT_TEMPLATE, height=420)
    return fig


def plot_heatmap(matrix: pd.DataFrame, title: str, colorscale: str = "RdBu", zmin: float | None = None, zmax: float | None = None) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            text=np.round(matrix.values, 3),
            texttemplate="%{text}",
            colorbar=dict(thickness=14),
        )
    )
    fig.update_layout(title=title, template=PLOT_TEMPLATE, height=520, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def ml_benchmark(data: pd.DataFrame, split_date: pd.Timestamp, feature_cols: list[str], target: str, lag_order: int) -> pd.DataFrame:
    x, y = make_lagged_features(data, feature_cols, target, lag_order)
    train_mask = x.index <= split_date
    test_mask = x.index > split_date
    if train_mask.sum() < 20 or test_mask.sum() < MIN_TEST_OBS:
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
        pred = model.predict(x_test)
        rows.append({"model": name, "rmse": rmse(y_test, pred), "mae": float(mean_absolute_error(y_test, pred))})
    return pd.DataFrame(rows).sort_values("rmse")


raw = read_indexed_csv(DATA_DIR / "raw_fred_macro.csv")
model_df = read_indexed_csv(DATA_DIR / "academic_model_data.csv")
architecture = read_csv(TABLE_DIR / "academic_final_model_architecture.csv")
ordering = read_csv(TABLE_DIR / "academic_cholesky_ordering.csv")
baseline_ranking = read_csv(TABLE_DIR / "academic_all_model_forecast_ranking.csv")


st.title("VAR / VARX Macroeconomic Forecasting Dashboard")
st.caption("Official baseline results plus interactive sensitivity analysis for VAR and VARX specifications.")

baseline_best = baseline_ranking.iloc[0]
baseline_var_lag = architecture.loc[architecture["model"] == "VAR", "lag_order"].iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Raw observations", f"{len(raw):,}")
col2.metric("Model observations", f"{len(model_df):,}")
col3.metric("Official VAR lag", str(baseline_var_lag))
col4.metric("Best baseline RMSE", f"{baseline_best['model']} ({baseline_best['rmse']:.3f})")

tabs = st.tabs(
    [
        "Official Baseline",
        "Interactive Model Lab",
        "Forecast Comparison",
        "Diagnostics",
        "VAR/SVAR Interpretation",
        "Data Explorer",
    ]
)

with tabs[0]:
    st.subheader("Official Baseline Specification")
    st.markdown(
        """
        <div class="note">
        The notebook preserves one fixed official baseline for academic interpretation and reproducibility.
        The dashboard's interactive lab is a sensitivity-analysis tool, not a replacement for the reported baseline.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(architecture, width="stretch", hide_index=True)
    st.subheader("Cholesky Ordering for Baseline IRF/FEVD")
    st.dataframe(ordering, width="stretch", hide_index=True)
    st.subheader("Baseline Model Ranking")
    st.dataframe(baseline_ranking.round(4), width="stretch", hide_index=True)

with tabs[1]:
    st.subheader("Interactive VAR / VARX Sensitivity Lab")
    st.markdown(
        """
        <div class="note">
        Change the split date, variables, model type, and lag order. The dashboard refits the model and updates forecasts,
        residual diagnostics, stability, and parameter significance. VARX forecasts are conditional on the supplied future exogenous paths.
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_vars = list(model_df.columns)
    default_split = model_df.index[-37].date()
    split_date = st.date_input(
        "Train/test split date: train on observations <= date, test on observations > date",
        value=default_split,
        min_value=model_df.index[60].date(),
        max_value=model_df.index[-MIN_TEST_OBS - 1].date(),
    )
    split_ts = pd.Timestamp(split_date)
    model_type = st.radio("Model type", ["VAR", "VARX"], horizontal=True)

    default_endog = ["FEDFUNDS", "INF", "UNRATE", "INDPRO_GROWTH", "M2_GROWTH", "SENTIMENT_CHANGE"] if model_type == "VAR" else ["INF", "UNRATE", "INDPRO_GROWTH", "M2_GROWTH"]
    endog = st.multiselect("Endogenous variables", all_vars, default=default_endog)

    exog: list[str] = []
    if model_type == "VARX":
        available_exog = [var for var in all_vars if var not in endog]
        exog = st.multiselect("Exogenous variables for VARX", available_exog, default=[var for var in ["FEDFUNDS", "SENTIMENT_CHANGE"] if var in available_exog])
        overlap = sorted(set(endog) & set(exog))
        if overlap:
            st.error(f"Variables cannot be both endogenous and exogenous: {overlap}")
            st.stop()

    train_n = int((model_df.index <= split_ts).sum())
    test_n = int((model_df.index > split_ts).sum())
    min_endog = 2 if model_type == "VAR" else 1
    if len(endog) < min_endog:
        st.error(f"{model_type} requires at least {min_endog} endogenous variable(s).")
        st.stop()
    if test_n < MIN_TEST_OBS:
        st.error("The test set is too short. Choose an earlier split date.")
        st.stop()

    dyn_max_lag = dynamic_max_lag(train_n, len(endog), len(exog))
    max_lag = st.slider("Maximum lag considered for selection", 1, dyn_max_lag, min(4, dyn_max_lag))
    data_json = model_df.to_json(date_format="iso", orient="split")
    with st.spinner("Selecting lag order..."):
        lag_table = select_lags_cached(data_json, str(split_ts.date()), model_type, tuple(endog), tuple(exog), max_lag)
    lag_table = lag_table.dropna(subset=["AIC", "BIC", "HQIC"], how="all")
    st.plotly_chart(plot_lag_table(lag_table), width="stretch", key="lab_lag_selection_plot")

    selected_by = {}
    for criterion in ["AIC", "BIC", "HQIC"]:
        if criterion in lag_table and lag_table[criterion].notna().any():
            selected_by[criterion] = int(lag_table.loc[lag_table[criterion].idxmin(), "lag"])
    st.write("Selected lag by criterion:", selected_by)
    default_lag = max(1, selected_by.get("AIC", 1))
    lag_order = st.slider("Manual lag order used for refitting", 1, max_lag, min(default_lag, max_lag))

    target_options = endog
    target = st.selectbox("Forecast target shown in plots", target_options, index=target_options.index("INF") if "INF" in target_options else 0)

    per_eq, total_params = parameter_count(len(endog), lag_order, len(exog))
    dof_ratio = train_n / max(total_params, 1)
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Training obs.", f"{train_n}")
    col_b.metric("Test obs.", f"{test_n}")
    col_c.metric("Parameters/equation", f"{per_eq}")
    col_d.metric("Obs./total parameter", f"{dof_ratio:.2f}")
    if dof_ratio < 5 or per_eq > train_n / 4:
        st.warning("Overparameterization risk: parameter count is high relative to the training sample. Forecasts and stability may be unreliable.")

    with st.spinner("Refitting model..."):
        try:
            payload = fit_model_cached(data_json, str(split_ts.date()), model_type, tuple(endog), tuple(exog), lag_order, target)
            run = unpack_model_run(payload, model_type, lag_order, target, endog, exog)
        except Exception as exc:
            st.error(f"Model could not be fitted: {exc}")
            st.stop()

    st.session_state["interactive_run"] = run
    if run.warning:
        st.warning(run.warning)
    if run.stable is False:
        st.error("The fitted system is unstable. Interpret forecasts, IRFs, and FEVD with extreme caution.")
    elif run.stable is True:
        st.success("The fitted dynamic system is stable under the selected specification.")

    st.subheader("Interactive Forecast")
    st.plotly_chart(plot_forecasts(run), width="stretch", key="lab_forecast_plot")
    st.dataframe(run.metrics.round(4), width="stretch", hide_index=True)

    ml_features = sorted(set(endog + exog))
    if target in model_df.columns:
        ml = ml_benchmark(model_df, split_ts, ml_features, target, lag_order)
        if not ml.empty:
            st.subheader("ML Benchmark for Same Split and Target")
            merged_metrics = pd.concat(
                [
                    run.metrics.assign(model=model_type).rename(columns={"variable": "target"}),
                    ml.assign(target=target),
                ],
                ignore_index=True,
                sort=False,
            )
            st.dataframe(merged_metrics.round(4), width="stretch", hide_index=True)

with tabs[2]:
    st.subheader("Forecast Comparison")
    run = st.session_state.get("interactive_run")
    if run is None:
        st.info("Configure and run a model in the Interactive Model Lab tab first.")
    else:
        st.plotly_chart(plot_forecasts(run), width="stretch", key="comparison_forecast_plot")
        st.dataframe(run.metrics.round(4), width="stretch", hide_index=True)
        st.markdown(
            """
            <div class="note">
            VAR forecasts are generated from endogenous joint dynamics. VARX forecasts condition on observed or assumed future exogenous paths.
            This is useful for scenario forecasting, but it is not a pure real-time forecast unless the future exogenous variables are themselves forecasted.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tabs[3]:
    st.subheader("Interactive Residual Diagnostics")
    run = st.session_state.get("interactive_run")
    if run is None:
        st.info("Configure and run a model in the Interactive Model Lab tab first.")
    else:
        st.plotly_chart(
            plot_time_series(run.residuals, list(run.residuals.columns), "Residual Time Series"),
            width="stretch",
            key="diagnostics_residual_timeseries",
        )
        equation = st.selectbox("Equation for residual ACF", list(run.residuals.columns))
        acf_df = acf_values(run.residuals[equation], max_lag=24)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=acf_df["lag"], y=acf_df["acf"], marker_color=np.where(acf_df["outside_bound"], "#b45309", "#315c7c"), name="ACF"))
        fig.add_trace(go.Scatter(x=acf_df["lag"], y=acf_df["upper"], mode="lines", line=dict(color="red", dash="dash"), name="95% bound"))
        fig.add_trace(go.Scatter(x=acf_df["lag"], y=acf_df["lower"], mode="lines", line=dict(color="red", dash="dash"), showlegend=False))
        fig.update_layout(title=f"Residual ACF: {equation}", template=PLOT_TEMPLATE, height=420)
        st.plotly_chart(fig, width="stretch", key=f"diagnostics_acf_{equation}")

        ccf = residual_ccf_matrix(run.residuals, max_lag=12)
        st.plotly_chart(
            plot_heatmap(ccf, "Max Absolute Residual Cross-Correlation by Equation Pair", colorscale="Blues", zmin=0),
            width="stretch",
            key="diagnostics_ccf_heatmap",
        )

        diag_rows = []
        for col in run.residuals.columns:
            lb = acorr_ljungbox(run.residuals[col].dropna(), lags=[min(12, max(1, len(run.residuals) // 5))], return_df=True)
            diag_rows.append(
                {
                    "equation": col,
                    "durbin_watson": durbin_watson(run.residuals[col].dropna()),
                    "ljung_box_p_value": lb["lb_pvalue"].iloc[0],
                    "arch_lm_p_value": safe_arch_pvalue(run.residuals[col]),
                }
            )
        st.dataframe(pd.DataFrame(diag_rows).round(4), width="stretch", hide_index=True)
        st.markdown(
            """
            <div class="note">
            Macro VAR residuals are rarely perfectly white. The practical goal is reasonable adequacy, reduced serial dependence,
            and transparent discussion of remaining limitations.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tabs[4]:
    st.subheader("VAR/SVAR Interpretation")
    run = st.session_state.get("interactive_run")
    if run is None:
        st.info("Configure and run a VAR model in the Interactive Model Lab tab first.")
    elif run.model_type != "VAR" or len(run.endog) < 2:
        st.info("IRF and FEVD are available only for VAR specifications with at least two endogenous variables.")
    else:
        st.markdown(
            """
            <div class="warn">
            IRFs depend strongly on identification assumptions. Cholesky ordering changes short-run responses,
            so compare alternative orderings before making strong policy claims.
            </div>
            """,
            unsafe_allow_html=True,
        )
        order_text = st.text_input("Cholesky order, comma-separated", value=", ".join(run.endog))
        order = [x.strip() for x in order_text.split(",") if x.strip()]
        if sorted(order) != sorted(run.endog):
            st.error("Ordering must contain exactly the selected endogenous variables once.")
        elif run.stable is False:
            st.error("IRF/FEVD hidden because the selected VAR is unstable.")
        else:
            try:
                ordered_train = run.train[order]
                fitted = VAR(ordered_train).fit(run.lag_order, trend="c")
                horizon = st.slider("IRF/FEVD horizon", 1, 36, 24)
                irf = fitted.irf(horizon)
                paths = []
                for response_idx, response in enumerate(order):
                    for shock_idx, shock in enumerate(order):
                        for h, value in enumerate(irf.orth_irfs[:, response_idx, shock_idx]):
                            paths.append({"response": response, "shock": shock, "horizon": h, "value": value})
                irf_df = pd.DataFrame(paths)
                shock = st.selectbox("Shock", order, index=0)
                responses = st.multiselect("Responses", order, default=order[: min(4, len(order))])
                fig = go.Figure()
                for response in responses:
                    subset = irf_df.query("shock == @shock and response == @response")
                    fig.add_trace(go.Scatter(x=subset["horizon"], y=subset["value"], mode="lines+markers", name=response))
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.update_layout(title=f"Orthogonalized IRF to {shock} Shock", template=PLOT_TEMPLATE, height=500)
                st.plotly_chart(fig, width="stretch", key=f"irf_{impulse}_{response}")

                fevd = fitted.fevd(horizon)
                fevd_rows = []
                for response_idx, response in enumerate(order):
                    for h in range(1, horizon + 1):
                        for shock_idx, shock_name in enumerate(order):
                            fevd_rows.append({"response": response, "horizon": h, "shock": shock_name, "variance_share": fevd.decomp[response_idx, h - 1, shock_idx]})
                fevd_df = pd.DataFrame(fevd_rows)
                fevd_response = st.selectbox("FEVD response", order, index=order.index(run.target) if run.target in order else 0)
                fig = px.area(fevd_df.query("response == @fevd_response"), x="horizon", y="variance_share", color="shock", template=PLOT_TEMPLATE, title=f"FEVD for {fevd_response}")
                fig.update_layout(height=500, yaxis_tickformat=".0%")
                st.plotly_chart(fig, width="stretch", key=f"fevd_{fevd_var}")
            except Exception as exc:
                st.error(f"IRF/FEVD could not be computed: {exc}")

with tabs[5]:
    st.subheader("Data Explorer")
    dataset = st.radio("Dataset", ["Transformed model data", "Raw FRED data"], horizontal=True)
    df = model_df if dataset == "Transformed model data" else raw
    default_vars = ["INF", "UNRATE", "FEDFUNDS"] if dataset == "Transformed model data" else ["CPI", "UNRATE", "FEDFUNDS"]
    variables = st.multiselect("Variables", list(df.columns), default=[v for v in default_vars if v in df.columns])
    if variables:
        st.plotly_chart(plot_time_series(df, variables, "Selected Time Series"), width="stretch", key="explorer_time_series")
        st.plotly_chart(
            plot_heatmap(df[variables].corr(), "Correlation Matrix", colorscale="RdBu", zmin=-1, zmax=1),
            width="stretch",
            key="explorer_correlation_heatmap",
        )
