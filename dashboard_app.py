from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from statsmodels.tsa.api import VAR

from src.data import load_project_data as load_project_data_uncached, modeling_frame
from src.dashboard_helpers import (
    MIN_TEST_OBS,
    ModelRun,
    add_rule_column,
    dataframe_from_json,
    dataframe_to_json,
    dynamic_max_lag,
    lag_selection_summary,
    parameter_count,
    round_numeric,
)
from src.diagnostics import (
    acf_values,
    integration_order_table,
    kpss_stationarity_table,
    residual_cross_correlation_summary,
    residual_cross_correlation_values,
    residual_normality_table,
    residual_test_table,
    series_acf_pacf_values,
    signed_ccf_heatmap,
)
from src.forecasting import combined_inflation_forecasts, official_var_varx_forecasts
from src.models_var import equation_fit_metrics, fit_var_system, select_var_lags, var_fevd_paths, var_irf_paths
from src.models_varx import fit_varx_system, select_varx_lags, varx_exogenous_scenario_response
from src.visualization import (
    plot_acf,
    plot_correlogram,
    plot_forecast_lines,
    plot_forecast_run,
    plot_granger_heatmap,
    plot_heatmap,
    plot_lag_table,
    plot_cross_correlation_lags,
    plot_residual_distribution,
    plot_response_grid,
    plot_time_series,
    plot_var_varx_forecast,
)


st.set_page_config(page_title="VAR / VARX Macro Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2.5rem;}
    div[data-testid="stMetric"] {
        background: color-mix(in srgb, var(--background-color) 86%, var(--text-color) 14%);
        border: 1px solid color-mix(in srgb, var(--background-color) 70%, var(--text-color) 30%);
        padding: 0.7rem;
        border-radius: 8px;
    }
    .note {
        border-left: 4px solid #3b82f6;
        padding: 0.75rem 0.95rem;
        background: color-mix(in srgb, var(--background-color) 90%, #3b82f6 10%);
        margin: 0.7rem 0 1rem 0;
    }
    .warn {
        border-left: 4px solid #d97706;
        padding: 0.75rem 0.95rem;
        background: color-mix(in srgb, var(--background-color) 88%, #d97706 12%);
        margin: 0.7rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


VAR_COLUMNS = ["FEDFUNDS", "INF", "UNRATE", "INDPRO_GROWTH", "M2_GROWTH", "SENTIMENT_CHANGE"]
VARX_ENDOG = ["INF", "UNRATE", "INDPRO_GROWTH", "M2_GROWTH"]
VARX_EXOG = ["FEDFUNDS", "SENTIMENT_CHANGE", "D_2008", "D_COVID"]


@st.cache_data
def load_project_data() -> dict[str, pd.DataFrame]:
    return load_project_data_uncached()


@st.cache_data(show_spinner=False)
def select_lags_cached(
    data_json: str,
    split_date: str,
    model_type: str,
    endog: tuple[str, ...],
    exog: tuple[str, ...],
    max_lag: int,
) -> pd.DataFrame:
    data = dataframe_from_json(data_json)
    train = data.loc[:split_date]
    if model_type == "VAR":
        return select_var_lags(train[list(endog)], max_lag)
    train_exog = train[list(exog)] if exog else None
    return select_varx_lags(train[list(endog)], train_exog, max_lag)


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
    data = dataframe_from_json(data_json)
    if model_type == "VAR":
        result = fit_var_system(data, split_date, list(endog), lag_order, target)
    else:
        result = fit_varx_system(data, split_date, list(endog), list(exog), lag_order, target)
    return {
        "train": dataframe_to_json(result["train"]),
        "test": dataframe_to_json(result["test"]),
        "residuals": dataframe_to_json(result["residuals"]),
        "forecasts": dataframe_to_json(result["forecasts"]),
        "metrics": result["metrics"].to_json(orient="split"),
        "fit_metrics": result["fit_metrics"].to_json(orient="split"),
        "fit_info": result["fit_info"].to_json(orient="split"),
        "parameter_table": result["parameter_table"].to_json(orient="split"),
        "stable": result["stable"],
        "roots": result["roots"].to_json(orient="split"),
        "whiteness_p_value": result["whiteness_p_value"],
        "normality_p_value": result["normality_p_value"],
        "warning": result.get("warning", ""),
    }


def unpack_run(
    payload: dict,
    model_type: str,
    lag_order: int,
    target: str,
    endog: list[str],
    exog: list[str],
) -> ModelRun:
    return ModelRun(
        model_type=model_type,
        lag_order=lag_order,
        target=target,
        train=dataframe_from_json(payload["train"]),
        test=dataframe_from_json(payload["test"]),
        endog=endog,
        exog=exog,
        residuals=dataframe_from_json(payload["residuals"]),
        forecasts=dataframe_from_json(payload["forecasts"]),
        metrics=pd.read_json(StringIO(payload["metrics"]), orient="split"),
        fit_metrics=pd.read_json(StringIO(payload["fit_metrics"]), orient="split"),
        fit_info=pd.read_json(StringIO(payload["fit_info"]), orient="split"),
        parameter_table=pd.read_json(StringIO(payload["parameter_table"]), orient="split"),
        stable=payload["stable"],
        roots=pd.read_json(StringIO(payload["roots"]), orient="split"),
        whiteness_p_value=payload.get("whiteness_p_value"),
        normality_p_value=payload.get("normality_p_value"),
        warning=payload.get("warning", ""),
    )


@st.cache_data(show_spinner=False)
def baseline_residual_run(model_type: str) -> dict:
    data = load_project_data()
    model_df = data["model"]
    dummies = data["dummies"]
    lag = 4
    if model_type == "VAR":
        fitted = VAR(model_df[VAR_COLUMNS], exog=dummies[["D_2008", "D_COVID"]]).fit(lag, trend="c")
    else:
        exog = pd.concat([model_df[["FEDFUNDS", "SENTIMENT_CHANGE"]], dummies], axis=1)
        fitted = VAR(model_df[VARX_ENDOG], exog=exog).fit(lag, trend="c")
    try:
        whiteness_p = float(fitted.test_whiteness(nlags=12).pvalue)
    except Exception:
        whiteness_p = None
    try:
        normality_p = float(fitted.test_normality().pvalue)
    except Exception:
        normality_p = None
    return {
        "residuals": dataframe_to_json(fitted.resid),
        "stable": bool(fitted.is_stable()),
        "whiteness_p": whiteness_p,
        "normality_p": normality_p,
    }


@st.cache_data(show_spinner=False)
def stationarity_payload(raw_json: str, model_json: str) -> dict:
    raw_df = dataframe_from_json(raw_json)
    model_data = dataframe_from_json(model_json)
    return {
        "raw_kpss": kpss_stationarity_table(raw_df).to_json(orient="split"),
        "model_kpss": kpss_stationarity_table(model_data).to_json(orient="split"),
        "integration": integration_order_table(raw_df).to_json(orient="split"),
    }


@st.cache_data(show_spinner=False)
def official_forecasts_payload(model_json: str, dummies_json: str) -> dict:
    model_data = dataframe_from_json(model_json)
    dummy_data = dataframe_from_json(dummies_json)
    forecasts, metrics = official_var_varx_forecasts(
        model_data,
        dummy_data,
        VAR_COLUMNS,
        VARX_ENDOG,
        VARX_EXOG,
        lag_order=4,
        test_months=36,
    )
    return {
        "forecasts": dataframe_to_json(forecasts.set_index("date")),
        "metrics": metrics.to_json(orient="split"),
    }


@st.cache_data(show_spinner=False)
def direct_model_results(model_type: str) -> dict:
    project = load_project_data()
    model_data = project["model"]
    dummy_data = project["dummies"]
    if model_type == "VAR":
        fitted = VAR(model_data[VAR_COLUMNS], exog=dummy_data[["D_2008", "D_COVID"]]).fit(4, trend="c")
        endog_data = model_data[VAR_COLUMNS]
    else:
        exog_data = pd.concat([model_data[["FEDFUNDS", "SENTIMENT_CHANGE"]], dummy_data], axis=1)
        fitted = VAR(model_data[VARX_ENDOG], exog=exog_data[VARX_EXOG]).fit(4, trend="c")
        endog_data = model_data[VARX_ENDOG]
    roots = 1 / fitted.roots
    roots_df = pd.DataFrame({"real": roots.real, "imag": roots.imag, "modulus": np.abs(roots)})
    return {
        "fit": equation_fit_metrics(fitted, endog_data).to_json(orient="split"),
        "roots": roots_df.to_json(orient="split"),
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
        "hqic": float(fitted.hqic),
        "stable": bool(fitted.is_stable()),
        "nobs": int(fitted.nobs),
        "params_per_equation": int(fitted.params.shape[0]),
        "total_params": int(fitted.params.shape[0] * len(fitted.names)),
    }


def test_table(table: pd.DataFrame, rule: str) -> pd.DataFrame:
    return round_numeric(add_rule_column(table, rule))


def interactive_model_control_form(key_prefix: str, title: str) -> tuple[ModelRun | None, pd.DataFrame]:
    st.subheader(title)
    st.markdown(
        """
        This control form is for sensitivity analysis. It does not replace the official baseline specification
        reported in the notebook; it shows how results change when the train/test split, variables, and lag order change.
        """
    )
    container = st.container(border=True)
    run: ModelRun | None = None
    lag_table = pd.DataFrame()
    with container:
        endog_options = list(model_df.columns)
        exog_options = list(modeling_df.columns)
        default_split = model_df.index[-37].date()
        c1, c2 = st.columns(2)
        split_date = c1.date_input(
            "Train/test split",
            value=default_split,
            min_value=model_df.index[60].date(),
            max_value=model_df.index[-MIN_TEST_OBS - 1].date(),
            key=f"{key_prefix}_split_date",
        )
        model_type = c2.radio(
            "Model type",
            ["VAR", "VARX"],
            horizontal=True,
            key=f"{key_prefix}_model_type",
        )

        default_endog = VAR_COLUMNS if model_type == "VAR" else VARX_ENDOG
        endog = st.multiselect(
            "Endogenous variables",
            endog_options,
            default=default_endog,
            key=f"{key_prefix}_{model_type}_endog",
        )

        exog: list[str] = []
        if model_type == "VARX":
            available_exog = [var for var in exog_options if var not in endog]
            exog = st.multiselect(
                "VARX exogenous variables",
                available_exog,
                default=[var for var in VARX_EXOG if var in available_exog],
                key=f"{key_prefix}_varx_exog",
            )

        valid_selection = True
        if len(set(endog) & set(exog)):
            st.error("A variable cannot be both endogenous and exogenous.")
            valid_selection = False
        if model_type == "VAR" and len(endog) < 2:
            st.error("VAR requires at least two endogenous variables.")
            valid_selection = False
        if model_type == "VARX" and len(endog) < 1:
            st.error("VARX requires at least one endogenous variable.")
            valid_selection = False

        split_ts = pd.Timestamp(split_date)
        train_n = int((model_df.index <= split_ts).sum())
        test_n = int((model_df.index > split_ts).sum())
        dyn_max_lag = dynamic_max_lag(train_n, len(endog) if endog else 1, len(exog))
        c3, c4, c5 = st.columns(3)
        max_lag = c3.slider(
            "Max lag for selection",
            1,
            dyn_max_lag,
            min(4, dyn_max_lag),
            key=f"{key_prefix}_max_lag",
        )

        data_json = dataframe_to_json(modeling_df)
        if valid_selection and test_n >= MIN_TEST_OBS:
            lag_table = select_lags_cached(
                data_json,
                str(split_ts.date()),
                model_type,
                tuple(endog),
                tuple(exog),
                max_lag,
            )
            lag_table = lag_table.dropna(subset=["AIC", "BIC", "HQIC"], how="all")
            selected_lag = 1
            if not lag_table.empty and lag_table["AIC"].notna().any():
                selected_lag = int(lag_table.loc[lag_table["AIC"].idxmin(), "lag"])
                selected_lag = max(1, selected_lag)
            lag_order = c4.slider(
                "Manual lag order",
                1,
                max_lag,
                min(selected_lag, max_lag),
                key=f"{key_prefix}_lag_order",
            )
            target = c5.selectbox(
                "Forecast target",
                endog,
                index=endog.index("INF") if "INF" in endog else 0,
                key=f"{key_prefix}_target",
            )
            per_eq, total_params = parameter_count(len(endog), lag_order, len(exog))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Train obs.", f"{train_n:,}")
            m2.metric("Test obs.", f"{test_n:,}")
            m3.metric("Parameters/equation", f"{per_eq:,}")
            m4.metric("Total parameters", f"{total_params:,}")
            if train_n / max(total_params, 1) < 5:
                st.warning("High overparameterization risk for this specification.")
            try:
                payload = fit_model_cached(
                    data_json,
                    str(split_ts.date()),
                    model_type,
                    tuple(endog),
                    tuple(exog),
                    lag_order,
                    target,
                )
                run = unpack_run(payload, model_type, lag_order, target, endog, exog)
                if run.warning:
                    st.warning(run.warning)
            except Exception as exc:
                st.error(f"Interactive model failed: {exc}")
        else:
            st.warning("Invalid model selection or too few test observations.")

    return run, lag_table


data = load_project_data()
raw = data["raw"]
model_df = data["model"]
dummies = data["dummies"]
modeling_df = modeling_frame(model_df, dummies)
PLOT_TEMPLATE = None

pages = [
    "Overview",
    "Stationarity and Data Preparation",
    "Model Architecture and Direct Results",
    "Forecast Comparison",
    "Residual Diagnostics",
    "Significance Analysis and Granger Causality",
    "IRF and FEVD",
    "Robustness",
    "Code quality",
]
page = st.sidebar.radio("Dashboard page", pages, index=0)
st.sidebar.caption("Model controls are on the main Model Architecture and Robustness pages.")


st.title("VAR / VARX Macroeconomic Forecasting Dashboard")
st.caption("Inflation and macroeconomic policy analysis with VAR, VARX, and ML forecast benchmarks.")

metric_cols = st.columns(4)
metric_cols[0].metric("Raw observations", f"{len(raw):,}")
metric_cols[1].metric("Model observations", f"{len(model_df):,}")
metric_cols[2].metric("Official VAR lag", "4")
metric_cols[3].metric("Best RMSE model", data["ranking"].iloc[0]["model"])


if page == "Overview":
    st.subheader("Project Overview")
    st.markdown(
        """
        <div class="note">
        This project studies U.S. inflation and macroeconomic policy dynamics using monthly FRED data.
        The central question is whether a macroeconomic system can both forecast inflation and explain how
        policy, labor-market, production, money, and sentiment shocks propagate over time.
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            **Why this matters**

            Inflation forecasting matters because it affects monetary policy, wage bargaining, investment,
            interest-rate expectations, and real purchasing power. A useful project should therefore do more
            than minimize forecast error: it should also explain channels of adjustment.
            """
        )
    with c2:
        st.markdown(
            """
            **Why VAR and VARX**

            VAR models joint macroeconomic dynamics among endogenous variables. VARX models condition forecasts
            on externally supplied policy, sentiment, or crisis paths. Machine-learning models are kept as
            predictive benchmarks, not as substitutes for policy interpretation.
            """
        )
    st.subheader("Dataset and Final Goals")
    if not data["variable_dictionary"].empty:
        st.dataframe(data["variable_dictionary"], width="stretch", hide_index=True)
    st.markdown(
        """
        Final goals:

        - model macroeconomic dynamics;
        - forecast inflation and other macro variables;
        - analyze shocks through IRF and FEVD;
        - compare econometric forecasts with ML benchmarks;
        - report negative diagnostic results transparently instead of hiding model limitations.
        """
    )
    st.subheader("Official Baseline Model")
    st.markdown(
        """
        The official report keeps one fixed baseline specification for reproducibility. The dashboard's
        interactive controls are for sensitivity analysis, not for replacing the reported model.
        """
    )
    st.dataframe(data["architecture"], width="stretch", hide_index=True)
    if not data["comparison"].empty:
        st.write("Official VAR/VARX diagnostic comparison")
        st.dataframe(
            test_table(
                data["comparison"],
                "stable system, reasonable parameter count, and lower forecast error are preferred",
            ),
            width="stretch",
            hide_index=True,
        )
    if not data["optimized_specs"].empty:
        st.subheader("Optimized Model Selection")
        st.markdown(
            """
            These specifications come from the controlled search in `src/model_optimization.py`.
            They are selected by a balanced rule using stability, residual diagnostics, forecast performance,
            parameter count, and economic interpretability.
            """
        )
        optimized_cols = [
            "model_type",
            "candidate_name",
            "lag_order",
            "dummy_specification",
            "endogenous_variables",
            "exogenous_variables",
            "selection_score",
            "stable",
            "portmanteau_whiteness_p_value",
            "inflation_RMSE",
            "mean_relative_RMSE_vs_naive",
        ]
        st.dataframe(
            round_numeric(data["optimized_specs"][[c for c in optimized_cols if c in data["optimized_specs"].columns]]),
            width="stretch",
            hide_index=True,
        )
        if not data["optimized_ranking"].empty:
            st.write("Top optimized candidates")
            ranking_cols = [
                "model_type",
                "candidate_name",
                "dummy_specification",
                "lag_order",
                "selection_score",
                "stable",
                "acf_exceedance_share",
                "inflation_RMSE",
                "obs_per_parameter_per_equation",
            ]
            st.dataframe(
                round_numeric(
                    data["optimized_ranking"]
                    .sort_values("selection_score", ascending=False)
                    [[c for c in ranking_cols if c in data["optimized_ranking"].columns]]
                    .head(12)
                ),
                width="stretch",
                hide_index=True,
            )

elif page == "Stationarity and Data Preparation":
    st.subheader("A. Original Raw Data Checks")
    raw_defaults = [v for v in ["CPI", "UNRATE", "FEDFUNDS", "INDPRO", "M2", "UMCSENT"] if v in raw.columns]
    raw_vars = st.multiselect("Raw variables", list(raw.columns), default=raw_defaults)
    if raw_vars:
        st.plotly_chart(plot_time_series(raw, raw_vars, "Original FRED Series"), width="stretch", key="raw_series")
    stationarity = stationarity_payload(dataframe_to_json(raw), dataframe_to_json(model_df))
    raw_kpss = pd.read_json(StringIO(stationarity["raw_kpss"]), orient="split")
    model_kpss = pd.read_json(StringIO(stationarity["model_kpss"]), orient="split")
    integration = pd.read_json(StringIO(stationarity["integration"]), orient="split")
    st.write("Raw ADF tests")
    st.dataframe(test_table(data["raw_adf"], "ADF p-value < 0.05 supports stationarity"), width="stretch", hide_index=True)
    st.write("Raw KPSS tests")
    st.dataframe(round_numeric(raw_kpss), width="stretch", hide_index=True)

    st.subheader("B. Integration Order Checks")
    st.markdown(
        """
        Decision rule: ADF rejects a unit root when `p < 0.05`; KPSS supports stationarity when `p > 0.05`.
        Variables are classified as `I(0)` if they pass these rules in levels and `I(1)` if the first difference passes.
        Mixed results are reported as potentially problematic rather than forced into a clean category.
        """
    )
    st.dataframe(round_numeric(integration), width="stretch", hide_index=True)

    st.subheader("C. Cointegration Checks")
    if not data["cointegration"].empty:
        st.dataframe(
            test_table(data["cointegration"], "residual ADF p-value < 0.05 supports pairwise cointegration"),
            width="stretch",
            hide_index=True,
        )
    st.markdown(
        """
        The project uses transformed stationary variables for VAR/VARX because broad cointegration support is not
        established among the relevant non-stationary level variables. Without a defensible cointegrating system,
        a VECM would impose long-run restrictions that are not supported by these tests.
        """
    )

    st.subheader("D. Final Transformed Variables")
    transformation_table = pd.DataFrame(
        [
            ["INF", "log(CPI).diff() * 100", "monthly inflation"],
            ["FEDFUNDS", "level", "monetary-policy rate"],
            ["UNRATE", "level", "labor-market slack"],
            ["INDPRO_GROWTH", "log(INDPRO).diff() * 100", "real-activity growth"],
            ["M2_GROWTH", "log(M2).diff() * 100", "money-growth channel"],
            ["SENTIMENT_CHANGE", "UMCSENT.diff()", "expectations/sentiment change"],
        ],
        columns=["model_variable", "construction", "economic_role"],
    )
    st.dataframe(transformation_table, width="stretch", hide_index=True)
    transformed_vars = st.multiselect("Variables in summary plot", list(model_df.columns), default=list(model_df.columns))
    if transformed_vars:
        st.plotly_chart(plot_time_series(model_df, transformed_vars, "Final Modeling Variables"), width="stretch", key="model_series")
        st.plotly_chart(plot_heatmap(model_df[transformed_vars].corr(), "Transformed Variable Correlation", zmin=-1, zmax=1), width="stretch", key="model_corr")

    selected_transformed = st.selectbox(
        "Selected transformed variable for stationarity diagnostics",
        list(model_df.columns),
        index=list(model_df.columns).index("INF") if "INF" in model_df.columns else 0,
    )
    st.plotly_chart(
        plot_time_series(model_df, [selected_transformed], f"{selected_transformed} Time Series"),
        width="stretch",
        key="selected_model_series",
    )

    selected_adf = data["transformed_adf"].loc[data["transformed_adf"]["variable"] == selected_transformed].copy()
    selected_kpss = model_kpss.loc[model_kpss["variable"] == selected_transformed].copy()
    c1, c2 = st.columns(2)
    with c1:
        st.write("ADF result")
        st.dataframe(test_table(selected_adf, "ADF p-value < 0.05 supports stationarity"), width="stretch", hide_index=True)
    with c2:
        st.write("KPSS result")
        st.dataframe(test_table(selected_kpss, "KPSS p-value > 0.05 supports stationarity"), width="stretch", hide_index=True)

    adf_stationary = bool(selected_adf["p_value"].iloc[0] < 0.05) if not selected_adf.empty else False
    kpss_stationary = bool(selected_kpss["p_value"].iloc[0] > 0.05) if not selected_kpss.empty and pd.notna(selected_kpss["p_value"].iloc[0]) else False
    if adf_stationary and kpss_stationary:
        st.success(f"{selected_transformed} passes both ADF and KPSS stationarity rules and is suitable for VAR/VARX modeling.")
    elif adf_stationary:
        st.warning(
            f"{selected_transformed} passes the ADF rule but does not clearly pass KPSS. Treat it as usable with caution and check residual diagnostics."
        )
    else:
        st.warning(
            f"{selected_transformed} does not clearly satisfy the stationarity rules. VAR/VARX results for this variable should be interpreted cautiously."
        )

    acf_df, pacf_df = series_acf_pacf_values(model_df[selected_transformed], max_lag=36)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_correlogram(acf_df, f"ACF: {selected_transformed}"), width="stretch", key="selected_acf")
    with c4:
        st.plotly_chart(plot_correlogram(pacf_df, f"PACF: {selected_transformed}"), width="stretch", key="selected_pacf")
    st.caption("ACF/PACF bars outside the dashed confidence bands indicate remaining serial dependence at that lag.")

    with st.expander("All transformed stationarity tests", expanded=False):
        st.write("Transformed ADF tests")
        st.dataframe(test_table(data["transformed_adf"], "ADF p-value < 0.05 supports stationarity"), width="stretch", hide_index=True)
        st.write("Transformed KPSS tests")
        st.dataframe(test_table(model_kpss, "KPSS p-value > 0.05 supports stationarity"), width="stretch", hide_index=True)

elif page == "Model Architecture and Direct Results":
    interactive_run, interactive_lag_table = interactive_model_control_form(
        "architecture_interactive",
        "Interactive Model Definition",
    )
    if not interactive_lag_table.empty:
        st.subheader("Lag Selection Results")
        st.write("Best lag by criterion")
        st.dataframe(round_numeric(lag_selection_summary(interactive_lag_table)), width="stretch", hide_index=True)
        st.write("Full lag-selection table")
        st.plotly_chart(plot_lag_table(interactive_lag_table), width="stretch", key="interactive_arch_lag_plot")
        st.dataframe(
            test_table(
                interactive_lag_table,
                "Lower AIC/BIC/HQIC is preferred; final lag must preserve degrees of freedom",
            ),
            width="stretch",
            hide_index=True,
        )
    if interactive_run is not None:
        st.subheader("Dynamic Model Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model type", interactive_run.model_type)
        c2.metric("Lag order", str(interactive_run.lag_order))
        c3.metric("Train obs.", f"{len(interactive_run.train):,}")
        c4.metric("Test obs.", f"{len(interactive_run.test):,}")
        spec_table = pd.DataFrame(
            [
                {
                    "model_type": interactive_run.model_type,
                    "endogenous_variables": ", ".join(interactive_run.endog),
                    "exogenous_variables": ", ".join(interactive_run.exog) if interactive_run.exog else "none",
                    "train_start": interactive_run.train.index.min().strftime("%Y-%m"),
                    "train_end": interactive_run.train.index.max().strftime("%Y-%m"),
                    "test_start": interactive_run.test.index.min().strftime("%Y-%m"),
                    "test_end": interactive_run.test.index.max().strftime("%Y-%m"),
                    "target": interactive_run.target,
                }
            ]
        )
        st.dataframe(spec_table, width="stretch", hide_index=True)
        st.write("Model fit information")
        st.dataframe(round_numeric(interactive_run.fit_info), width="stretch", hide_index=True)
        st.write("Equation-level fit metrics")
        st.dataframe(round_numeric(interactive_run.fit_metrics), width="stretch", hide_index=True)
        st.write("Forecast metrics")
        st.dataframe(round_numeric(interactive_run.metrics), width="stretch", hide_index=True)
        st.write("Stability roots")
        st.dataframe(
            test_table(interactive_run.roots, "inverse companion-root modulus < 1 indicates stability in this display"),
            width="stretch",
            hide_index=True,
        )
        if interactive_run.stable is False:
            st.warning("The selected dynamic specification is unstable. Interpret forecasts and responses with caution.")
        st.write("Coefficient and parameter significance table")
        st.dataframe(round_numeric(interactive_run.parameter_table), width="stretch", hide_index=True)
        st.plotly_chart(plot_forecast_run(interactive_run), width="stretch", key="interactive_arch_forecast")

elif page == "Forecast Comparison":
    st.subheader("VAR and VARX Forecasts for Modeled Variables")
    forecast_payload = official_forecasts_payload(dataframe_to_json(model_df), dataframe_to_json(dummies))
    varvarx_forecasts = dataframe_from_json(forecast_payload["forecasts"]).reset_index()
    varvarx_forecasts = varvarx_forecasts.rename(columns={"index": "date"})
    varvarx_metrics = pd.read_json(StringIO(forecast_payload["metrics"]), orient="split")
    variables = sorted(varvarx_forecasts["variable"].unique())
    forecast_variable = st.selectbox("Forecast variable", variables, index=variables.index("INF") if "INF" in variables else 0)
    st.plotly_chart(plot_var_varx_forecast(varvarx_forecasts, forecast_variable), width="stretch", key="var_varx_variable_forecast")
    st.dataframe(
        round_numeric(varvarx_metrics.loc[varvarx_metrics["variable"] == forecast_variable].sort_values("RMSE")),
        width="stretch",
        hide_index=True,
    )
    st.write("All VAR/VARX variable forecast metrics")
    st.dataframe(round_numeric(varvarx_metrics), width="stretch", hide_index=True)

    st.subheader("Inflation Forecast Comparison Across All Models")
    forecasts = combined_inflation_forecasts(data["econ_forecasts"], data["ml_forecasts"])
    if not forecasts.empty:
        available_models = [c for c in forecasts.columns if c not in {"Actual INF", "actual_INF"}]
        selected_models = st.multiselect("Inflation models shown", available_models, default=available_models)
        st.plotly_chart(plot_forecast_lines(forecasts, selected_models, "Actual Inflation and All Model Forecasts"), width="stretch", key="all_model_forecasts")
    ranking = data["ranking"].rename(columns={"rmse": "RMSE", "mae": "MAE"})
    st.dataframe(test_table(ranking, "Lower RMSE/MAE is better; higher directional accuracy is better"), width="stretch", hide_index=True)
    if not data["multihorizon"].empty:
        horizon = st.selectbox("Horizon-specific inflation metrics", sorted(data["multihorizon"]["horizon"].unique()))
        horizon_table = data["multihorizon"].loc[data["multihorizon"]["horizon"] == horizon].sort_values("rmse")
        st.dataframe(test_table(horizon_table, "Lower RMSE/MAE and relative RMSE < 1 are better"), width="stretch", hide_index=True)
        fig = px.line(data["multihorizon"], x="horizon", y="rmse", color="model", markers=True, title="Inflation RMSE by Horizon")
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch", key="horizon_rmse")

elif page == "Residual Diagnostics":
    st.subheader("Residual Diagnostics")
    st.markdown(
        """
        Autocorrelation diagnostics ignore lag 0 because a series is always perfectly correlated with itself at lag 0.
        Cross-correlation between different residual equations includes lag 0 because contemporaneous residual dependence is meaningful.
        """
    )
    model_label = st.radio("Model", ["VAR", "VARX"], horizontal=True)
    baseline = baseline_residual_run(model_label)
    residuals = dataframe_from_json(baseline["residuals"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Stable", "Yes" if baseline["stable"] else "No")
    c2.metric("Portmanteau p-value", f"{baseline['whiteness_p']:.4g}" if baseline["whiteness_p"] is not None else "n/a")
    c3.metric("Normality p-value", f"{baseline['normality_p']:.4g}" if baseline["normality_p"] is not None else "n/a")
    if baseline["whiteness_p"] is not None and baseline["whiteness_p"] < 0.05:
        st.warning("Whiteness is rejected. Residual dependence remains; forecast and IRF interpretation should be cautious.")
    st.plotly_chart(plot_time_series(residuals, list(residuals.columns), f"{model_label} Residual Time Series"), width="stretch", key="resid_ts")
    equation = st.selectbox("Residual equation", list(residuals.columns))
    st.plotly_chart(plot_acf(acf_values(residuals[equation]), equation), width="stretch", key="resid_acf")
    st.dataframe(round_numeric(residual_test_table(residuals)), width="stretch", hide_index=True)

    st.subheader("Lagged Residual Cross-Correlation")
    ccf_summary = residual_cross_correlation_summary(residuals, max_lag=12)
    if not ccf_summary.empty:
        ccf_heatmap = signed_ccf_heatmap(ccf_summary)
        st.plotly_chart(plot_heatmap(ccf_heatmap, "Signed CCF at Maximum Absolute Lag", zmin=-1, zmax=1), width="stretch", key="resid_ccf_heatmap")
        st.dataframe(round_numeric(ccf_summary.sort_values("max_abs_ccf", ascending=False)), width="stretch", hide_index=True)
        pair_source = st.selectbox("CCF source residual", list(residuals.columns), key=f"ccf_source_{model_label}")
        pair_target = st.selectbox(
            "CCF target residual",
            list(residuals.columns),
            index=min(1, len(residuals.columns) - 1),
            key=f"ccf_target_{model_label}",
        )
        ccf_values = residual_cross_correlation_values(residuals, pair_source, pair_target, max_lag=12)
        st.plotly_chart(
            plot_cross_correlation_lags(ccf_values, f"Lagged CCF: {pair_source} and {pair_target}"),
            width="stretch",
            key="resid_ccf_lags",
        )

    st.subheader("Normality and Heteroskedasticity")
    st.plotly_chart(plot_residual_distribution(residuals, equation), width="stretch", key="resid_dist")
    st.dataframe(round_numeric(residual_normality_table(residuals)), width="stretch", hide_index=True)
    norm_table = data["var_norm"] if model_label == "VAR" else data["varx_norm"]
    het_table = data["var_het"] if model_label == "VAR" else data["varx_het"]
    st.dataframe(test_table(norm_table, "Jarque-Bera/system normality p-value > 0.05 is preferred"), width="stretch", hide_index=True)
    st.dataframe(test_table(het_table, "ARCH/Breusch-Pagan/White p-values > 0.05 are preferred for homoskedastic residuals"), width="stretch", hide_index=True)

elif page == "Significance Analysis and Granger Causality":
    st.subheader("Part A: Parameter Significance Analysis")
    sig_model = st.radio("Significance model", ["VAR", "VARX"], horizontal=True)
    table = data["var_sig"] if sig_model == "VAR" else data["varx_sig"]
    robust = data["var_robust"] if sig_model == "VAR" else data["varx_robust"]
    st.markdown(
        """
        Parameter p-values show whether individual lag coefficients are statistically different from zero.
        In VAR and VARX systems, individual coefficients are useful supporting evidence, but they are often
        hard to interpret directly because each equation contains many correlated lags.
        """
    )
    st.dataframe(test_table(table, "p-value < 0.05 indicates statistical significance; interpret individual coefficients cautiously"), width="stretch", hide_index=True)
    st.write("Robust p-value sensitivity")
    if not robust.empty:
        changed = robust.loc[robust["significance_changed_hc3"] | robust["significance_changed_hac"]]
        st.dataframe(test_table(changed, "stable significance under classical, HC3, and HAC p-values is stronger evidence"), width="stretch", hide_index=True)

    st.subheader("Part B: Granger Causality")
    granger = data["granger"].copy()
    granger = add_rule_column(granger, "p-value < 0.05 indicates Granger/predictive causality, not structural causality")
    if "significant_at_5pct" in granger:
        granger["significant"] = np.where(granger["significant_at_5pct"], "yes", "no")
    st.plotly_chart(plot_granger_heatmap(data["granger"]), width="stretch", key="granger_heatmap")
    st.dataframe(round_numeric(granger.sort_values("p_value")), width="stretch", hide_index=True)
    st.markdown(
        """
        Granger results help identify predictive channels and can support the economic motivation for the
        Cholesky ordering. They do not prove structural causality.
        """
    )

elif page == "IRF and FEVD":
    st.subheader("IRF and FEVD")
    model_choice = st.radio("Response model", ["VAR", "VARX"], horizontal=True)
    horizon = st.slider("Horizon", 1, 36, 24)
    if model_choice == "VAR":
        st.markdown(
            """
            <div class="warn">
            VAR IRFs use recursive Cholesky identification. Ordering affects short-run responses, so these are conditional structural interpretations.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(data["ordering"], width="stretch", hide_index=True)
        endog_order = st.text_input("Cholesky order", value=", ".join(VAR_COLUMNS))
        order = [x.strip() for x in endog_order.split(",") if x.strip()]
        if sorted(order) != sorted(VAR_COLUMNS):
            st.error("Ordering must contain the official VAR variables exactly once.")
        else:
            fitted = VAR(model_df[order], exog=dummies[["D_2008", "D_COVID"]]).fit(4, trend="c")
            irf_df = var_irf_paths(fitted, horizon)
            shock = st.selectbox("Shock variable", order, index=order.index("FEDFUNDS"))
            st.plotly_chart(
                plot_response_grid(irf_df, shock, f"VAR IRFs: All Responses to {shock} Shock"),
                width="stretch",
                key="var_irf_grid",
            )
            st.dataframe(
                test_table(
                    irf_df.loc[irf_df["shock"] == shock],
                    "interpret responses as Cholesky-identified conditional dynamics; ordering affects short-run results",
                ),
                width="stretch",
                hide_index=True,
            )
            fevd_df = var_fevd_paths(fitted, horizon)
            fevd_response = st.selectbox("FEVD response", order, index=order.index("INF"))
            fig = px.area(fevd_df.query("response == @fevd_response"), x="horizon", y="variance_share", color="shock", title=f"FEVD for {fevd_response}")
            st.plotly_chart(fig, width="stretch", key="var_fevd")
    else:
        st.markdown(
            """
            <div class="warn">
            VARX responses are conditional scenario responses, not standard structural IRFs. Exogenous paths are assumed and shocked externally.
            </div>
            """,
            unsafe_allow_html=True,
        )
        exog_data = pd.concat([model_df[["FEDFUNDS", "SENTIMENT_CHANGE"]], dummies], axis=1)
        fitted = VAR(model_df[VARX_ENDOG], exog=exog_data).fit(4, trend="c")
        shock = st.selectbox("Exogenous scenario shock", list(exog_data.columns), index=0)
        response_df = varx_exogenous_scenario_response(fitted, model_df[VARX_ENDOG], list(exog_data.columns), shock, horizon)
        st.markdown(
            """
            The shock is applied to an exogenous variable for one month. The plotted values are the difference between
            a shocked exogenous path and a zero-deviation baseline path, propagated through the VARX equations.
            """
        )
        if response_df.empty:
            st.warning("No VARX scenario response is available for this specification.")
        else:
            st.plotly_chart(
                plot_response_grid(response_df, shock, f"VARX Conditional Scenario Responses to {shock} Shock"),
                width="stretch",
                key="varx_response_grid",
            )
            st.dataframe(
                test_table(response_df, "VARX scenario responses depend on assumed future exogenous paths, not structural identification"),
                width="stretch",
                hide_index=True,
            )

elif page == "Robustness":
    st.subheader("Robustness and Sensitivity")
    st.write("Alternative lag choices")
    if not data["lag_robustness"].empty:
        st.dataframe(test_table(data["lag_robustness"], "stable system, fewer ACF exceedances, lower IC values, and whiteness p-value > 0.05 are preferred"), width="stretch", hide_index=True)
    st.write("Alternative train/test split")
    robustness_run, robustness_lag_table = interactive_model_control_form(
        "robustness_interactive",
        "Interactive Train/Test Sensitivity Control Form",
    )
    if not robustness_lag_table.empty:
        st.write("Sensitivity lag selection")
        st.dataframe(
            test_table(
                robustness_lag_table,
                "Lower AIC/BIC/HQIC is preferred; final lag must preserve degrees of freedom",
            ),
            width="stretch",
            hide_index=True,
        )
    if robustness_run is not None:
        st.dataframe(round_numeric(robustness_run.metrics), width="stretch", hide_index=True)
        st.plotly_chart(plot_forecast_run(robustness_run), width="stretch", key="robust_interactive_forecast")
    if not data["crisis"].empty:
        st.write("Crisis dummy robustness")
        st.dataframe(test_table(data["crisis"], "lower AIC/BIC/RMSE is better; whiteness p-value > 0.05 is preferred"), width="stretch", hide_index=True)
    if not data["multihorizon"].empty:
        st.write("Forecast robustness across horizons")
        st.dataframe(test_table(data["multihorizon"], "lower RMSE/MAE and relative RMSE < 1 are better"), width="stretch", hide_index=True)
    if not data["dm"].empty:
        st.write("Diebold-Mariano tests")
        st.dataframe(test_table(data["dm"], "p-value < 0.05 rejects equal forecast accuracy against benchmark"), width="stretch", hide_index=True)
    if not data["expanding"].empty:
        relationship = st.selectbox("Expanding-window relationship", sorted(data["expanding"]["relationship"].unique()))
        subset = data["expanding"].loc[data["expanding"]["relationship"] == relationship].copy()
        subset["window_end"] = pd.to_datetime(subset["window_end"])
        fig = px.line(subset, x="window_end", y="coefficient_L1", markers=True, title=f"Expanding-Window Coefficient: {relationship}")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, width="stretch", key="expanding_coef")
        st.dataframe(test_table(subset, "stable signs/significance across windows indicate stronger regime robustness"), width="stretch", hide_index=True)
    if not data["regime"].empty:
        st.write("Pre-COVID / full-sample and regime split comparison")
        st.dataframe(test_table(data["regime"], "stable signs/significance across regimes indicate stronger robustness"), width="stretch", hide_index=True)
    if not data["irf_robustness"].empty:
        st.write("Alternative Cholesky ordering")
        response = st.selectbox("IRF ordering response", sorted(data["irf_robustness"]["response"].unique()))
        subset = data["irf_robustness"].loc[data["irf_robustness"]["response"] == response]
        fig = px.bar(subset, x="ordering_name", y="response_h6", title=f"{response} response to FEDFUNDS shock at horizon 6")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, width="stretch", key="irf_robust_order")
        st.dataframe(test_table(subset, "similar response signs across orderings indicate stronger IRF robustness"), width="stretch", hide_index=True)

elif page == "Code quality":
    st.subheader("Code Quality and Project Structure")
    modules = pd.DataFrame(
        [
            ["dashboard_app.py", "Streamlit UI and page routing only"],
            ["src/data.py", "cached data loading and project-table access"],
            ["src/models_var.py", "VAR fitting, forecast objects, IRF/FEVD helpers, equation fit metrics"],
            ["src/models_varx.py", "VARX fitting and conditional scenario-response helpers"],
            ["src/diagnostics.py", "stationarity, residual tests, ACF and lagged CCF diagnostics"],
            ["src/forecasting.py", "forecast metrics, ML helpers, official VAR/VARX forecast comparison"],
            ["src/model_optimization.py", "controlled VAR/VARX specification search and context-file generation"],
            ["src/visualization.py", "Plotly figure builders"],
            ["src/advanced_macro_var_analysis.py", "reproducible academic pipeline and output generation"],
        ],
        columns=["file", "responsibility"],
    )
    st.dataframe(modules, width="stretch", hide_index=True)
    st.markdown(
        """
        Performance choices:

        - project data loading is cached;
        - lag selection and interactive model fitting are cached by selected variables, split date, model type, and lag;
        - official VAR/VARX forecast comparison is cached;
        - real-time IRF and VARX scenario-response displays avoid expensive bootstrap recomputation;
        - residual CCF summaries are computed from cached residuals and include lag 0 only for cross-equation correlations.
        - optimized model-selection outputs are precomputed by `src/model_optimization.py` and loaded as dashboard tables.

        Validation rules in the main-page control forms prevent overlapping endogenous/exogenous selections, too few endogenous variables,
        too few test observations, and obviously overparameterized choices.
        """
    )
    st.code(
        ".venv/bin/python -m py_compile dashboard_app.py src/*.py\n"
        ".venv/bin/python -m src.model_optimization\n"
        ".venv/bin/jupyter nbconvert --to notebook --execute macro_time_series_project.ipynb --output macro_time_series_project.ipynb\n"
        ".venv/bin/streamlit run dashboard_app.py --server.address 127.0.0.1 --server.port 8501",
        language="bash",
    )
