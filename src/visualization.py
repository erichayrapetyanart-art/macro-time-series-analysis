from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def apply_theme(fig: go.Figure, template: str | None = None) -> go.Figure:
    if template:
        fig.update_layout(template=template)
    fig.update_layout(margin=dict(l=24, r=24, t=60, b=28))
    return fig


def plot_time_series(df: pd.DataFrame, columns: list[str], title: str, template: str | None = None) -> go.Figure:
    fig = go.Figure()
    for col in columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    fig.update_layout(title=title, height=480)
    return apply_theme(fig, template)


def plot_forecast_lines(
    forecasts: pd.DataFrame,
    selected_models: list[str],
    title: str,
    template: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    actual_col = "Actual INF" if "Actual INF" in forecasts else "actual_INF"
    if actual_col in forecasts:
        fig.add_trace(
            go.Scatter(
                x=forecasts.index,
                y=forecasts[actual_col],
                mode="lines+markers",
                name="Actual inflation",
                line=dict(width=3),
            )
        )
    for model in selected_models:
        if model in forecasts.columns and model != actual_col:
            fig.add_trace(
                go.Scatter(
                    x=forecasts.index,
                    y=forecasts[model],
                    mode="lines+markers",
                    name=model,
                )
            )
    fig.update_layout(title=title, height=540)
    return apply_theme(fig, template)


def plot_forecast_run(run, template: str | None = None) -> go.Figure:
    target = run.target
    fig = go.Figure()
    history = run.train[target].tail(60)
    fig.add_trace(go.Scatter(x=history.index, y=history, mode="lines", name="Training history"))
    fig.add_trace(
        go.Scatter(
            x=run.test.index,
            y=run.test[target],
            mode="lines+markers",
            name="Actual test",
            line=dict(width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=run.forecasts.index,
            y=run.forecasts[target],
            mode="lines+markers",
            name=f"{run.model_type} forecast",
        )
    )
    lower = f"{target}_lower_95"
    upper = f"{target}_upper_95"
    if lower in run.forecasts and upper in run.forecasts:
        fig.add_trace(go.Scatter(x=run.forecasts.index, y=run.forecasts[upper], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=run.forecasts.index,
                y=run.forecasts[lower],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(80, 140, 190, 0.20)",
                line=dict(width=0),
                name="95% forecast interval",
            )
        )
    fig.update_layout(title=f"Out-of-Sample Forecast for {target}", height=520)
    return apply_theme(fig, template)


def plot_lag_table(lag_table: pd.DataFrame, template: str | None = None) -> go.Figure:
    fig = go.Figure()
    for col in ["AIC", "BIC", "HQIC"]:
        if col in lag_table:
            fig.add_trace(go.Scatter(x=lag_table["lag"], y=lag_table[col], mode="lines+markers", name=col))
    fig.update_layout(title="Lag Selection Criteria", height=420)
    return apply_theme(fig, template)


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    template: str | None = None,
    colorscale: str = "RdBu",
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            zmid=0 if zmin is not None and zmin < 0 else None,
            text=np.round(matrix.values, 3),
            texttemplate="%{text}",
            colorbar=dict(thickness=14),
        )
    )
    fig.update_layout(title=title, height=520)
    return apply_theme(fig, template)


def plot_acf(acf_df: pd.DataFrame, equation: str, template: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=acf_df["lag"],
            y=acf_df["acf"],
            marker_color=np.where(acf_df["outside_bound"], "#d97706", "#2563eb"),
            name="ACF",
        )
    )
    fig.add_trace(go.Scatter(x=acf_df["lag"], y=acf_df["upper"], mode="lines", line=dict(color="#ef4444", dash="dash"), name="95% bound"))
    fig.add_trace(go.Scatter(x=acf_df["lag"], y=acf_df["lower"], mode="lines", line=dict(color="#ef4444", dash="dash"), showlegend=False))
    fig.update_layout(title=f"Residual ACF: {equation}", height=420)
    return apply_theme(fig, template)


def plot_correlogram(corr_df: pd.DataFrame, title: str, template: str | None = None) -> go.Figure:
    fig = go.Figure()
    if not corr_df.empty:
        fig.add_trace(
            go.Bar(
                x=corr_df["lag"],
                y=corr_df["value"],
                marker_color=np.where(corr_df["outside_bound"], "#d97706", "#2563eb"),
                name=title,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=corr_df["lag"],
                y=corr_df["upper"],
                mode="lines",
                line=dict(color="#ef4444", dash="dash"),
                name="95% bound",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=corr_df["lag"],
                y=corr_df["lower"],
                mode="lines",
                line=dict(color="#ef4444", dash="dash"),
                showlegend=False,
            )
        )
        fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(title=title, height=380)
    return apply_theme(fig, template)


def plot_residual_distribution(residuals: pd.DataFrame, equation: str, template: str | None = None) -> go.Figure:
    series = residuals[equation].dropna()
    reference_color = "#6b7280"
    sorted_values = np.sort(series.values)
    probs = (np.arange(1, len(sorted_values) + 1) - 0.5) / len(sorted_values)
    theoretical = stats.norm.ppf(probs, loc=series.mean(), scale=series.std(ddof=1))
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Histogram", "Q-Q Plot"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Histogram(x=series, histnorm="probability density", nbinsx=28, name="Residual histogram"), row=1, col=1)
    grid = np.linspace(series.min(), series.max(), 200)
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=stats.norm.pdf(grid, loc=series.mean(), scale=series.std(ddof=1)),
            mode="lines",
            name="Normal density",
            line=dict(color=reference_color),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=theoretical, y=sorted_values, mode="markers", name="Q-Q points"), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=[theoretical.min(), theoretical.max()],
            y=[theoretical.min(), theoretical.max()],
            mode="lines",
            name="45-degree line",
            line=dict(color=reference_color, dash="dash"),
        ),
        row=1,
        col=2,
    )
    fig.update_layout(title=f"Residual Normality Diagnostics: {equation}", height=430)
    return apply_theme(fig, template)


def plot_cross_correlation_lags(ccf_df: pd.DataFrame, title: str, template: str | None = None) -> go.Figure:
    fig = go.Figure()
    if not ccf_df.empty:
        fig.add_trace(
            go.Bar(
                x=ccf_df["lag"],
                y=ccf_df["cross_correlation"],
                marker_color=np.where(ccf_df["cross_correlation"].abs() >= 0.2, "#d97706", "#2563eb"),
                name="Lagged cross-correlation",
            )
        )
        bound = ccf_df["upper"].iloc[0] if "upper" in ccf_df else 1.96 / np.sqrt(max(len(ccf_df), 1))
        fig.add_hline(y=0, line_color="gray", line_width=1)
        fig.add_hline(y=bound, line_dash="dash", line_color="#ef4444")
        fig.add_hline(y=-bound, line_dash="dash", line_color="#ef4444")
    fig.update_layout(title=title, height=420)
    return apply_theme(fig, template)


def plot_var_varx_forecast(
    forecast_long: pd.DataFrame,
    variable: str,
    template: str | None = None,
) -> go.Figure:
    subset = forecast_long.loc[forecast_long["variable"] == variable]
    fig = go.Figure()
    if subset.empty:
        return apply_theme(fig, template)
    actual = subset.drop_duplicates("date")
    fig.add_trace(
        go.Scatter(
            x=actual["date"],
            y=actual["actual"],
            mode="lines+markers",
            name=f"Actual {variable}",
            line=dict(width=3),
        )
    )
    for model_name, group in subset.groupby("model"):
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["forecast"],
                mode="lines+markers",
                name=f"{model_name} forecast",
            )
        )
    fig.update_layout(title=f"Actual vs VAR/VARX Forecasts: {variable}", height=520)
    return apply_theme(fig, template)


def plot_response_grid(
    response_df: pd.DataFrame,
    shock: str,
    title: str,
    template: str | None = None,
) -> go.Figure:
    subset = response_df.loc[response_df["shock"] == shock].copy()
    y_col = "value" if "value" in subset.columns else "orthogonalized_response"
    fig = px.line(
        subset,
        x="horizon",
        y=y_col,
        facet_col="response",
        facet_col_wrap=2,
        markers=True,
        title=title,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_yaxes(matches=None)
    fig.update_layout(height=max(520, 240 * max(1, int(np.ceil(subset["response"].nunique() / 2)))))
    return apply_theme(fig, template)


def plot_granger_heatmap(granger: pd.DataFrame, template: str | None = None) -> go.Figure:
    pivot = granger.pivot(index="source", columns="target", values="p_value")
    fig = px.imshow(
        pivot,
        text_auto=".3f",
        color_continuous_scale="Viridis_r",
        zmin=0,
        zmax=0.1,
        title="Granger Causality P-Values",
    )
    return apply_theme(fig, template)
