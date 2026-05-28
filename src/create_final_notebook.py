from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = BASE_DIR / "macroeconomic_var_varx_final_report.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "pygments_lexer": "ipython3",
    }

    cells: list[nbf.NotebookNode] = []

    cells.append(
        md(
            """
            # Macroeconomic VAR/VARX Time-Series Project

            **Topic.** Inflation and macroeconomic policy analysis using U.S. monthly macroeconomic data.

            **Core models.**

            - **VAR** is the main dynamic macroeconomic system for Granger causality, impulse responses, FEVD, and policy interpretation.
            - **VARX** is the conditional/scenario forecasting model where externally supplied policy and sentiment paths can be imposed.
            - **Machine learning models** are used only as forecasting benchmarks, not structural policy models.

            This notebook is the clean final report notebook. It loads the generated project outputs from `data/` and `outputs/`, so it is reproducible from the saved workflow and consistent with the Streamlit dashboard.
            """
        )
    )

    cells.append(
        code(
            """
            from pathlib import Path
            import warnings

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from IPython.display import Image, Markdown, display

            warnings.filterwarnings("ignore")
            pd.set_option("display.max_columns", 80)
            pd.set_option("display.max_rows", 80)
            pd.set_option("display.width", 160)

            BASE_DIR = Path.cwd()
            DATA_DIR = BASE_DIR / "data"
            TABLE_DIR = BASE_DIR / "outputs" / "tables"
            FIGURE_DIR = BASE_DIR / "outputs" / "figures"
            LABEL_COLUMNS = [
                "variable", "target", "source", "response", "shock", "equation",
                "parameter", "source_variable", "source_residual", "target_residual",
                "endogenous_variables", "exogenous_variables",
            ]
            LABEL_DTYPES = {column: "string" for column in LABEL_COLUMNS}

            def read_table(name: str, index_col=None) -> pd.DataFrame:
                path = TABLE_DIR / name
                if not path.exists():
                    raise FileNotFoundError(path)
                return pd.read_csv(path, index_col=index_col, keep_default_na=False, na_values=[""], dtype=LABEL_DTYPES)

            def read_data(name: str) -> pd.DataFrame:
                return pd.read_csv(DATA_DIR / name, parse_dates=["date"], index_col="date", keep_default_na=False, na_values=[""])

            def round_df(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
                out = df.copy()
                numeric_cols = out.select_dtypes(include=[np.number]).columns
                out[numeric_cols] = out[numeric_cols].round(digits)
                return out

            def show_table(name: str, cols=None, n: int | None = None, title: str | None = None) -> pd.DataFrame:
                df = read_table(name)
                if cols is not None:
                    df = df[[c for c in cols if c in df.columns]]
                if n is not None:
                    df = df.head(n)
                if title:
                    display(Markdown(f"**{title}**"))
                display(round_df(df))
                return df

            def show_image(name: str, title: str | None = None) -> None:
                path = FIGURE_DIR / name
                if title:
                    display(Markdown(f"**{title}**"))
                if path.exists():
                    display(Image(filename=str(path)))
                else:
                    display(Markdown(f"`{path}` not found."))

            raw = read_data("raw_fred_macro.csv")
            model_data = read_data("academic_model_data.csv")
            dummies = read_data("academic_break_dummies.csv")

            print(f"Raw sample: {raw.index.min().date()} to {raw.index.max().date()}, shape={raw.shape}")
            print(f"Model sample: {model_data.index.min().date()} to {model_data.index.max().date()}, shape={model_data.shape}")
            """
        )
    )

    cells.append(
        md(
            """
            ## 1. Research Motivation and Problem Formulation

            Macroeconomic forecasting matters because inflation, unemployment, production, money growth, and interest rates are central to policy design and private-sector planning. A central policy question is whether a multivariate macroeconomic system can describe inflation dynamics, real-side responses, and monetary-policy transmission in a way that is both forecastable and economically interpretable.

            The final project uses monthly U.S. data and asks:

            - How do inflation, monetary policy, unemployment, industrial production, money growth, and consumer sentiment interact dynamically?
            - Which variables have predictive content for others?
            - How do shocks propagate under an explicit recursive identification assumption?
            - Does econometric interpretability come with competitive forecast performance relative to ML benchmarks?

            The project avoids treating machine learning models as structural policy models. ML models are included for forecast comparison only.
            """
        )
    )

    cells.append(
        md(
            """
            ## 2. Data and Transformations

            Data are monthly U.S. macroeconomic series from FRED. The transformed modeling variables are:

            - `INF`: monthly CPI inflation rate.
            - `FEDFUNDS`: federal funds rate.
            - `UNRATE`: unemployment rate.
            - `INDPRO_GROWTH`: industrial production growth.
            - `M2_GROWTH`: money supply growth.
            - `SENTIMENT_CHANGE`: change in consumer sentiment.
            - `D_2008`, `D_COVID`: crisis-period dummy variables used only in robustness checks.

            The VAR/VARX modeling uses stationary or approximately stationary transformed variables rather than non-stationary macroeconomic levels.
            """
        )
    )
    cells.append(
        code(
            """
            display(Markdown("**Variable dictionary**"))
            try:
                display(round_df(read_table("academic_variable_dictionary.csv")))
            except FileNotFoundError:
                display(pd.DataFrame({"variable": model_data.columns}))

            display(Markdown("**Missing-value check**"))
            show_table("academic_missing_values.csv")

            display(Markdown("**Model-data summary statistics**"))
            show_table("academic_model_summary_statistics.csv")
            """
        )
    )
    cells.append(code('show_image("academic_01_raw_series.png", "Raw macroeconomic series")\nshow_image("academic_02_transformed_series.png", "Final transformed modeling variables")'))

    cells.append(
        md(
            """
            ## 3. Stationarity, Integration, and Cointegration

            Stationarity checks use ADF and KPSS-style logic where available:

            - ADF p-value below 0.05 supports stationarity.
            - KPSS p-value above 0.05 supports stationarity.

            Raw variables such as CPI and M2 are generally non-stationary in levels, while the transformed variables are designed to be usable in VAR/VARX models. Pairwise cointegration checks are reported for relevant level variables. Because robust cointegration support is limited and the project focuses on short-run dynamics and forecasting, the final system is a VAR/VARX in transformed stationary variables rather than a VECM.
            """
        )
    )
    cells.append(
        code(
            """
            show_table("academic_raw_adf_tests.csv", title="ADF tests on raw variables")
            show_table("academic_transformed_adf_tests.csv", title="ADF tests on transformed variables")
            show_table("academic_pairwise_cointegration.csv", title="Pairwise Engle-Granger cointegration checks")
            show_image("academic_07_inflation_acf_pacf.png", "Inflation ACF/PACF")
            """
        )
    )

    cells.append(
        md(
            """
            ## 4. Exploratory Data Analysis

            EDA checks the joint behavior of transformed macro variables before modeling. Correlations and scatter plots are not causal evidence, but they help identify plausible dynamic relationships and potential residual cross-dependence.
            """
        )
    )
    cells.append(
        code(
            """
            show_table("academic_model_correlation_matrix.csv", title="Correlation matrix")
            show_image("academic_03_rolling_means.png", "Rolling mean and regime inspection")
            show_image("academic_04_correlation_matrix.png", "Correlation heatmap")
            show_image("academic_05_scatter_matrix.png", "Scatter matrix")
            show_image("academic_06_distributions.png", "Variable distributions")
            show_image("academic_08_structural_break_inspection.png", "Structural break inspection")
            """
        )
    )

    cells.append(
        md(
            """
            ## 5. Controlled Model Optimization

            The optimized baseline models are selected from a controlled specification search, not by blindly minimizing a single metric.

            **Selected VAR.** `VAR_core_plus_sentiment`, lag 5, no crisis dummies.

            - Endogenous variables: `INF`, `FEDFUNDS`, `UNRATE`, `INDPRO_GROWTH`, `SENTIMENT_CHANGE`.
            - AIC/FPE preferred lag 3, BIC/HQIC preferred lag 2.
            - Lag 5 is retained because broader optimization balanced forecast performance, residual ACF behavior, stability, and policy interpretability.

            **Selected VARX.** `VARX_A_policy_sentiment_exog`, lag 4, no crisis dummies.

            - Endogenous variables: `INF`, `UNRATE`, `INDPRO_GROWTH`, `M2_GROWTH`.
            - Exogenous variables: `FEDFUNDS`, `SENTIMENT_CHANGE`.
            - AIC/FPE support lag 4; BIC prefers lag 1 and HQIC prefers lag 3.
            - Lag 4 is defensible for conditional forecasting and scenario analysis.
            """
        )
    )
    cells.append(
        code(
            """
            final_cols = [
                "model_type", "candidate_name", "dummy_specification", "lag_order",
                "endogenous_variables", "exogenous_variables", "train_end", "test_start",
                "total_parameters", "obs_per_parameter_per_equation", "stable",
                "portmanteau_whiteness_p_value", "acf_exceedance_share",
                "inflation_RMSE", "inflation_MAE", "mean_RMSE", "selection_score"
            ]
            show_table("optimized_final_model_specs.csv", cols=final_cols, title="Final optimized model specifications")

            rank_cols = [
                "model_type", "candidate_name", "dummy_specification", "lag_order",
                "selection_score", "stable", "portmanteau_whiteness_p_value",
                "acf_exceedance_share", "inflation_RMSE", "mean_relative_RMSE_vs_naive",
                "obs_per_parameter_per_equation"
            ]
            ranking = show_table("optimized_candidate_model_ranking.csv", cols=rank_cols, n=12, title="Top candidate models by balanced selection score")

            best_lags = read_table("optimized_lag_selection_best_by_criterion.csv")
            selected_lags = best_lags[
                ((best_lags["model_type"] == "VAR") & (best_lags["candidate_name"] == "VAR_core_plus_sentiment") & (best_lags["dummy_specification"] == "no_dummies"))
                | ((best_lags["model_type"] == "VARX") & (best_lags["candidate_name"] == "VARX_A_policy_sentiment_exog") & (best_lags["dummy_specification"] == "no_dummies"))
            ]
            display(Markdown("**Lag criteria for selected specifications**"))
            display(round_df(selected_lags))

            show_table("optimized_crisis_dummy_search.csv", cols=[
                "model_type", "dummy_specification", "best_candidate", "best_lag",
                "selection_score", "aic", "bic", "portmanteau_whiteness_p_value",
                "acf_exceedance_share", "inflation_RMSE", "stable"
            ], title="Crisis dummy robustness search")
            """
        )
    )

    cells.append(
        md(
            """
            ## 6. Final VAR Results

            The selected unrestricted VAR is the main policy-interpretation model. It models the joint endogenous dynamics of inflation, the federal funds rate, unemployment, industrial production growth, and sentiment changes.

            Individual VAR coefficients are not interpreted mechanically because coefficients represent lagged partial associations inside a dynamic system. The stronger economic interpretation comes from Granger causality, IRF, FEVD, forecast behavior, and diagnostics.
            """
        )
    )
    cells.append(
        code(
            """
            show_table("optimized_final_var_fit_metrics.csv", title="VAR equation-level fit")
            show_table("optimized_final_var_metrics.csv", title="VAR out-of-sample forecast metrics")
            show_table("optimized_final_var_significance_summary.csv", title="VAR classical parameter-significance summary")
            show_table("optimized_final_var_robust_significance_summary.csv", title="VAR robust-significance summary")

            display(Markdown("**Most significant VAR coefficients by classical p-value**"))
            params = read_table("optimized_final_var_parameters.csv")
            display(round_df(params.sort_values("p_value").head(20)))
            """
        )
    )

    cells.append(
        md(
            """
            ### VAR Residual Diagnostics

            Equation-level Ljung-Box tests mostly look acceptable and Durbin-Watson values are near 2. However, system-level Portmanteau whiteness rejects, so the VAR is useful but not perfect. Residual non-normality and ARCH effects are common in macroeconomic crisis periods and weaken classical p-values and confidence intervals.
            """
        )
    )
    cells.append(
        code(
            """
            show_table("optimized_final_var_residual_tests.csv", title="VAR residual Ljung-Box / Durbin-Watson / ARCH")
            show_table("optimized_final_var_residual_acf.csv", title="VAR positive-lag residual ACF summary")
            show_table("optimized_final_var_residual_ccf.csv", n=20, title="VAR residual cross-correlations, including lag 0 for cross-equation residuals")
            show_table("optimized_final_var_normality.csv", title="VAR residual normality")
            show_image("academic_var_residual_plots.png", "VAR residual time-series plots")
            show_image("academic_var_residual_acf.png", "VAR residual ACF plots")
            show_image("academic_var_residual_acf_ccf_matrix.png", "VAR residual CCF summary")
            show_image("academic_var_residual_qq_hist.png", "VAR residual Q-Q and histograms")
            """
        )
    )

    cells.append(
        md(
            """
            ### VAR Granger Causality, IRF, and FEVD

            Granger causality is predictive causality, not structural causal proof. The selected VAR shows:

            - `FEDFUNDS` predicts real-side variables such as `UNRATE` and `INDPRO_GROWTH`.
            - `INF` predicts `FEDFUNDS`, consistent with a policy-reaction function.
            - `UNRATE` and `INDPRO_GROWTH` predict each other.

            IRFs use recursive Cholesky identification. This is an assumption, not a fact. The observed FEDFUNDS shock pattern includes a short-run inflation increase and real-side responses that likely mix monetary tightening with the Federal Reserve's endogenous reaction to strong macroeconomic conditions. This is a price-puzzle / identification warning, not a clean textbook contractionary policy shock.
            """
        )
    )
    cells.append(
        code(
            """
            granger = read_table("optimized_final_var_granger.csv")
            display(Markdown("**Significant VAR Granger relationships at 5%**"))
            display(round_df(granger.loc[granger["significant_at_5pct"] == True]))
            show_image("academic_09_granger_causality_heatmap.png", "Granger causality p-value heatmap")

            irf_key = show_table("optimized_final_var_irf_key_fedfunds.csv", title="Key VAR IRF responses to FEDFUNDS shock")
            fevd_key = show_table("optimized_final_var_fevd_key_inflation.csv", title="Inflation FEVD selected horizons")
            show_table("academic_var_irf_confidence_intervals.csv", n=30, title="VAR IRF confidence intervals, selected rows")
            show_table("academic_cholesky_ordering_robustness.csv", n=30, title="Alternative Cholesky ordering robustness, selected rows")
            show_image("academic_irf_with_confidence_intervals.png", "VAR IRFs with confidence intervals")
            show_image("academic_cholesky_ordering_comparison.png", "Alternative Cholesky ordering comparison")
            show_image("academic_11_fevd.png", "Forecast error variance decomposition")
            """
        )
    )

    cells.append(
        md(
            """
            ## 7. Final VARX Results

            VARX is not the primary structural policy model. It is a conditional/scenario forecasting model. In the selected VARX, `FEDFUNDS` and `SENTIMENT_CHANGE` are externally conditioned paths. Forecasts and scenario responses therefore depend on assumed future exogenous values.
            """
        )
    )
    cells.append(
        code(
            """
            show_table("optimized_final_varx_fit_metrics.csv", title="VARX equation-level fit")
            show_table("optimized_final_varx_metrics.csv", title="VARX out-of-sample forecast metrics")
            show_table("optimized_final_varx_significance_summary.csv", title="VARX classical parameter-significance summary")
            show_table("optimized_final_varx_robust_significance_summary.csv", title="VARX robust-significance summary")

            display(Markdown("**Most significant VARX coefficients by classical p-value**"))
            varx_params = read_table("optimized_final_varx_parameters.csv")
            display(round_df(varx_params.sort_values("p_value").head(20)))
            """
        )
    )
    cells.append(
        code(
            """
            show_table("optimized_final_varx_residual_tests.csv", title="VARX residual Ljung-Box / Durbin-Watson / ARCH")
            show_table("optimized_final_varx_residual_acf.csv", title="VARX positive-lag residual ACF summary")
            show_table("optimized_final_varx_residual_ccf.csv", n=20, title="VARX residual cross-correlations, including lag 0")
            show_table("optimized_final_varx_normality.csv", title="VARX residual normality")
            show_table("optimized_final_varx_scenario_response.csv", n=30, title="VARX FEDFUNDS conditional/scenario response")
            show_image("academic_varx_residual_plots.png", "VARX residual time-series plots")
            show_image("academic_varx_residual_acf.png", "VARX residual ACF plots")
            show_image("academic_varx_residual_acf_ccf_matrix.png", "VARX residual CCF summary")
            show_image("academic_varx_residual_qq_hist.png", "VARX residual Q-Q and histograms")
            show_image("academic_varx_fedfunds_scenario_response.png", "VARX FEDFUNDS conditional scenario response")
            """
        )
    )

    cells.append(
        md(
            """
            ## 8. Forecast Comparison

            The optimized selected-model recursive holdout metrics are different from one-step/direct benchmark metrics. The selected VAR inflation RMSE is about 0.176 under the selected-model recursive holdout design, while the all-benchmark table can show VAR around 0.199 under a different one-step/direct forecast design. These are not contradictions; they are different evaluation protocols.

            For pure one-step prediction, Ridge can be competitive or best. For macroeconomic policy interpretation, VAR/VARX remain stronger because they support Granger causality, IRF, FEVD, and explicit shock/scenario analysis.
            """
        )
    )
    cells.append(
        code(
            """
            show_table("optimized_final_var_metrics.csv", title="Selected VAR recursive holdout metrics")
            show_table("optimized_final_varx_metrics.csv", title="Selected VARX recursive holdout metrics")
            show_table("academic_all_model_forecast_ranking.csv", title="Inflation forecast ranking across econometric, ML, and naive benchmarks")
            show_table("academic_multihorizon_forecast_comparison.csv", title="Multi-horizon inflation forecast comparison")
            show_table("academic_diebold_mariano_tests.csv", title="Diebold-Mariano forecast-comparison tests")
            show_image("academic_12_econometric_forecast_comparison.png", "Econometric inflation forecast comparison")
            show_image("academic_14_ml_forecast_comparison.png", "ML inflation forecast comparison")
            show_image("academic_multihorizon_rmse.png", "Multi-horizon RMSE comparison")
            """
        )
    )

    cells.append(
        md(
            """
            ## 9. Restricted VAR and Restricted VARX Parsimony Checks

            Restricted models are robustness checks built from the official baseline models. They are not interactive dashboard refits and not automatic replacements for the baseline. Restrictions remove whole lag blocks only when economic logic, Granger/block causality, and robust HC3/HAC evidence support parsimony.

            The unrestricted VAR remains the official policy-interpretation model because it preserves complete dynamic channels for IRF and FEVD. The restricted VARX is useful as a parsimonious conditional-forecasting robustness check.
            """
        )
    )
    cells.append(
        code(
            """
            show_table("restricted_var_metrics.csv", title="Restricted VAR summary")
            show_table("restricted_var_restrictions.csv", cols=[
                "equation", "source_variable", "restriction_type", "removed_parameters",
                "granger_p_value", "classical_block_f_p_value", "min_hc3_p_value_in_block",
                "min_hac_p_value_in_block", "economic_override", "imposed", "reason"
            ], title="Restricted VAR block decisions")
            show_table("restricted_var_forecast_comparison.csv", title="Restricted VAR forecast comparison")
            show_table("restricted_var_residual_diagnostics.csv", n=25, title="Restricted VAR residual diagnostics, selected rows")

            show_table("restricted_varx_metrics.csv", title="Restricted VARX summary")
            show_table("restricted_varx_restrictions.csv", cols=[
                "equation", "source_variable", "restriction_type", "removed_parameters",
                "granger_p_value", "classical_block_f_p_value", "min_hc3_p_value_in_block",
                "min_hac_p_value_in_block", "economic_override", "imposed", "reason"
            ], title="Restricted VARX block decisions")
            show_table("restricted_varx_forecast_comparison.csv", title="Restricted VARX forecast comparison")
            show_table("restricted_varx_residual_diagnostics.csv", n=25, title="Restricted VARX residual diagnostics, selected rows")
            """
        )
    )

    cells.append(
        md(
            """
            ## 10. Additional Robustness Checks

            The project reports robustness checks for crisis dummies, regime splits, expanding windows, lag sensitivity, multi-horizon forecasting, and alternative Cholesky orderings. Negative or fragile results are retained because they are important for honest econometric interpretation.
            """
        )
    )
    cells.append(
        code(
            """
            show_table("academic_var_varx_diagnostic_comparison.csv", title="VAR vs VARX diagnostic comparison")
            show_table("academic_model_complexity_overparameterization.csv", title="Model complexity and overparameterization")
            show_table("academic_var_lag_residual_autocorr_robustness.csv", title="Lag robustness for residual autocorrelation")
            show_table("academic_crisis_dummy_robustness.csv", title="Crisis dummy robustness")
            show_table("academic_regime_split_comparison.csv", title="Regime split comparison")
            show_table("academic_expanding_window_robustness.csv", n=30, title="Expanding-window robustness, selected rows")
            """
        )
    )

    cells.append(
        md(
            """
            ## 11. Main Findings

            1. The selected VAR is the main policy-analysis model: `INF`, `FEDFUNDS`, `UNRATE`, `INDPRO_GROWTH`, and `SENTIMENT_CHANGE`, lag 5, no dummies.
            2. The selected VARX is the main conditional/scenario model: endogenous `INF`, `UNRATE`, `INDPRO_GROWTH`, `M2_GROWTH`; exogenous `FEDFUNDS`, `SENTIMENT_CHANGE`; lag 4, no dummies.
            3. VAR beats the no-leak naive benchmark for optimized selected-model inflation forecasting, with inflation RMSE about 0.176 versus naive about 0.188.
            4. VARX is useful for scenario analysis but is not the strongest unrestricted inflation forecaster in the selected baseline.
            5. Ridge may be best for pure one-step prediction, but it does not provide Granger causality, IRF, FEVD, or policy-shock interpretation.
            6. Granger results suggest `FEDFUNDS` has stronger predictive content for real-side variables than direct inflation prediction, while `INF -> FEDFUNDS` is consistent with a policy reaction function.
            7. Inflation FEVD is dominated by inflation's own innovations. FEDFUNDS contributes a smaller but nonzero share, around 3.8% at horizons 12 and 24.
            8. FEDFUNDS shock responses show price-puzzle / identification issues, so the IRFs should not be interpreted as clean exogenous monetary policy shocks.
            9. Restricted VAR and VARX checks show parsimony can preserve or improve some forecast metrics, but restrictions are robustness checks rather than structural truth.
            """
        )
    )

    cells.append(
        md(
            """
            ## 12. Limitations

            - System-level residual whiteness still rejects for the selected baseline models.
            - Residual normality is strongly rejected, consistent with crisis-period fat tails.
            - ARCH effects remain in some equations, motivating robust inference and cautious confidence intervals.
            - Recursive Cholesky identification is ordering-dependent.
            - VARX scenario responses depend on externally supplied exogenous paths.
            - Restricted models are data-driven parsimony checks and should not be interpreted as proof that excluded channels are structurally zero.
            """
        )
    )

    cells.append(
        md(
            """
            ## 13. Reproducibility Commands

            To regenerate the optimized model-selection context and tables:

            ```bash
            .venv/bin/python -m src.model_optimization
            ```

            To regenerate restricted VAR/VARX robustness tables:

            ```bash
            .venv/bin/python -m src.restricted_models
            ```

            To rebuild this final notebook:

            ```bash
            .venv/bin/python -m src.create_final_notebook
            .venv/bin/jupyter nbconvert --to notebook --execute macroeconomic_var_varx_final_report.ipynb --output macroeconomic_var_varx_final_report.ipynb
            ```

            To launch the dashboard:

            ```bash
            .venv/bin/streamlit run dashboard_app.py --server.address 127.0.0.1 --server.port 8501
            ```
            """
        )
    )

    cells.append(
        md(
            """
            ## Appendix: Produced Output Files

            The main report above displays the most important results. The project also produces a larger set of output tables and figures. The code cell below lists them so any result can be opened directly from `outputs/tables/` or `outputs/figures/`.
            """
        )
    )
    cells.append(
        code(
            """
            output_files = []
            for folder in [TABLE_DIR, FIGURE_DIR]:
                for path in sorted(folder.glob("*")):
                    output_files.append({"type": folder.name, "path": str(path.relative_to(BASE_DIR))})
            outputs_index = pd.DataFrame(output_files)
            display(outputs_index)
            """
        )
    )

    nb.cells = cells
    return nb


def main() -> None:
    notebook = build_notebook()
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
