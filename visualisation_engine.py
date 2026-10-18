"""
visualisation_engine.py
=======================
Data Visualization Module

This module takes the structured result produced by the Data Analysis Engine
and generates an interactive Plotly chart according to the user-selected chart type.

Supported Chart Types:
- Bar Chart
- Line Chart
- Scatter Plot
- Pie Chart
- Area Chart
- Histogram
- Box Plot
"""

import plotly.express as px
import pandas as pd


def generate_chart(result: dict, df: pd.DataFrame = None, chart_type: str = "Bar Chart"):
    """
    Generate a Plotly chart based on the analysis result and user-selected chart type.

    Parameters:
        result: The dictionary returned by execute_plan() in analysis_engine.py
        df: The original pandas DataFrame (used for full-distribution plots)
        chart_type: The user-selected chart type string

    Returns:
        A plotly Figure object, or None if a chart cannot be generated.
    """
    if not result or result.get("status") != "success":
        return None

    analysis_type = result.get("analysis_type")
    result_df = result.get("result_df")
    target_col = result.get("target_column")
    group_col = result.get("group_column")
    operation = (result.get("operation") or "Result").title()
    aggregation = (result.get("aggregation") or "").capitalize()

    fig = None

    # Common layout styling for consistent dark theme
    def _apply_theme(fig_obj):
        if fig_obj is not None:
            fig_obj.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0.6)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                margin=dict(l=40, r=40, t=60, b=40),
                font=dict(color="#e2e8f0")
            )
        return fig_obj

    # CASE 1: Groupby Aggregate Result (has group_col and aggregated target_col)
    if analysis_type == "groupby_aggregate" and result_df is not None and not result_df.empty:
        x_col = result_df.columns[0]
        y_col = result_df.columns[1]

        if chart_type == "Line Chart":
            fig = px.line(
                result_df, x=x_col, y=y_col, markers=True,
                title=f"{aggregation} {target_col} by {group_col}",
                color_discrete_sequence=['#38bdf8']
            )
        elif chart_type == "Scatter Plot":
            fig = px.scatter(
                result_df, x=x_col, y=y_col,
                size=y_col if pd.api.types.is_numeric_dtype(result_df[y_col]) and (result_df[y_col] > 0).all() else None,
                color=y_col,
                title=f"{aggregation} {target_col} vs {group_col}",
                color_continuous_scale=px.colors.sequential.Blues
            )
        elif chart_type == "Pie Chart":
            fig = px.pie(
                result_df, names=x_col, values=y_col,
                title=f"Proportion of {target_col} by {group_col}",
                hole=0.35,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
        elif chart_type == "Area Chart":
            fig = px.area(
                result_df, x=x_col, y=y_col,
                title=f"{aggregation} {target_col} by {group_col}",
                color_discrete_sequence=['#3b82f6']
            )
        elif chart_type == "Histogram":
            if df is not None and target_col in df.columns:
                fig = px.histogram(
                    df, x=target_col, color=group_col if group_col in df.columns else None,
                    title=f"Distribution of {target_col} across {group_col}",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
            else:
                fig = px.histogram(result_df, x=y_col, title=f"Distribution of {y_col}")
        elif chart_type == "Box Plot":
            if df is not None and target_col in df.columns:
                fig = px.box(
                    df, x=group_col if group_col in df.columns else None, y=target_col,
                    title=f"Box Plot of {target_col} by {group_col}",
                    color=group_col if group_col in df.columns else None
                )
            else:
                fig = px.box(result_df, y=y_col, title=f"Box Plot of {y_col}")
        else:
            # Default: Bar Chart
            fig = px.bar(
                result_df, x=x_col, y=y_col,
                title=f"{operation} of {target_col} by {group_col}",
                text_auto='.2s' if pd.api.types.is_numeric_dtype(result_df[y_col]) else True,
                color=y_col,
                color_continuous_scale=px.colors.sequential.Blues
            )
        return _apply_theme(fig)

    # CASE 2: Value Counts / Distribution Result
    elif analysis_type == "value_counts" and result_df is not None and not result_df.empty:
        x_col = result_df.columns[0]
        y_col = result_df.columns[1]
        plot_df = result_df.head(25)

        if chart_type == "Line Chart":
            fig = px.line(
                plot_df, x=x_col, y=y_col, markers=True,
                title=f"Distribution Trend of {target_col}",
                color_discrete_sequence=['#a855f7']
            )
        elif chart_type == "Pie Chart":
            fig = px.pie(
                plot_df.head(10), names=x_col, values=y_col,
                title=f"Distribution Share of {target_col}",
                hole=0.35,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
        elif chart_type == "Scatter Plot":
            fig = px.scatter(
                plot_df, x=x_col, y=y_col, size=y_col,
                title=f"Frequencies of {target_col}",
                color=y_col,
                color_continuous_scale=px.colors.sequential.Purples
            )
        elif chart_type == "Area Chart":
            fig = px.area(
                plot_df, x=x_col, y=y_col,
                title=f"Distribution Area of {target_col}",
                color_discrete_sequence=['#a855f7']
            )
        elif chart_type == "Histogram":
            if df is not None and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
                fig = px.histogram(
                    df, x=target_col, nbins=20,
                    title=f"Histogram of {target_col}",
                    color_discrete_sequence=['#3b82f6']
                )
            else:
                fig = px.histogram(plot_df, x=x_col, y=y_col, title=f"Frequency Histogram of {target_col}")
        elif chart_type == "Box Plot":
            if df is not None and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
                fig = px.box(df, y=target_col, title=f"Box Plot of {target_col}")
            else:
                fig = px.box(plot_df, y=y_col, title=f"Box Plot of Counts")
        else:
            # Default: Bar Chart
            fig = px.bar(
                plot_df, x=x_col, y=y_col,
                title=f"Count of {target_col}",
                text_auto=True,
                color=y_col,
                color_continuous_scale=px.colors.sequential.Purples
            )
        return _apply_theme(fig)

    # CASE 3: Single Aggregate (e.g. Average Salary, Total Sales)
    elif target_col and df is not None and target_col in df.columns:
        if chart_type == "Histogram" or chart_type == "Bar Chart":
            fig = px.histogram(
                df, x=target_col, nbins=25,
                title=f"Distribution of {target_col} (Aggregate: {result.get('result_value')})",
                color_discrete_sequence=['#38bdf8']
            )
        elif chart_type == "Box Plot":
            fig = px.box(df, y=target_col, title=f"Box Plot of {target_col}")
        elif chart_type == "Line Chart":
            fig = px.line(df.head(50), y=target_col, markers=True, title=f"{target_col} Value Sequence (First 50 Rows)")
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df.head(100), y=target_col, title=f"{target_col} Values (First 100 Rows)")
        elif chart_type == "Area Chart":
            fig = px.area(df.head(50), y=target_col, title=f"{target_col} Area Curve (First 50 Rows)")
        elif chart_type == "Pie Chart":
            if df[target_col].nunique() <= 15:
                counts = df[target_col].value_counts().reset_index()
                counts.columns = [target_col, "Count"]
                fig = px.pie(counts, names=target_col, values="Count", title=f"Proportion of {target_col}", hole=0.35)
            else:
                fig = px.histogram(df, x=target_col, nbins=20, title=f"Distribution of {target_col}")
        return _apply_theme(fig)

    return None

