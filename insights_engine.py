"""
insights_engine.py
==================
Stage 9: Final Answer and Insights Generation

This module takes the structured result dictionary from the Data Analysis Engine
and generates a plain-English final answer using simple Python templates.
It also derives key insights and supporting statistics strictly from the
actual pandas calculation results — no invented values, no external LLMs.

HOW IT WORKS (beginner-friendly explanation)
--------------------------------------------
No external AI or API is used.

The module reads:
  - analysis_type  → what kind of analysis was done (groupby, single, etc.)
  - operation      → the user's requested operation (maximum, average, etc.)
  - result_value   → the computed scalar answer (if any)
  - result_df      → the computed table answer (if any)
  - plan           → the reasoning plan (for sort direction, limit, etc.)

Then it picks the right English sentence template and fills it in with the
real numbers from the calculation results.

Supporting statistics are derived from result_df using pandas methods:
  - mean, std, max, min, count — all computed from actual data.

This is Stage 9 of the AI-Powered Data Analyst Agent pipeline.
"""

import pandas as pd
import numpy as np


# ==============================================================================
# HELPER: format a number for display
# ==============================================================================
def _fmt(val) -> str:
    """
    Return a cleanly formatted string for a number.
    - Integers  → comma-separated, no decimal  (e.g. 45,000)
    - Floats    → comma-separated, 2 decimals  (e.g. 1,234.56)
    - NaN/inf   → "N/A"  (prevents ValueError from int(NaN))
    - Anything else → plain str()
    """
    if val is None:
        return "N/A"
    if isinstance(val, float):
        # Guard against NaN and infinity before calling int()
        import math
        if math.isnan(val) or math.isinf(val):
            return "N/A"
        if val == int(val):         # e.g. 45000.0 → show as 45,000
            return f"{int(val):,}"
        return f"{val:,.2f}"
    if isinstance(val, (int, np.integer)):
        return f"{int(val):,}"
    return str(val)


# ==============================================================================
# HELPER: compute supporting statistics from a numeric Series
# ==============================================================================
def _supporting_stats(series: pd.Series, col_name: str) -> list:
    """
    Given a numeric pandas Series, compute supporting statistics
    and return them as a list of dicts: { "label": str, "value": str }.
    All values come directly from pandas — nothing invented.
    """
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return []

    stats = [
        {"label": f"Average {col_name}",    "value": _fmt(series.mean())},
        {"label": f"Maximum {col_name}",    "value": _fmt(series.max())},
        {"label": f"Minimum {col_name}",    "value": _fmt(series.min())},
        {"label": f"Std Dev ({col_name})",  "value": _fmt(series.std())},
        {"label": "Non-null Rows Used",     "value": _fmt(int(series.count()))},
    ]
    return stats


# ==============================================================================
# MAIN: generate final answer, insights, and supporting statistics
# ==============================================================================
def generate_answer_and_insights(result: dict, plan: dict) -> tuple:
    """
    Generate a templated English answer, key insights, and supporting statistics
    based solely on the actual pandas calculation results.

    Parameters
    ----------
    result : dict   — returned by execute_plan() in analysis_engine.py
    plan   : dict   — returned by build_plan() in reasoning_engine.py

    Returns
    -------
    tuple of:
        final_answer  : str        — one clear English sentence
        insights      : list[str]  — bullet-point key observations
        stats         : list[dict] — supporting statistics as {label, value} dicts
    """

    # ── Guard: analysis failed ────────────────────────────────────────────────
    if result.get("status") != "success":
        error_msg = result.get("error_message", "Unknown error")
        return (
            f"I could not answer your question. Reason: {error_msg}",
            ["Please check that the column name in your question matches the dataset columns."],
            []
        )

    # ── Extract commonly needed fields ────────────────────────────────────────
    analysis_type = result.get("analysis_type", "")
    target_col    = result.get("target_column") or "the column"
    group_col     = result.get("group_column")  or "the category"
    operation     = result.get("operation")     or "analysis"
    aggregation   = result.get("aggregation")   or "aggregate"
    result_value  = result.get("result_value")
    result_df     = result.get("result_df")

    sort_dir = plan.get("sort", "none")
    limit    = plan.get("limit")

    answer   = ""
    insights = []
    stats    = []

    # ==========================================================================
    # CASE 1: SINGLE AGGREGATE
    # Example: "What is the average salary?" → df["Salary"].mean() = 52000
    # ==========================================================================
    if analysis_type == "single_aggregate":

        if result_value is not None:
            val_str = _fmt(result_value)
            op = operation.lower()

            if op == "average":
                answer = f"The average {target_col} is **{val_str}**."
            elif op == "maximum":
                answer = f"The maximum value in {target_col} is **{val_str}**."
            elif op == "minimum":
                answer = f"The minimum value in {target_col} is **{val_str}**."
            elif op == "sum":
                answer = f"The total (sum) of {target_col} is **{val_str}**."
            elif op == "count":
                answer = f"There are **{val_str}** non-null entries in the {target_col} column."
            else:
                answer = f"The {operation} of {target_col} is **{val_str}**."

            insights.append(
                f"This value was computed from the **{target_col}** column "
                f"using the **{aggregation}** function."
            )
            insights.append(
                "Missing (null) values were excluded automatically before the calculation."
            )

            # Supporting statistics: we have only the scalar result here
            stats.append({"label": f"{aggregation.upper()} of {target_col}", "value": val_str})

        else:
            answer = f"I could not compute the {operation} for {target_col}."

    # ==========================================================================
    # CASE 2: GROUPBY AGGREGATE
    # Example: "Which product has highest revenue?" → Product A = 45,000
    # ==========================================================================
    elif analysis_type == "groupby_aggregate":

        if result_df is not None and not result_df.empty:
            y_col     = result_df.columns[1]
            top_row   = result_df.iloc[0]
            top_group = top_row[group_col]
            top_val   = top_row[y_col]
            val_str   = _fmt(top_val)

            if limit == 1:
                if sort_dir == "descending":
                    answer = (
                        f"**{top_group}** has the highest {target_col} "
                        f"with a total of **{val_str}**."
                    )
                elif sort_dir == "ascending":
                    answer = (
                        f"**{top_group}** has the lowest {target_col} "
                        f"with a value of **{val_str}**."
                    )
                else:
                    answer = (
                        f"The top result for {target_col} grouped by {group_col} is "
                        f"**{top_group}** with **{val_str}**."
                    )
            else:
                answer = (
                    f"The {aggregation} of {target_col} grouped by {group_col} "
                    f"has been calculated. The leading {group_col} is "
                    f"**{top_group}** with **{val_str}**."
                )

            # Key insights — derived entirely from result_df
            insights.append(f"🥇 The highest {group_col} is **{top_group}** ({val_str}).")

            if len(result_df) > 1:
                bot_row   = result_df.iloc[-1]
                bot_group = bot_row[group_col]
                bot_val   = bot_row[y_col]
                bot_str   = _fmt(bot_val)
                insights.append(
                    f"🔻 The lowest {group_col} in this result is **{bot_group}** ({bot_str})."
                )
                if isinstance(top_val, (int, float)) and isinstance(bot_val, (int, float)):
                    diff = top_val - bot_val
                    if diff > 0:
                        insights.append(
                            f"📊 The gap between the highest and lowest is **{_fmt(diff)}**."
                        )

            num_groups = len(result_df)
            insights.append(
                f"📂 A total of **{num_groups}** unique {group_col}(s) were found in the dataset."
            )

            # Supporting statistics from the aggregated numeric column
            if len(result_df.columns) > 1:
                agg_col = result_df.columns[1]
                stats = _supporting_stats(result_df[agg_col], agg_col)

        else:
            answer = f"I grouped the data by {group_col}, but no valid results were returned."

    # ==========================================================================
    # CASE 3: VALUE COUNTS
    # Example: "How many orders per category?" → Electronics=450, Clothing=200
    # ==========================================================================
    elif analysis_type == "value_counts":

        if result_df is not None and not result_df.empty:
            count_col  = result_df.columns[1]
            top_row    = result_df.iloc[0]
            top_cat    = top_row[result_df.columns[0]]
            top_count  = top_row[count_col]
            total_unique = len(result_df)
            total_rows   = int(result_df[count_col].sum())

            answer = (
                f"There are **{total_unique}** unique categories in **{target_col}**. "
                f"The most common is **'{top_cat}'** which appears **{_fmt(top_count)}** times "
                f"out of **{_fmt(total_rows)}** total records."
            )

            insights.append(
                f"🏆 The most frequent {target_col} is **{top_cat}** ({_fmt(top_count)} occurrences)."
            )
            if len(result_df) > 1:
                bot_row   = result_df.iloc[-1]
                bot_cat   = bot_row[result_df.columns[0]]
                bot_count = bot_row[count_col]
                insights.append(
                    f"📉 The least common {target_col} is **{bot_cat}** ({_fmt(bot_count)} occurrences)."
                )
            insights.append(
                f"📂 **{total_unique}** unique values across **{_fmt(total_rows)}** total rows."
            )

            stats = _supporting_stats(result_df[count_col], "Count")

        else:
            answer = f"No category distribution could be computed for {target_col}."

    # ==========================================================================
    # CASE 4: FULL SUMMARY
    # ==========================================================================
    elif analysis_type == "full_summary":

        if result_df is not None:
            num_rows = len(result_df)
            if target_col and target_col != "the column":
                answer = (
                    f"Here is the data for the **{target_col}** column — "
                    f"showing **{_fmt(num_rows)}** row(s)."
                )
            else:
                answer = f"Here is the full dataset — **{_fmt(num_rows)}** row(s) returned."

            insights.append(f"📋 **{_fmt(num_rows)}** rows are shown in the result below.")
            if limit:
                insights.append(
                    f"ℹ️ Only the top **{limit}** rows were requested by the question."
                )
        else:
            answer = "The dataset was returned but appears to be empty."

    # ==========================================================================
    # CASE 5: UNKNOWN
    # ==========================================================================
    else:
        answer = (
            "The analysis completed but I cannot generate a specific natural-language "
            "answer for this analysis type. Please review the result table below."
        )
        insights.append(
            "Try a question with clearer keywords: average, highest, lowest, count, sum."
        )

    return answer, insights, stats
