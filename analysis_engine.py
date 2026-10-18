"""
analysis_engine.py
==================
Stage 7: Data Analysis Engine

This module receives:
  1. df   — the pandas DataFrame uploaded in Stage 3
  2. plan — the structured analysis plan built by reasoning_engine.py (Stage 6)

It performs the actual pandas calculations and returns a structured
result dict that can be consumed by the Results & Insights page (Stage 8).

HOW IT WORKS (beginner-friendly, viva-ready)
---------------------------------------------
The engine reads the "analysis_type" field from the plan and routes
to the correct calculation function.  Each function uses only standard
pandas operations — no external libraries, no AI.

Supported analysis types:
  - single_aggregate    : mean / sum / min / max / count / median on one column
  - groupby_aggregate   : group by col A, apply aggregation on col B
  - value_counts        : count occurrences per unique value in a column
  - full_summary        : return the full DataFrame or a selected column

Return value structure (always a dict):
  {
    "status"       : "success" | "error"
    "analysis_type": str
    "operation"    : str
    "target_column": str | None
    "group_column" : str | None
    "aggregation"  : str
    "result_value" : scalar | None   (single-number answer)
    "result_df"    : pd.DataFrame | None  (tabular answer)
    "error_message": str | None
    "notes"        : list[str]       (human-readable explanation of what was done)
  }
"""

import pandas as pd
import numpy as np


# ==============================================================================
# HELPER: safe column access
# ==============================================================================
def _check_column(df: pd.DataFrame, col: str | None, role: str) -> str | None:
    """
    Verify a column name exists in df.
    Returns an error message string if invalid, or None if OK.
    """
    if col is None:
        return f"No {role} column was identified in the analysis plan."
    if col not in df.columns:
        return (
            f"{role.capitalize()} column '{col}' was not found in the dataset. "
            f"Available columns: {', '.join(df.columns.tolist())}"
        )
    return None


# ==============================================================================
# HELPER: coerce column to numeric safely
# ==============================================================================
def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Convert a column to numeric dtype using pd.to_numeric().
    Non-numeric values are replaced with NaN (errors='coerce').

    Why: If the column contains strings like "1,000" or "$500", pandas
    reads them as text.  to_numeric() converts what it can and silently
    drops the rest so calculations still work.
    """
    return pd.to_numeric(df[col], errors="coerce")


# ==============================================================================
# PANDAS AGGREGATION MAP
# ==============================================================================
# Maps the plan's aggregation name to the actual pandas Series method name.
# These are called with getattr(series, method_name)().
AGG_METHOD_MAP = {
    "mean"   : "mean",
    "sum"    : "sum",
    "min"    : "min",
    "max"    : "max",
    "count"  : "count",
    "median" : "median",
    "std"    : "std",       # standard deviation (bonus)
    "nunique": "nunique",   # count of unique values (bonus)
}


# ==============================================================================
# ENGINE: single_aggregate
# ==============================================================================
def _run_single_aggregate(df: pd.DataFrame, plan: dict) -> dict:
    """
    Apply a single aggregation function to one column.

    Example: "What is the average salary?"
      → df["Salary"].mean()

    Steps:
      1. Validate the target column exists.
      2. Convert column to numeric (handles string-formatted numbers).
      3. Drop NaN values so they don't corrupt the result.
      4. Call the aggregation method (mean / sum / min / max / count / median).
      5. Return the scalar result.
    """
    notes  = []
    target = plan.get("target_column")
    agg    = plan.get("aggregation", "mean")

    # ── Validate ─────────────────────────────────────────────────────────────
    err = _check_column(df, target, "target")
    if err:
        return _error(plan, err)

    # ── Coerce to numeric ────────────────────────────────────────────────────
    series = _to_numeric(df, target)
    notes.append(f"Converted column '{target}' to numeric dtype.")

    # ── Handle non-numeric columns ───────────────────────────────────────────
    if agg != "count" and series.isna().all():
        # count works on any dtype; other ops need numbers
        return _error(
            plan,
            f"Column '{target}' contains no numeric values. "
            "Cannot perform this aggregation."
        )

    non_null = series.dropna()
    notes.append(
        f"Found {len(non_null):,} non-null numeric values out of "
        f"{len(series):,} total rows."
    )

    # ── Calculate ────────────────────────────────────────────────────────────
    method = AGG_METHOD_MAP.get(agg, "mean")

    if agg == "count":
        # count works on any dtype — use the original column, not numeric-coerced
        result_value = int(df[target].count())
        notes.append(f"Calculated count of non-null values in '{target}'.")
    else:
        result_value = round(float(getattr(non_null, method)()), 4)
        notes.append(f"Calculated {agg}({target}) = {result_value:,}")

    # ── Build result DataFrame (single row for display) ─────────────────────
    result_df = pd.DataFrame({
        "Column"    : [target],
        "Operation" : [agg.upper()],
        "Result"    : [result_value],
    })

    return {
        "status"       : "success",
        "analysis_type": "single_aggregate",
        "operation"    : plan.get("operation"),
        "target_column": target,
        "group_column" : None,
        "aggregation"  : agg,
        "result_value" : result_value,
        "result_df"    : result_df,
        "error_message": None,
        "notes"        : notes,
    }


# ==============================================================================
# ENGINE: groupby_aggregate
# ==============================================================================
def _run_groupby_aggregate(df: pd.DataFrame, plan: dict) -> dict:
    """
    Group the DataFrame by one column, then aggregate another.

    Example: "Which product has the highest revenue?"
      → df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)

    Steps:
      1. Validate both group_column and target_column exist.
      2. Drop rows where either column is null (they would distort groups).
      3. Convert target_column to numeric.
      4. Group by group_column using df.groupby().
      5. Apply the aggregation (sum / mean / count / min / max / median).
      6. Sort results according to the plan (ascending or descending).
      7. Apply limit (e.g. top 1 if question asked "which has the highest").
      8. Reset the index so the result is a flat, printable DataFrame.
    """
    notes  = []
    target = plan.get("target_column")
    group  = plan.get("group_column")
    agg    = plan.get("aggregation", "sum")
    sort   = plan.get("sort", "descending")
    limit  = plan.get("limit")          # None means "return all"

    # ── Validate columns ─────────────────────────────────────────────────────
    for col, role in [(group, "group"), (target, "target")]:
        err = _check_column(df, col, role)
        if err:
            return _error(plan, err)

    # ── Drop rows where either key column is null ────────────────────────────
    working = df[[group, target]].copy()
    before  = len(working)
    working = working.dropna(subset=[group, target])
    after   = len(working)
    if before != after:
        notes.append(
            f"Dropped {before - after:,} rows that had missing values in "
            f"'{group}' or '{target}'."
        )

    if working.empty:
        return _error(plan, "No data remains after dropping rows with missing values.")

    # ── Coerce target to numeric ──────────────────────────────────────────────
    if agg != "count":
        working[target] = pd.to_numeric(working[target], errors="coerce")
        if working[target].isna().all():
            return _error(
                plan,
                f"Column '{target}' contains no numeric values. "
                "Cannot perform aggregation."
            )
        notes.append(f"Converted '{target}' to numeric. Ready for aggregation.")

    # ── Group & aggregate ────────────────────────────────────────────────────
    # df.groupby(group_col)[target_col].agg_func()
    method = AGG_METHOD_MAP.get(agg, "sum")

    if agg == "count":
        grouped = working.groupby(group)[target].count()
    else:
        grouped = working.groupby(group)[target].agg(method)

    notes.append(
        f"Grouped by '{group}', applied {agg}() on '{target}'. "
        f"Produced {len(grouped):,} group results."
    )

    # ── Sort ─────────────────────────────────────────────────────────────────
    ascending = (sort == "ascending")
    grouped   = grouped.sort_values(ascending=ascending)
    notes.append(
        f"Sorted results in {'ascending' if ascending else 'descending'} order."
    )

    # ── Limit ────────────────────────────────────────────────────────────────
    if limit:
        grouped = grouped.head(limit)
        notes.append(f"Applied limit: returning top {limit} result(s).")

    # ── Reset index → flat DataFrame ─────────────────────────────────────────
    result_df = grouped.reset_index()
    result_df.columns = [group, f"{agg.upper()}({target})"]
    result_df = result_df.round(4)

    # Scalar top result (first row of aggregated column)
    result_value = round(float(result_df.iloc[0, 1]), 4) if not result_df.empty else None

    return {
        "status"       : "success",
        "analysis_type": "groupby_aggregate",
        "operation"    : plan.get("operation"),
        "target_column": target,
        "group_column" : group,
        "aggregation"  : agg,
        "result_value" : result_value,
        "result_df"    : result_df,
        "error_message": None,
        "notes"        : notes,
    }


# ==============================================================================
# ENGINE: value_counts
# ==============================================================================
def _run_value_counts(df: pd.DataFrame, plan: dict) -> dict:
    """
    Count occurrences of each unique value in a column.

    Example: "How many customers are from each city?"
      → df["City"].value_counts()

    Steps:
      1. Validate the target column.
      2. Call Series.value_counts() which returns a Series sorted by frequency.
      3. Convert to a DataFrame for display.
    """
    notes  = []
    target = plan.get("target_column")

    err = _check_column(df, target, "target")
    if err:
        return _error(plan, err)

    # value_counts() counts how many times each unique value appears.
    # Automatically sorted from most frequent to least frequent.
    vc = df[target].value_counts().reset_index()
    vc.columns = [target, "Count"]

    notes.append(
        f"Counted occurrences of each unique value in '{target}'. "
        f"Found {len(vc):,} unique values."
    )

    return {
        "status"       : "success",
        "analysis_type": "value_counts",
        "operation"    : plan.get("operation"),
        "target_column": target,
        "group_column" : None,
        "aggregation"  : "value_counts",
        "result_value" : None,
        "result_df"    : vc,
        "error_message": None,
        "notes"        : notes,
    }


# ==============================================================================
# ENGINE: full_summary
# ==============================================================================
def _run_full_summary(df: pd.DataFrame, plan: dict) -> dict:
    """
    Return the full DataFrame or a selected column.

    Example: "Show me all employees"
      → return df (or df[["Employee"]])

    Steps:
      1. If a target column is specified, return just that column.
      2. Otherwise return the full DataFrame.
    """
    notes  = []
    target = plan.get("target_column")
    limit  = plan.get("limit")

    if target and target in df.columns:
        result_df = df[[target]].copy()
        notes.append(f"Returning column '{target}' from the dataset.")
    else:
        result_df = df.copy()
        notes.append("Returning the full dataset.")

    if limit:
        result_df = result_df.head(limit)
        notes.append(f"Showing top {limit} rows.")

    return {
        "status"       : "success",
        "analysis_type": "full_summary",
        "operation"    : plan.get("operation"),
        "target_column": target,
        "group_column" : None,
        "aggregation"  : "list",
        "result_value" : None,
        "result_df"    : result_df,
        "error_message": None,
        "notes"        : notes,
    }


# ==============================================================================
# HELPER: standardised error result
# ==============================================================================
def _error(plan: dict, message: str) -> dict:
    """Return a standard error result dict."""
    return {
        "status"       : "error",
        "analysis_type": plan.get("analysis_type", "unknown"),
        "operation"    : plan.get("operation"),
        "target_column": plan.get("target_column"),
        "group_column" : plan.get("group_column"),
        "aggregation"  : plan.get("aggregation"),
        "result_value" : None,
        "result_df"    : None,
        "error_message": message,
        "notes"        : [],
    }


# ==============================================================================
# MAIN: execute the plan
# ==============================================================================
def execute_plan(df: pd.DataFrame, plan: dict) -> dict:
    """
    Public entry point called by app.py.

    Receives:
        df   — the uploaded pandas DataFrame (from session_state["df"])
        plan — the structured dict produced by reasoning_engine.build_plan()

    Routes to the correct calculation function based on plan["analysis_type"].
    Always returns a result dict regardless of success or failure.
    """
    # ── Guard: empty dataframe ───────────────────────────────────────────────
    if df is None or df.empty:
        return _error(plan, "The dataset is empty. Please upload a valid file.")

    # ── Guard: plan not ready ────────────────────────────────────────────────
    if plan.get("status") == "cannot_plan":
        return _error(
            plan,
            "The reasoning engine could not generate a valid plan. "
            "Please rephrase your question."
        )

    analysis_type = plan.get("analysis_type", "unknown")

    # ── Route to the correct engine ──────────────────────────────────────────
    if analysis_type == "single_aggregate":
        return _run_single_aggregate(df, plan)

    elif analysis_type == "groupby_aggregate":
        return _run_groupby_aggregate(df, plan)

    elif analysis_type == "value_counts":
        return _run_value_counts(df, plan)

    elif analysis_type == "full_summary":
        return _run_full_summary(df, plan)

    else:
        return _error(
            plan,
            f"Unknown analysis type '{analysis_type}'. "
            "The reasoning engine produced an unrecognised plan."
        )
