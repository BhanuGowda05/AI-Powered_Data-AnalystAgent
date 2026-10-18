"""
reasoning_engine.py
====================
Stage 6: Agent Reasoning Layer

This module takes the structured question interpretation produced by
question_parser.py (Stage 5) and converts it into a concrete,
step-by-step analysis plan.

WHY SEPARATE REASONING FROM CALCULATION? (Viva explanation)
------------------------------------------------------------
Keeping reasoning and calculation in separate modules is a core
principle of good software design called "Separation of Concerns":

  - reasoning_engine.py   → WHAT to do and WHY  (the plan)
  - analysis_engine.py    → HOW to do it        (the execution, Stage 7)

Benefits:
  1. EXPLAINABILITY  — The user can read the plan before any numbers
                       are computed. This builds trust.
  2. DEBUGGABILITY   — If the answer is wrong, we can inspect the plan
                       to find the mistake without running calculations.
  3. MODULARITY      — The calculation engine can be swapped or improved
                       without touching the reasoning layer at all.
  4. TESTABILITY     — Each layer can be unit-tested independently.

This mirrors how real AI agent frameworks (like LangChain's "chains" or
OpenAI's "function calling") separate intent-detection from execution.
Here we do the same thing using pure, beginner-friendly Python.

HOW THE REASONING ENGINE WORKS
-------------------------------
Input  : a dict produced by parse_question() from question_parser.py
Output : a plan dict + a human-readable list of numbered reasoning steps

Step 1  Read the detected operation from the NLU result.
Step 2  Map the operation to one of our supported analysis types:
            single_aggregate    → mean / min / max / sum / count on one column
            groupby_aggregate   → group by column A, aggregate column B
            value_counts        → count occurrences per category
            full_summary        → show all data / list all rows
Step 3  Decide sort direction and limit based on the operation keyword.
Step 4  Build the plan dict (JSON-like structure).
Step 5  Generate numbered, English reasoning steps that match the plan.
"""

# ==============================================================================
# MAPPING: canonical operation → aggregation function name
# ==============================================================================
# These are the pandas aggregation function names we will use in Stage 7.
OPERATION_TO_AGG = {
    "average":      "mean",
    "maximum":      "max",
    "minimum":      "min",
    "sum":          "sum",
    "count":        "count",
    "distribution": "value_counts",
    "list":         "list",
    "median":       "median",   # added
}

# Which operations produce a sorted descending result?
DESCENDING_OPS = {"maximum", "sum"}

# Which operations produce a sorted ascending result?
ASCENDING_OPS  = {"minimum"}

# Numeric-column hint words (mirrors question_parser._NUMERIC_HINT_WORDS)
_NUMERIC_HINTS = {
    "salary", "revenue", "sales", "price", "cost", "amount", "total",
    "qty", "quantity", "score", "rate", "income", "profit",
    "age", "value", "weight", "height", "units", "discount", "number",
}

# Categorical-column hint words
_CATEGORICAL_HINTS = {
    "product", "category", "department", "region", "city", "country",
    "name", "type", "status", "gender", "group", "brand", "segment",
    "customer", "employee", "location", "store", "team",
}


def _guess_numeric_column(df_columns: list[str]) -> str | None:
    """
    When no target column was matched from the question, pick the most
    appropriate numeric-looking column from the DataFrame's column list.
    Skips ID, Date, Name columns as they are rarely the analysis target.
    """
    # Pass 1: column whose name contains a known numeric hint word
    for col in df_columns:
        col_words = set(col.lower().replace("_", " ").split())
        if col_words & _NUMERIC_HINTS:
            return col
    # Pass 2: first column that doesn't look like an ID / date / name
    skip = {"id", "date", "time", "name", "code", "key"}
    for col in df_columns:
        col_words = set(col.lower().replace("_", " ").split())
        if not (col_words & skip):
            return col
    return df_columns[0] if df_columns else None


def _guess_categorical_column(df_columns: list[str]) -> str | None:
    """
    Pick the most appropriate categorical-looking column when none matched.
    """
    for col in df_columns:
        col_words = set(col.lower().replace("_", " ").split())
        if col_words & _CATEGORICAL_HINTS:
            return col
    return df_columns[0] if df_columns else None


# ==============================================================================
# HELPER: determine the analysis type
# ==============================================================================
def _determine_analysis_type(operation: str | None,
                              target_col: str | None,
                              group_col:  str | None) -> str:
    """
    Decide which high-level analysis pattern applies.

    Returns one of:
        "groupby_aggregate"  — group by a column, then aggregate another
        "single_aggregate"   — apply one function to a single column
        "value_counts"       — count occurrences of each category value
        "full_summary"       — list / show all rows
        "unknown"            — not enough information to decide
    """
    if operation == "distribution" or (operation == "count" and target_col and not group_col):
        # "How many X per category?" or "distribution of X"
        return "value_counts"

    if operation == "list":
        return "full_summary"

    if group_col and target_col and operation in OPERATION_TO_AGG:
        return "groupby_aggregate"

    if target_col and operation in OPERATION_TO_AGG:
        return "single_aggregate"

    return "unknown"


# ==============================================================================
# HELPER: decide sort direction
# ==============================================================================
def _sort_direction(operation: str | None) -> str:
    if operation in DESCENDING_OPS:
        return "descending"
    if operation in ASCENDING_OPS:
        return "ascending"
    return "none"


# ==============================================================================
# MAIN: build the analysis plan
# ==============================================================================
def build_plan(nlu_result: dict, df_columns: list[str]) -> dict:
    """
    Convert the NLU interpretation dict into a structured analysis plan.

    Parameters
    ----------
    nlu_result  : dict returned by parse_question() in question_parser.py
    df_columns  : list of actual column names in the DataFrame

    Returns
    -------
    dict with keys:
        status          : "ready" | "partial" | "cannot_plan"
        analysis_type   : str
        operation       : str
        aggregation     : str   (pandas function name)
        group_column    : str | None
        target_column   : str | None
        sort            : str   ("ascending" | "descending" | "none")
        limit           : int | None  (e.g. 1 for "which has the highest")
        has_filter      : bool
        steps           : list[str]   (numbered reasoning steps)
        warnings        : list[str]   (issues found during planning)
    """
    operation    = nlu_result.get("operation")
    target_col   = nlu_result.get("target_column")
    group_col    = nlu_result.get("group_column")
    has_filter   = nlu_result.get("has_filter", False)
    confidence   = nlu_result.get("confidence", "LOW")

    # analysis_type and aggregation are computed AFTER the column fallback below
    sort_direction = _sort_direction(operation)
    aggregation    = OPERATION_TO_AGG.get(operation, "unknown")   # preliminary
    analysis_type  = _determine_analysis_type(operation, target_col, group_col)  # preliminary

    # "highest / lowest / best / worst" implies limit = 1
    question_lower = nlu_result.get("original_question", "").lower()
    limit_keywords = {"highest", "lowest", "most", "least", "top", "bottom",
                      "best", "worst", "peak", "maximum", "minimum"}
    limit = 1 if any(kw in question_lower for kw in limit_keywords) else None

    warnings = []
    steps    = []

    # ── Validate columns still exist in df ──────────────────────────────────
    if target_col and target_col not in df_columns:
        warnings.append(
            f"⚠️ Target column '{target_col}' was not found in the dataset."
        )
        target_col = None

    if group_col and group_col not in df_columns:
        warnings.append(
            f"⚠️ Group column '{group_col}' was not found in the dataset."
        )
        group_col = None

    # ── Smart fallback: guess best column when target_col is still None ──────
    # Handles "What is the total?" / "What is the average?" with no column match.
    if target_col is None and operation in ("average", "maximum", "minimum",
                                            "sum", "median", "count"):
        guessed = _guess_numeric_column(df_columns)
        if guessed:
            target_col = guessed
            warnings.append(
                f"ℹ️ No specific column detected in your question — "
                f"analysing **'{target_col}'** as the best match. "
                f"Mention a column name for a precise answer."
            )

    if target_col is None and operation == "distribution":
        guessed = _guess_categorical_column(df_columns)
        if guessed:
            target_col = guessed
            warnings.append(
                f"ℹ️ No column detected — showing distribution of **'{target_col}'**."
            )

    # Re-compute analysis_type and aggregation now that target_col may be filled
    analysis_type  = _determine_analysis_type(operation, target_col, group_col)
    aggregation    = OPERATION_TO_AGG.get(operation, "unknown")

    # ── Determine plan status ────────────────────────────────────────────────
    if analysis_type == "unknown" or confidence == "LOW":
        status = "cannot_plan"
    elif target_col is None and analysis_type not in ("full_summary",):
        status = "partial"
    else:
        status = "ready"

    # ── Build numbered reasoning steps ───────────────────────────────────────
    step_num = 1

    def add(text: str):
        nonlocal step_num
        steps.append(f"Step {step_num}: {text}")
        step_num += 1

    if status == "cannot_plan":
        add("Read the user's question.")
        add("Attempted to detect an operation keyword — none found or confidence too low.")
        add("Attempted to match question words to column names — no reliable match.")
        add(
            "Cannot generate a plan. Ask the user to rephrase using column "
            "names visible in the Dataset Overview."
        )

    elif analysis_type == "groupby_aggregate":
        add(f"Read the user's question: \"{nlu_result.get('original_question', '')}\"")
        add(f"Detected operation keyword → **{operation}**")
        add(f"Matched grouping column from question → **{group_col}**")
        add(f"Matched target column from question   → **{target_col}**")
        add(f"Determine aggregation function         → **{aggregation}** (pandas `{aggregation}()`)")
        add(f"Group the DataFrame by **{group_col}**")
        add(f"For each group, calculate **{aggregation}** of **{target_col}**")
        if sort_direction != "none":
            add(f"Sort results in **{sort_direction}** order by **{target_col}**")
        if limit:
            add(f"Select the **top {limit}** result(s) from sorted output")
        if has_filter:
            add("⚠️ Filter signal detected — filter conditions will be applied in the execution stage")
        add("Pass the completed plan to the Data Analysis Engine (Stage 7) for execution")

    elif analysis_type == "single_aggregate":
        add(f"Read the user's question: \"{nlu_result.get('original_question', '')}\"")
        add(f"Detected operation keyword → **{operation}**")
        add(f"Matched target column from question → **{target_col}**")
        add(f"Determine aggregation function       → **{aggregation}** (pandas `{aggregation}()`)")
        add(f"Apply **{aggregation}** directly to the column **{target_col}**")
        if has_filter:
            add("⚠️ Filter signal detected — filter conditions will be applied in the execution stage")
        add("Pass the completed plan to the Data Analysis Engine (Stage 7) for execution")

    elif analysis_type == "value_counts":
        add(f"Read the user's question: \"{nlu_result.get('original_question', '')}\"")
        add(f"Detected operation → **count / distribution**")
        add(f"Matched target column → **{target_col}**")
        add(f"Apply `value_counts()` on **{target_col}** to count occurrences per unique value")
        add("Sort results by frequency in descending order")
        add("Pass the completed plan to the Data Analysis Engine (Stage 7) for execution")

    elif analysis_type == "full_summary":
        add(f"Read the user's question: \"{nlu_result.get('original_question', '')}\"")
        add("Detected operation → **list / show all**")
        add("No aggregation needed — return the full dataset or selected columns")
        if target_col:
            add(f"Focus output on column: **{target_col}**")
        add("Pass the completed plan to the Data Analysis Engine (Stage 7) for execution")

    # ── Assemble final plan dict ─────────────────────────────────────────────
    plan = {
        "status":        status,
        "analysis_type": analysis_type,
        "operation":     operation,
        "aggregation":   aggregation,
        "group_column":  group_col,
        "target_column": target_col,
        "sort":          sort_direction,
        "limit":         limit,
        "has_filter":    has_filter,
        "steps":         steps,
        "warnings":      warnings,
    }

    return plan
