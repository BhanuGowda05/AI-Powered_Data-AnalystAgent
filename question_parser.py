"""
question_parser.py
==================
Stage 5: Natural Language Question Understanding Engine

This module reads the user's plain-English question and tries to identify:
  - The operation  (average, maximum, minimum, count, sum, list, …)
  - The target column  (the column the operation applies to)
  - The grouping column  (the column to group by, if any)
  - A simple filter condition  (words like "where", "for", "in", "of")

HOW IT WORKS (beginner-friendly explanation)
---------------------------------------------
No external AI, LLM, or API is used.

The algorithm works in plain Python steps:

Step 1 - Keyword scanning
    We define dictionaries that map common English words to operation
    names.  We scan the user's question word-by-word and check for
    matches.

Step 2 - Column matching
    We tokenise both the question AND every column name into lowercase
    words.  If any word in the question appears as a word inside a
    column name (or vice-versa), we call it a match.

Step 3 - Role assignment
    The FIRST matching column that makes sense for the detected
    operation becomes the target column.  Any SECOND matching column
    becomes the grouping column.

Step 4 - Confidence scoring
    We report a confidence level: HIGH, MEDIUM, or LOW depending on
    how many parts of the interpretation we could fill.

This is a rule-based / keyword-based system — completely deterministic,
fully explainable, and easy to describe in a college viva.
"""

import re


# ==============================================================================
# KEYWORD DICTIONARIES
# ==============================================================================
# Each dictionary maps English words → a canonical operation name.
# We keep every list in lowercase.

OPERATION_KEYWORDS = {
    # Average / Mean
    "average": "average",
    "mean":    "average",
    "avg":     "average",

    # Maximum / Highest
    "maximum": "maximum",
    "max":     "maximum",
    "highest": "maximum",
    "largest": "maximum",
    "most":    "maximum",
    "top":     "maximum",
    "best":    "maximum",
    "peak":    "maximum",

    # Minimum / Lowest
    "minimum": "minimum",
    "min":     "minimum",
    "lowest":  "minimum",
    "smallest":"minimum",
    "least":   "minimum",
    "bottom":  "minimum",
    "worst":   "minimum",

    # Count
    "count":   "count",
    "how many":"count",
    "number":  "count",

    # Sum
    "sum":     "sum",
    "total":   "sum",           # "total" → sum (e.g. "total sales")
    "revenue": "sum",

    # List / Show
    "list":    "list",
    "show":    "list",
    "display": "list",
    "what are":"list",
    "give":    "list",

    # Distribution
    "distribution": "distribution",
    "spread":       "distribution",
    "range":        "distribution",
}

# Words that signal a GROUPING relationship
GROUP_KEYWORDS = {
    "by",        # "group by product"
    "per",       # "per category"
    "for each",  # "for each department"
    "each",
    "across",
    "within",
    "among",
    "between",
}

# Words that signal a FILTER condition
FILTER_KEYWORDS = {
    "where",
    "when",
    "if",
    "only",
    "with",
    "above",
    "below",
    "greater",
    "less",
    "equal",
    "in",
    "of",
    "for",
}


# ==============================================================================
# HELPER: clean and tokenise text
# ==============================================================================
def tokenise(text: str) -> list[str]:
    """
    Convert a sentence to a list of lowercase words.
    Removes punctuation using regex, then splits on whitespace.

    Example:
        "Which product has the highest revenue?"
        → ['which', 'product', 'has', 'the', 'highest', 'revenue']
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # replace punctuation with space
    return text.split()


# ==============================================================================
# HELPER: match question words against column names
# ==============================================================================
def _split_camel(text: str) -> list[str]:
    """
    Split CamelCase or compound identifiers into individual words.
    e.g. 'CustomerID' → ['customer', 'id']
         'ProductName' → ['product', 'name']
         'total_sales' → ['total', 'sales']
    """
    import re as _re
    # Insert space before uppercase letters that follow lowercase (CamelCase)
    spaced = _re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Replace underscores/hyphens with spaces, lowercase everything
    spaced = spaced.replace('_', ' ').replace('-', ' ').lower()
    return spaced.split()


def find_matching_columns(question_tokens: list[str],
                          df_columns: list[str]) -> list[str]:
    """
    Return the ordered list of DataFrame column names whose name words
    appear anywhere in the question tokens.

    Matching logic (robust word overlap + CamelCase splitting + plural stemming):
      - Tokenise each column name the same way AND split CamelCase parts.
      - Strip trailing 's' for simple plural matching (customers → customer).
      - A column is a match if ANY of its tokens appear in the question
        tokens, OR if ANY question token is a substring of a column token
        (or vice-versa), accounting for pluralisation.

    Returns a list of matched column names in the order they appear in df.
    """
    matched = []
    q_set = set(question_tokens)
    # Build set of stems (strip trailing 's' for plural matching)
    q_stems = {t.rstrip('s') for t in question_tokens if len(t) > 3}

    for col in df_columns:
        # Normal tokenisation
        col_tokens = tokenise(col)
        # Also split CamelCase (e.g. CustomerID → ['customer', 'id'])
        col_tokens_extended = list(set(col_tokens + _split_camel(col)))

        # Direct word overlap
        overlap = q_set.intersection(set(col_tokens_extended))

        # Stem overlap (plural matching: "customers" → "customer" matches "customerid" part)
        stem_match = any(
            stem in ct or ct.startswith(stem)
            for stem in q_stems
            for ct in col_tokens_extended
            if len(stem) >= 4 and len(ct) >= 4
        )

        # Substring check: question word inside column word or vice-versa
        substring_hit = any(
            qt in ct or ct in qt
            for qt in q_set
            for ct in col_tokens_extended
            if len(qt) >= 3 and len(ct) >= 3   # avoid matching very short words
        )

        if overlap or stem_match or substring_hit:
            matched.append(col)

    return matched


# ==============================================================================
# HELPER: split matched columns into numerical and categorical by dtype hint
# ==============================================================================
_NUMERIC_HINT_WORDS = {
    "salary", "revenue", "sales", "price", "cost", "amount", "total",
    "count", "qty", "quantity", "score", "rate", "income", "profit",
    "age", "value", "weight", "height", "number", "id"
}
_CATEGORICAL_HINT_WORDS = {
    "product", "category", "department", "region", "city", "country",
    "name", "type", "status", "gender", "group", "brand", "segment",
    "customer", "employee", "location", "store", "team"
}

def _split_by_role(matched_columns: list[str]) -> tuple[list[str], list[str]]:
    """
    Heuristically split matched columns into likely-numeric (target) and
    likely-categorical (group) based on keyword hints in the column name.

    Returns (numeric_cols, categorical_cols) — either list may be empty.
    """
    numeric = []
    categorical = []
    for col in matched_columns:
        tokens = set(tokenise(col))
        if tokens & _NUMERIC_HINT_WORDS:
            numeric.append(col)
        elif tokens & _CATEGORICAL_HINT_WORDS:
            categorical.append(col)
        else:
            # No hint — put in numeric by default (aggregation is more common)
            numeric.append(col)
    return numeric, categorical


# ==============================================================================
# HELPER: detect the operation from the question
# ==============================================================================
def detect_operation(question: str) -> str | None:
    """
    Scan the question for known operation keywords and return the
    canonical operation name, or None if nothing is found.

    We check multi-word phrases first (e.g. "how many") before single
    words to avoid partial mis-matches.
    """
    q_lower = question.lower()

    # Check multi-word phrases first (longer ones take priority)
    multi_word = sorted(
        [k for k in OPERATION_KEYWORDS if " " in k],
        key=len,
        reverse=True
    )
    for phrase in multi_word:
        if phrase in q_lower:
            return OPERATION_KEYWORDS[phrase]

    # Check single words
    tokens = tokenise(question)
    for token in tokens:
        if token in OPERATION_KEYWORDS:
            return OPERATION_KEYWORDS[token]

    return None


# ==============================================================================
# HELPER: detect if a grouping keyword is present
# ==============================================================================
def has_group_keyword(question: str) -> bool:
    """Return True if the question contains any grouping signal word."""
    q_lower = question.lower()
    for kw in GROUP_KEYWORDS:
        if kw in q_lower:
            return True
    return False


# ==============================================================================
# HELPER: detect if a filter keyword is present
# ==============================================================================
def has_filter_keyword(question: str) -> bool:
    """Return True if the question likely contains a filter condition."""
    tokens = tokenise(question)
    for token in tokens:
        if token in FILTER_KEYWORDS:
            return True
    return False


# ==============================================================================
# MAIN: parse the full question
# ==============================================================================
def parse_question(question: str, df_columns: list[str]) -> dict:
    """
    Analyse the user's question and return a structured interpretation dict.

    Returns
    -------
    dict with keys:
        original_question  : str   — the raw question
        operation          : str   — e.g. "average", "maximum", "count"
        target_column      : str   — the column the operation applies to
        group_column       : str   — the column to group by (or None)
        has_filter         : bool  — True if a filter signal was found
        matched_columns    : list  — all columns matched from the question
        confidence         : str   — "HIGH", "MEDIUM", or "LOW"
        notes              : list  — human-readable explanation steps
    """
    result = {
        "original_question": question,
        "operation":         None,
        "target_column":     None,
        "group_column":      None,
        "has_filter":        False,
        "matched_columns":   [],
        "confidence":        "LOW",
        "notes":             [],
    }

    if not question.strip():
        result["notes"].append("❌ Empty question — nothing to parse.")
        return result

    # ── Step 1: Detect operation ────────────────────────────────────────────
    operation = detect_operation(question)
    result["operation"] = operation

    if operation:
        result["notes"].append(f"✅ Operation detected: **{operation}**")
    else:
        result["notes"].append(
            "⚠️ No operation keyword found (e.g. average, sum, count, highest)."
        )

    # ── Step 2: Match columns ────────────────────────────────────────────────
    question_tokens = tokenise(question)
    matched = find_matching_columns(question_tokens, df_columns)
    result["matched_columns"] = matched

    if matched:
        result["notes"].append(
            f"✅ Column(s) matched from question: **{', '.join(matched)}**"
        )
    else:
        result["notes"].append(
            "⚠️ No column names could be matched in the question. "
            "Try using words that appear in your column names."
        )

    # ── Step 3: Assign target and group columns ──────────────────────────────
    # Smart role assignment:
    #   - Numerical columns  → target (the column being aggregated)
    #   - Categorical columns → group  (the column to group by)
    # If a grouping keyword is found AND we have 2+ columns, we use
    # dtype-hint splitting. Otherwise, fall back to order in df.

    if len(matched) >= 2:
        # Use smart dtype-hint splitting to assign roles correctly
        numeric_cols, categorical_cols = _split_by_role(matched)

        if numeric_cols and categorical_cols:
            # Classic case: e.g. "Which product has highest revenue?"
            # → group_column = product (categorical), target = revenue (numeric)
            result["group_column"]  = categorical_cols[0]
            result["target_column"] = numeric_cols[0]
            result["notes"].append(
                f"✅ Smart role detection → "
                f"group by **{categorical_cols[0]}** (categorical), "
                f"apply **{operation or '?'}** on **{numeric_cols[0]}** (numerical)"
            )
        elif has_group_keyword(question):
            # Fallback: use grouping keyword position
            result["group_column"]  = matched[0]
            result["target_column"] = matched[1]
            result["notes"].append(
                f"✅ Grouping keyword found → "
                f"group by **{matched[0]}**, "
                f"apply **{operation or '?'}** on **{matched[1]}**"
            )
        else:
            # Both same type — first is target, second is group
            result["target_column"] = matched[0]
            result["group_column"]  = matched[1]
            result["notes"].append(
                f"✅ Target column: **{matched[0]}** | "
                f"Possible grouping column: **{matched[1]}**"
            )
    elif len(matched) == 1:
        result["target_column"] = matched[0]
        result["notes"].append(f"✅ Target column: **{matched[0]}**")

    # ── Step 4: Filter detection ─────────────────────────────────────────────
    result["has_filter"] = has_filter_keyword(question)
    if result["has_filter"]:
        result["notes"].append(
            "ℹ️ Filter signal detected (e.g. 'where', 'for', 'above'). "
            "Exact filter values will be handled in a future stage."
        )

    # ── Step 5: Confidence scoring ───────────────────────────────────────────
    filled = sum([
        result["operation"]     is not None,
        result["target_column"] is not None,
        result["group_column"]  is not None,
    ])
    if filled == 3:
        result["confidence"] = "HIGH"
    elif filled == 2:
        result["confidence"] = "HIGH"
    elif filled == 1:
        result["confidence"] = "MEDIUM"
    else:
        result["confidence"] = "LOW"

    return result
