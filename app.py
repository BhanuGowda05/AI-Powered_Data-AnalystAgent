"""
Data analyst agent
==================
An explainable data analysis assistant built with Python and Streamlit.
Pipeline: NLU → Reasoning → Calculation → Visualization → Final Answer
All numerical values come directly from pandas results.
No external LLM or API is used.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Import local modules with hot-reload support for Streamlit server runtime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
import question_parser
import reasoning_engine
import analysis_engine
import visualisation_engine
import insights_engine

importlib.reload(question_parser)
importlib.reload(reasoning_engine)
importlib.reload(analysis_engine)
importlib.reload(visualisation_engine)
importlib.reload(insights_engine)

from question_parser  import parse_question
from reasoning_engine import build_plan
from analysis_engine  import execute_plan
from visualisation_engine import generate_chart
from insights_engine import generate_answer_and_insights

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Data analyst agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. LOAD EXTERNAL CSS
# ==============================================================================
def load_css(css_path: str):
    """
    Reads an external CSS file and injects it into the Streamlit app.
    This keeps Python code focused on UI structure and logic.
    """
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Locate and apply assets/style.css relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.path.join(CURRENT_DIR, "assets", "style.css")
load_css(CSS_PATH)

# ==============================================================================
# 3. SESSION STATE INITIALISATION
# ==============================================================================
if "df" not in st.session_state:
    st.session_state["df"] = None          # Will hold the pandas DataFrame

if "filename" not in st.session_state:
    st.session_state["filename"] = None    # Will hold the uploaded filename

if "user_question" not in st.session_state:
    st.session_state["user_question"] = "" # Holds the active user query across modules

if "cleaned_df" not in st.session_state:
    st.session_state["cleaned_df"] = None   # Will hold the cleaned DataFrame

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Dashboard"  # For programmatic navigation

# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
# Sidebar Header / Branding
st.sidebar.markdown("""
<div class="sidebar-brand">
    <h2>🤖 Data analyst agent</h2>
</div>
""", unsafe_allow_html=True)

# Navigation Options
nav_options = [
    "Dashboard",
    "Dataset Upload",
    "Dataset Overview",
    "Ask Dataset & Agent Reasoning",
    "Data Analysis Engine",
    "Results & Insights"
]

# Derive the index from current_page so the radio moves when we
# set current_page + call st.rerun() (works without a widget key).
if st.session_state["current_page"] in nav_options:
    _nav_index = nav_options.index(st.session_state["current_page"])
else:
    _nav_index = 0

selected_page = st.sidebar.radio(
    "Navigation Menu",
    options=nav_options,
    index=_nav_index
)

# Keep current_page in sync when the user clicks the sidebar directly
st.session_state["current_page"] = selected_page

# Sidebar: show active dataset badge if loaded
if st.session_state["df"] is not None:
    _fname = st.session_state["filename"] or "dataset"
    _rows  = st.session_state["df"].shape[0]
    _cols  = st.session_state["df"].shape[1]
    st.sidebar.markdown(
        f"""
        <div style="background:#0f172a;border:1px solid #1e40af;
             border-radius:8px;padding:0.6rem 0.9rem;margin-top:0.5rem;">
            <p style="margin:0;color:#60a5fa;font-size:0.7rem;
               text-transform:uppercase;letter-spacing:0.05em;">Active Dataset</p>
            <p style="margin:0.15rem 0 0 0;color:#f1f5f9;font-size:0.82rem;
               font-weight:600;white-space:nowrap;overflow:hidden;
               text-overflow:ellipsis;">{_fname}</p>
            <p style="margin:0.1rem 0 0 0;color:#94a3b8;font-size:0.72rem;">
               {_rows:,} rows &times; {_cols} cols</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# 4. HELPER: NEXT-PAGE BUTTON
# ==============================================================================
def next_button(label: str, target_page: str, key: str) -> None:
    """Render a centred 'Next' button that jumps to target_page."""
    st.markdown("---")
    _nb_c1, _nb_c2, _nb_c3 = st.columns([3, 2, 3])
    with _nb_c2:
        if st.button(label, key=key, use_container_width=True, type="primary"):
            st.session_state["current_page"] = target_page
            st.rerun()

# ==============================================================================
# 5. PAGE ROUTING & CONTENT RENDERING
# ==============================================================================

# ------------------------------------------------------------------------------
# PAGE 1: DASHBOARD
# ------------------------------------------------------------------------------
if selected_page == "Dashboard":
    # Professional Project Header (Hero Banner)
    st.markdown("""
    <div class="hero-header">
        <span class="badge-pill badge-stage">Data Analytics System</span>
        <h1>Data analyst agent</h1>
        <p>
            An autonomous, explainable data analysis assistant built to ingest, inspect, 
            and analyze structured tabular data using deterministic logic and local Python computation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Real System Information Metrics
    st.subheader("📊 System & Architecture Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Supported Inputs", value="CSV & Excel", help="File formats handled by the ingestion module.")
    with m2:
        st.metric(label="Pipeline Modules", value="6 Modules", help="Total logical sections in the application.")
    with m3:
        st.metric(label="Execution Model", value="Deterministic", help="Rule-based, explainable algorithmic pipeline.")
    with m4:
        st.metric(label="Compute Layer", value="100% Local", help="Runs locally with zero external API dependencies.")

    st.markdown("---")

    # Core Capabilities Cards
    st.subheader("💡 Core System Capabilities")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="project-card">
            <h4>📁 Tabular Data Ingestion</h4>
            <p>
                Structured parsing of <code>.csv</code> and <code>.xlsx</code> formats with 
                automatic encoding detection and column schema mapping.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="project-card">
            <h4>🧠 Transparent Reasoning</h4>
            <p>
                Clear explanation of query interpretation, data filtering, and calculation steps 
                without black-box decision making.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="project-card">
            <h4>📈 Interactive Analytics</h4>
            <p>
                Dynamic statistical profiling, aggregate calculations, and Plotly visualizations 
                generated straight from your dataset.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Project Workflow & Architecture Roadmap
    st.subheader("🗺️ End-to-End Pipeline Workflow")
    st.markdown("""
    <div class="step-item">
        <div class="step-num">1</div>
        <div class="step-content">
            <strong>Dashboard</strong>
            <p>Central overview of project capabilities, real metrics, and architecture roadmap.</p>
        </div>
    </div>
    <div class="step-item">
        <div class="step-num">2</div>
        <div class="step-content">
            <strong>Dataset Upload</strong>
            <p>Upload CSV or Excel files into session memory with preview and validation checks.</p>
        </div>
    </div>
    <div class="step-item">
        <div class="step-num">3</div>
        <div class="step-content">
            <strong>Dataset Overview</strong>
            <p>Statistical summaries, data types, missing values, and column-level distributions.</p>
        </div>
    </div>
    <div class="step-item">
        <div class="step-num">4</div>
        <div class="step-content">
            <strong>Ask Dataset &amp; Agent Reasoning</strong>
            <p>Natural query input for requesting analysis with real-time step-by-step reasoning trace.</p>
        </div>
    </div>
    <div class="step-item">
        <div class="step-num">5</div>
        <div class="step-content">
            <strong>Data Analysis Engine</strong>
            <p>Computational execution layer utilizing pandas, NumPy, and statistical calculations.</p>
        </div>
    </div>
    <div class="step-item">
        <div class="step-num">6</div>
        <div class="step-content">
            <strong>Results &amp; Insights</strong>
            <p>Direct presentation of computed results and interactive user-selected visual graphs.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    next_button("➡️ Go to Dataset Upload", "Dataset Upload", key="next_dashboard")

# ------------------------------------------------------------------------------
# PAGE 2: DATASET UPLOAD
# ------------------------------------------------------------------------------
elif selected_page == "Dataset Upload":

    st.markdown("""
    <div class="hero-header">
        <span class="badge-pill badge-stage">Dataset Ingestion</span>
        <h1>📁 Dataset Upload</h1>
        <p>Upload a structured CSV or Excel file once. It will be remembered
        across all sections until you reset the session.</p>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # CASE A: Dataset is already loaded — show it without re-upload prompt
    # ------------------------------------------------------------------
    if st.session_state["df"] is not None:
        _df   = st.session_state["df"]
        _fn   = st.session_state["filename"]

        st.success(f"✅ Dataset loaded and ready: **{_fn}**")
        st.markdown("---")

        st.subheader("📊 Active Dataset Summary")
        mi1, mi2, mi3 = st.columns(3)
        with mi1:
            st.metric(label="📄 File Name", value=_fn)
        with mi2:
            st.metric(label="📋 Total Rows", value=f"{_df.shape[0]:,}")
        with mi3:
            st.metric(label="🔢 Total Columns", value=_df.shape[1])

        st.markdown("---")
        st.subheader("👀 Dataset Preview")
        st.caption(f"Showing first {min(100, len(_df))} of {len(_df):,} rows.")
        st.dataframe(_df.head(100), use_container_width=True)

        st.markdown("---")

        # Collapse the uploader so the user is NOT prompted to re-upload by default
        with st.expander("🔄 Replace dataset (upload a different file)", expanded=False):
            st.info("💡 Uploading a new file will replace the current dataset and clear the active question.")
            uploaded_file = st.file_uploader(
                label="Upload a replacement file:",
                type=["csv", "xlsx", "xls"],
                help="Supported formats: CSV (.csv), Excel (.xlsx, .xls)",
                key="replace_uploader"
            )

            if uploaded_file is not None:
                filename = uploaded_file.name
                file_extension = filename.split(".")[-1].lower()
                try:
                    if file_extension == "csv":
                        df = pd.read_csv(uploaded_file)
                    elif file_extension in ["xlsx", "xls"]:
                        df = pd.read_excel(uploaded_file, sheet_name=0)
                    else:
                        st.error("❌ Unsupported file type. Please upload a CSV or Excel file.")
                        st.stop()

                    if df.empty:
                        st.warning("⚠️ The uploaded file is empty.")
                        st.stop()

                    if len(df.columns) == 0:
                        st.warning("⚠️ The uploaded file has no columns. Please check the file format.")
                        st.stop()

                    st.session_state["df"]           = df
                    st.session_state["filename"]      = filename
                    st.session_state["user_question"] = ""
                    st.success(f"✅ '{filename}' loaded successfully! Refreshing…")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Could not read the file. Error: {e}")
                    st.info("💡 Make sure the file is a valid, non-password-protected CSV or Excel file.")

        next_button("➡️ Go to Dataset Overview", "Dataset Overview", key="next_upload")

    # ------------------------------------------------------------------
    # CASE B: No dataset yet — show reference files + own uploader
    # ------------------------------------------------------------------
    else:
        # ── Reference file metadata ───────────────────────────────────
        REFERENCE_FILES = [
            {
                "name":        "📦 Sales Dataset",
                "description": "25 rows · 10 columns\nSales transactions with products, regions, reps, revenue & discounts.",
                "columns":     "Order_ID, Product, Category, Region, Sales_Rep, Units_Sold, Unit_Price, Revenue, Discount, Order_Date",
                "filename":    "sample_sales.csv",
                "key_suffix":  "sales",
                "sample_q":    "What is the total revenue?",
            },
            {
                "name":        "👥 HR Dataset",
                "description": "20 rows · 11 columns\nEmployee records with salaries, departments, experience & performance.",
                "columns":     "Employee_ID, Name, Department, Job_Title, Gender, Age, Experience_Years, Salary, Performance_Score, Joining_Date, City",
                "filename":    "sample_hr.csv",
                "key_suffix":  "hr",
                "sample_q":    "What is the average salary?",
            },
            {
                "name":        "🛒 E-Commerce Dataset",
                "description": "20 rows · 13 columns\nOnline orders with products, customers, categories, amounts & order status.",
                "columns":     "Transaction_ID, Customer_ID, Customer_Name, Product_Name, Category, Quantity, Price, Total_Amount, Payment_Method, Order_Status, Order_Date, Delivery_Date, Country",
                "filename":    "sample_ecommerce.csv",
                "key_suffix":  "ecom",
                "sample_q":    "What is the total amount by category?",
            },
        ]

        # ── Section heading ───────────────────────────────────────────
        st.markdown("""
        <div style="margin-bottom:0.5rem;">
            <h3 style="color:#60a5fa;margin-bottom:0.25rem;">📂 Reference Datasets</h3>
            <p style="color:#94a3b8;font-size:0.9rem;">
                New here? Try one of our built-in sample datasets to explore the full
                pipeline before uploading your own file.
                Click <strong>Preview</strong> to inspect a dataset, or
                <strong>Use This Dataset</strong> to load it directly.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Three reference cards ─────────────────────────────────────
        ref_cols = st.columns(3)
        for idx, ref in enumerate(REFERENCE_FILES):
            with ref_cols[idx]:
                st.markdown(
                    f"""
                    <div style="background:#0f172a;border:1px solid #1e40af;
                         border-radius:12px;padding:1.1rem 1.2rem;margin-bottom:0.6rem;">
                        <h4 style="margin:0 0 0.4rem 0;color:#f1f5f9;font-size:1rem;">{ref['name']}</h4>
                        <p style="margin:0 0 0.5rem 0;color:#94a3b8;font-size:0.78rem;
                           white-space:pre-line;">{ref['description']}</p>
                        <p style="margin:0;color:#475569;font-size:0.7rem;
                           font-family:monospace;line-height:1.5;">{ref['columns']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Read file bytes for download button
                _ref_path = os.path.join(CURRENT_DIR, "assets", ref["filename"])
                with open(_ref_path, "rb") as _f:
                    _ref_bytes = _f.read()

                # Download button (lets user grab the file)
                st.download_button(
                    label="⬇️ Download sample",
                    data=_ref_bytes,
                    file_name=ref["filename"],
                    mime="text/csv",
                    key=f"dl_{ref['key_suffix']}",
                    use_container_width=True,
                )

                # Preview button — shows the data in an expander below
                if st.button(
                    "🔍 Preview dataset",
                    key=f"prev_{ref['key_suffix']}",
                    use_container_width=True,
                ):
                    st.session_state[f"preview_{ref['key_suffix']}"] = True

                # Use-This-Dataset button — loads it directly into session
                if st.button(
                    "✅ Use this dataset",
                    key=f"use_{ref['key_suffix']}",
                    use_container_width=True,
                    type="primary",
                ):
                    _ref_df = pd.read_csv(_ref_path)
                    st.session_state["df"]            = _ref_df
                    st.session_state["filename"]       = ref["filename"]
                    st.session_state["user_question"]  = ref["sample_q"]
                    st.rerun()

        # ── Inline previews (shown when preview button clicked) ───────
        for ref in REFERENCE_FILES:
            if st.session_state.get(f"preview_{ref['key_suffix']}", False):
                st.markdown("---")
                st.markdown(
                    f"<h4 style='color:#60a5fa;'>🔍 Preview: {ref['name']}</h4>",
                    unsafe_allow_html=True,
                )
                _ref_path = os.path.join(CURRENT_DIR, "assets", ref["filename"])
                _prev_df = pd.read_csv(_ref_path)
                pc1, pc2, pc3 = st.columns(3)
                with pc1:
                    st.metric("Rows", f"{_prev_df.shape[0]:,}")
                with pc2:
                    st.metric("Columns", _prev_df.shape[1])
                with pc3:
                    st.metric("File", ref["filename"])
                st.dataframe(_prev_df, use_container_width=True, hide_index=True)
                if st.button(
                    f"✅ Use '{ref['filename']}' now",
                    key=f"use_prev_{ref['key_suffix']}",
                    type="primary",
                ):
                    st.session_state["df"]           = _prev_df
                    st.session_state["filename"]      = ref["filename"]
                    st.session_state["user_question"] = ref["sample_q"]
                    st.rerun()

        # ── Divider before own-file uploader ──────────────────────────
        st.markdown("---")
        st.markdown("""
        <h3 style="color:#60a5fa;margin-bottom:0.25rem;">📤 Upload Your Own File</h3>
        <p style="color:#94a3b8;font-size:0.9rem;margin-bottom:0.8rem;">
            Have your own dataset? Upload a <strong>CSV</strong> or
            <strong>Excel</strong> file below.
        </p>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            label="Choose a file:",
            type=["csv", "xlsx", "xls"],
            help="Supported formats: CSV (.csv), Excel (.xlsx, .xls)",
            key="own_uploader"
        )

        if uploaded_file is not None:
            filename = uploaded_file.name
            file_extension = filename.split(".")[-1].lower()

            try:
                if file_extension == "csv":
                    df = pd.read_csv(uploaded_file)
                elif file_extension in ["xlsx", "xls"]:
                    df = pd.read_excel(uploaded_file, sheet_name=0)
                else:
                    st.error("❌ Unsupported file type. Please upload a CSV or Excel file.")
                    st.stop()

                if df.empty:
                    st.warning("⚠️ The uploaded file is empty. Please upload a file with at least one row of data.")
                    st.stop()

                if len(df.columns) == 0:
                    st.warning("⚠️ The uploaded file has no columns. Please check the file format.")
                    st.stop()

                # Persist in session state for the entire session
                st.session_state["df"]           = df
                st.session_state["filename"]      = filename
                st.session_state["user_question"] = ""

                st.success(f"✅ '{filename}' loaded successfully!")
                st.markdown("---")

                st.subheader("📊 File Summary")
                mi1, mi2, mi3 = st.columns(3)
                with mi1:
                    st.metric(label="📄 File Name", value=filename)
                with mi2:
                    st.metric(label="📋 Total Rows", value=f"{df.shape[0]:,}")
                with mi3:
                    st.metric(label="🔢 Total Columns", value=df.shape[1])

                st.markdown("---")
                st.subheader("👀 Dataset Preview")
                st.caption(f"Showing first {min(100, len(df))} of {len(df):,} rows.")
                st.dataframe(df.head(100), use_container_width=True)

                next_button("➡️ Go to Dataset Overview", "Dataset Overview", key="next_upload")

            except Exception as e:
                st.error(f"❌ Could not read the file. Error: {e}")
                st.info("💡 Make sure the file is a valid, non-password-protected CSV or Excel file.")

# ------------------------------------------------------------------------------
# PAGE 3: DATASET OVERVIEW
# ------------------------------------------------------------------------------
elif selected_page == "Dataset Overview":

    st.markdown("""
    <div class="hero-header">
        <span class="badge-pill badge-stage">Dataset Profiling</span>
        <h1>📋 Dataset Overview</h1>
        <p>A complete structural and statistical profile of your uploaded dataset.
        All values are computed live from your data using pandas and NumPy.</p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # GUARD: No dataset uploaded yet
    # ----------------------------------------------------------------
    # Read the DataFrame from session_state (saved during upload).
    # If no file has been uploaded, guide the user to the Upload page.
    df = st.session_state.get("df", None)
    filename = st.session_state.get("filename", None)

    if df is None:
        st.warning("⚠️ No dataset loaded yet. Please go to **Dataset Upload** and upload a file first.")
        st.stop()   # Stop rendering the rest of this page

    # ==============================================================
    # SECTION 1 — Basic Shape Metrics
    # ==============================================================
    st.subheader("📊 Basic Information")

    b1, b2, b3, b4 = st.columns(4)

    with b1:
        # filename stored in session_state during upload
        st.metric(label="📄 File Name", value=filename)

    with b2:
        # df.shape returns a tuple (rows, columns)
        # df.shape[0] is the number of rows
        st.metric(label="📋 Rows", value=f"{df.shape[0]:,}")

    with b3:
        # df.shape[1] is the number of columns
        st.metric(label="🔢 Columns", value=df.shape[1])

    with b4:
        # df.duplicated() returns a boolean Series: True where a row is
        # an exact copy of a previous row.  .sum() counts the Trues.
        dup_count = int(df.duplicated().sum())
        st.metric(label="🔁 Duplicate Rows", value=f"{dup_count:,}")

    st.markdown("---")

    # ==============================================================
    # SECTION 2 — Column Classification
    # ==============================================================
    # pandas dtype tells us what kind of data each column holds.
    # We split columns into three groups for clear reporting.
    # ----------------------------------------------------------
    # select_dtypes(include=["number"]) keeps int64 and float64 columns.
    # select_dtypes(include=["object", "category"]) keeps text columns.
    # select_dtypes(include=["datetime"]) keeps datetime64 columns.
    # ----------------------------------------------------------

    numerical_cols   = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    datetime_cols    = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    # Extra step: try to detect datetime columns stored as plain strings.
    # pd.to_datetime(..., errors='coerce') converts what it can; if most
    # values parse successfully, we count that column as a datetime.
    for col in categorical_cols[:]:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() > 0.8:   # >80 % parsed → treat as datetime
                datetime_cols.append(col)
                categorical_cols.remove(col)
        except Exception:
            pass

    st.subheader("🏷️ Column Classification")
    cc1, cc2, cc3 = st.columns(3)

    with cc1:
        st.markdown("""
        <div class="project-card">
            <h4>🔢 Numerical Columns</h4>
        """, unsafe_allow_html=True)
        if numerical_cols:
            for c in numerical_cols:
                st.markdown(f"`{c}`")
        else:
            st.markdown("_None detected_")
        st.markdown("</div>", unsafe_allow_html=True)

    with cc2:
        st.markdown("""
        <div class="project-card">
            <h4>📝 Categorical Columns</h4>
        """, unsafe_allow_html=True)
        if categorical_cols:
            for c in categorical_cols:
                st.markdown(f"`{c}`")
        else:
            st.markdown("_None detected_")
        st.markdown("</div>", unsafe_allow_html=True)

    with cc3:
        st.markdown("""
        <div class="project-card">
            <h4>📅 Date / Time Columns</h4>
        """, unsafe_allow_html=True)
        if datetime_cols:
            for c in datetime_cols:
                st.markdown(f"`{c}`")
        else:
            st.markdown("_None detected_")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ==============================================================
    # SECTION 3 — Column Names & Data Types
    # ==============================================================
    # df.dtypes returns a Series: index = column names, values = dtype.
    # We convert it to a DataFrame so st.dataframe() can display it
    # as a clean, styled table.

    st.subheader("📜 Column Names & Data Types")
    dtype_df = pd.DataFrame({
        "Column Name" : df.columns.tolist(),
        "Data Type"   : [str(dt) for dt in df.dtypes.tolist()]
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ==============================================================
    # SECTION 4 — Missing Value Analysis
    # ==============================================================
    # df.isnull() returns a boolean DataFrame: True where a value is NaN.
    # .sum() counts missing values per column.
    # We then compute the percentage: (missing / total rows) * 100.

    st.subheader("❌ Missing Value Report")

    missing_count   = df.isnull().sum()                    # Count per column
    missing_pct     = (missing_count / len(df) * 100).round(2)  # % per column
    missing_df      = pd.DataFrame({
        "Column Name"    : df.columns.tolist(),
        "Missing Count"  : missing_count.values,
        "Missing %"      : missing_pct.values
    })

    # Overall missing metric
    total_missing = int(missing_count.sum())
    total_cells   = df.shape[0] * df.shape[1]             # rows × columns

    mv1, mv2, mv3 = st.columns(3)
    with mv1:
        st.metric("Total Missing Cells",  f"{total_missing:,}")
    with mv2:
        st.metric("Total Cells",          f"{total_cells:,}")
    with mv3:
        overall_pct = round(total_missing / total_cells * 100, 2) if total_cells else 0
        st.metric("Overall Missing %",    f"{overall_pct} %")

    # Highlight columns that have any missing values
    cols_with_missing = missing_df[missing_df["Missing Count"] > 0]
    if cols_with_missing.empty:
        st.success("✅ No missing values found in any column.")
    else:
        st.warning(f"⚠️ {len(cols_with_missing)} column(s) contain missing values.")

    st.dataframe(missing_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ==============================================================
    # SECTION 5 — Dataset Preview
    # ==============================================================
    st.subheader("👀 Dataset Preview")
    preview_rows = min(50, len(df))
    st.caption(f"Showing first {preview_rows} of {len(df):,} rows.")
    st.dataframe(df.head(preview_rows), use_container_width=True)

    st.markdown("---")

    # ==============================================================
    # SECTION 6 — Data Cleaning
    # ==============================================================
    st.subheader("🧹 Data Cleaning")
    st.markdown("""
    <div class="cleaning-info-card">
        <p>Automatically cleans the dataset by removing duplicate rows and filling missing values.
        Numerical columns are filled with the <strong>column median</strong>; 
        categorical columns are filled with the <strong>most frequent value (mode)</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    col_clean1, col_clean2 = st.columns(2)
    with col_clean1:
        remove_dups = st.checkbox("🔁 Remove duplicate rows", value=True, key="clean_dups")
    with col_clean2:
        fill_missing = st.checkbox("❓ Fill missing values", value=True, key="clean_missing")

    if st.button("✨ Clean Dataset", key="btn_clean", type="primary", use_container_width=False):
        cleaned_df = df.copy()

        rows_before = len(cleaned_df)
        dups_removed = 0
        cells_filled = 0

        if remove_dups:
            cleaned_df = cleaned_df.drop_duplicates()
            dups_removed = rows_before - len(cleaned_df)

        if fill_missing:
            for col in cleaned_df.columns:
                if cleaned_df[col].isnull().sum() > 0:
                    if cleaned_df[col].dtype in ["float64", "int64", "float32", "int32"]:
                        fill_val = cleaned_df[col].median()
                    else:
                        mode_vals = cleaned_df[col].mode()
                        fill_val = mode_vals[0] if not mode_vals.empty else "Unknown"
                    cells_filled += cleaned_df[col].isnull().sum()
                    cleaned_df[col] = cleaned_df[col].fillna(fill_val)

        st.session_state["cleaned_df"] = cleaned_df

        # Summary metrics
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Rows Removed (Duplicates)", f"{dups_removed:,}")
        with mc2:
            st.metric("Cells Filled (Missing)", f"{cells_filled:,}")
        with mc3:
            st.metric("Cleaned Dataset Rows", f"{len(cleaned_df):,}")

    if "cleaned_df" in st.session_state and st.session_state["cleaned_df"] is not None:
        cleaned_df = st.session_state["cleaned_df"]
        st.markdown("---")
        st.subheader("✅ Cleaned Dataset Preview")
        st.caption(f"Showing first {min(50, len(cleaned_df))} of {len(cleaned_df):,} rows — after cleaning.")
        st.dataframe(cleaned_df.head(50), use_container_width=True)

        st.markdown("")
        # Download cleaned dataset
        csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
        original_name = (filename or "dataset").rsplit(".", 1)[0]
        st.download_button(
            label="⬇️ Download Cleaned Dataset (CSV)",
            data=csv_bytes,
            file_name=f"{original_name}_cleaned.csv",
            mime="text/csv",
            use_container_width=False,
            key="download_cleaned"
        )

    next_button("➡️ Ask a Question", "Ask Dataset & Agent Reasoning", key="next_overview")

# ------------------------------------------------------------------------------
# PAGE 4: ASK DATASET & AGENT REASONING
# ------------------------------------------------------------------------------
elif selected_page == "Ask Dataset & Agent Reasoning":

    st.markdown("""
    <div class="hero-header">
        <span class="badge-pill badge-stage">Natural Language &amp; Reasoning</span>
        <h1>💬 Ask Dataset &amp; Agent Reasoning</h1>
        <p>Type your question in plain English. The agent formulates an explainable, 
        step-by-step reasoning trace showing exactly how it will analyze the data.</p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # GUARD: No dataset uploaded yet
    # ----------------------------------------------------------------
    df = st.session_state.get("df", None)
    filename = st.session_state.get("filename", None)

    if df is None:
        st.warning("⚠️ No dataset loaded. Please go to **Dataset Upload** first.")
        st.stop()

    st.info(f"ℹ️ Active dataset: **{filename}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

    with st.expander("🔍 Available Column Names (click to expand)", expanded=False):
        cols_display = "  |  ".join([f"`{c}`" for c in df.columns.tolist()])
        st.markdown(cols_display)

    st.markdown("---")

    # Example question buttons
    st.caption("Click an example or type your question below:")
    ex_c1, ex_c2, ex_c3 = st.columns(3)
    with ex_c1:
        if st.button("Average salary", use_container_width=True):
            st.session_state["user_question"] = "What is the average salary?"
            st.rerun()
    with ex_c2:
        if st.button("Highest revenue product", use_container_width=True):
            st.session_state["user_question"] = "Which product has the highest revenue?"
            st.rerun()
    with ex_c3:
        if st.button("Total sales", use_container_width=True):
            st.session_state["user_question"] = "What is the total sales?"
            st.rerun()

    # Question Input
    current_q = st.session_state.get("user_question", "")
    question = st.text_input(
        label="⌨️ Enter your question:",
        value=current_q,
        placeholder="e.g. Which product has the highest revenue?",
        help="Type your question once. Downstream modules will automatically analyze it."
    )

    if question.strip() != current_q:
        st.session_state["user_question"] = question.strip()

    st.markdown("---")

    # When question is provided: run NLU & Reasoning, and show ONLY Step-by-Step Reasoning Trace
    if question.strip():
        nlu_result = parse_question(question, df.columns.tolist())
        plan = build_plan(nlu_result, df.columns.tolist())

        st.subheader("Step-by-Step Reasoning Trace")
        st.caption("These are the exact logical steps the agent will follow to answer your question:")

        for i, step_text in enumerate(plan.get("steps", [])):
            icon = "🟢" if i == 0 else ("🏁" if i == len(plan.get("steps", [])) - 1 else "🔵")
            import re
            cleaned_step = re.sub(r"^Step\s*\d+:\s*", "", step_text)
            st.markdown(
                f"""
                <div class="step-item">
                    <div class="step-num">{i + 1}</div>
                    <div class="step-content">
                        <p>{icon} {cleaned_step}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown("""
        <div class="placeholder-box">
            <h3>Waiting for a Question</h3>
            <p>Type a question in the box above or click an example. The agent will formulate a step-by-step reasoning trace.</p>
        </div>
        """, unsafe_allow_html=True)

    # Next button — only when a question has been entered
    if st.session_state.get("user_question", ""):
        next_button("➡️ Run Data Analysis", "Data Analysis Engine", key="next_ask")

# ------------------------------------------------------------------------------
# PAGE 5: DATA ANALYSIS ENGINE
# ------------------------------------------------------------------------------
elif selected_page == "Data Analysis Engine":

    st.markdown("""
    <div class="hero-header">
        <span class="badge-pill badge-stage">Computational Execution</span>
        <h1>⚙️ Data Analysis Engine</h1>
        <p>Computes the exact mathematical result from your uploaded dataset
        using deterministic pandas operations. No external APIs or hardcoded values.</p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # GUARD: No dataset
    # ----------------------------------------------------------------
    df       = st.session_state.get("df", None)
    filename = st.session_state.get("filename", None)

    if df is None:
        st.warning("⚠️ No dataset loaded. Please go to **Dataset Upload** first.")
        st.stop()

    st.info(f"ℹ️ Active dataset: **{filename}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

    with st.expander("🔍 Available Column Names", expanded=False):
        st.markdown("  |  ".join([f"`{c}`" for c in df.columns.tolist()]))

    st.markdown("---")

    question = st.session_state.get("user_question", "").strip()

    if not question:
        st.markdown("""
        <div class="placeholder-box">
            <h3>No Question Asked Yet</h3>
            <p>Enter a question below or in the <strong>Ask Dataset &amp; Agent Reasoning</strong> section.</p>
        </div>
        """, unsafe_allow_html=True)
        new_q = st.text_input(
            label="⌨️ Enter your question:",
            placeholder="e.g. Which product has the highest revenue?",
            help="Type your question to execute analysis."
        )
        if new_q.strip():
            st.session_state["user_question"] = new_q.strip()
            st.rerun()
    else:
        # Show active question card with an edit expander so user never needs to retype
        st.markdown(
            f"""
            <div style="background:#0f172a;border-left:4px solid #3b82f6;padding:1rem 1.4rem;border-radius:10px;margin-bottom:1rem;">
                <p style="margin:0;color:#94a3b8;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;">Active Question</p>
                <h4 style="margin:0.25rem 0 0 0;color:#f1f5f9;font-size:1.15rem;font-weight:600;">{question}</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("✏️ Ask a different question (optional)", expanded=False):
            change_q = st.text_input("Enter new question:", value=question, key="engine_change_q")
            if st.button("Update Question", key="btn_update_engine_q"):
                if change_q.strip():
                    st.session_state["user_question"] = change_q.strip()
                    st.rerun()

        # Run pipeline
        nlu_result = parse_question(question, df.columns.tolist())
        plan = build_plan(nlu_result, df.columns.tolist())
        result = execute_plan(df, plan)

        # Full Pipeline Summary is removed

        # ----------------------------------------------------------------
        # DISPLAY: Execution Result
        # ----------------------------------------------------------------
        st.subheader("📊 Calculation Result")

        if result["status"] == "error":
            st.error(f"❌ Execution Error: {result.get('error_message', 'Unknown error')}")
            st.info(
                "💡 **Tips:** Make sure your question uses column names visible "
                "in the 'Available Column Names' expander above."
            )
        else:
            # ---- Scalar Answer Banner ----
            if result.get("result_value") is not None:
                op_label = (result.get("operation") or "").upper()
                col_label = result.get("target_column") or ""
                rv = result['result_value']
                rv_display = f"{int(rv):,}" if isinstance(rv, float) and rv == int(rv) else f"{rv:,}" if isinstance(rv, int) else f"{rv:,.2f}" if isinstance(rv, float) else str(rv)
                st.markdown(
                    f"""
                    <div style="background:linear-gradient(135deg,#1e3a8a,#2563eb);
                    color:#fff;padding:1.5rem 2rem;border-radius:12px;
                    margin-bottom:1.2rem;">
                        <p style="margin:0 0 0.4rem 0;font-size:0.9rem;color:#bfdbfe;">
                            {op_label} of {col_label}
                        </p>
                        <h2 style="margin:0;font-size:2.4rem;font-weight:700;">
                            {rv_display}
                        </h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ---- Result Table ----
            if result.get("result_df") is not None and not result["result_df"].empty:
                rdf = result["result_df"]
                st.markdown(f"📋 Result table — **{len(rdf):,} row(s)**")
                st.dataframe(rdf, use_container_width=True, hide_index=True)
            elif result.get("result_value") is None:
                st.info("ℹ️ No tabular result was produced for this query.")

        # ----------------------------------------------------------------
        # DISPLAY: Calculation Notes (what pandas actually did)
        # ----------------------------------------------------------------
        if result.get("notes"):
            st.markdown("---")
            st.subheader("🗒️ Calculation Notes")
            for note in result["notes"]:
                st.markdown(f"✅ {note}")

        # ----------------------------------------------------------------
        # DISPLAY: Plan Warnings
        # ----------------------------------------------------------------
        if plan.get("warnings"):
            st.markdown("---")
            st.subheader("⚠️ Plan Warnings")
            for w in plan["warnings"]:
                st.warning(w)

    next_button("➡️ View Results & Insights", "Results & Insights", key="next_analysis")

# ------------------------------------------------------------------------------
# PAGE 6: RESULTS & INSIGHTS
# ------------------------------------------------------------------------------
elif selected_page == "Results & Insights":

    st.markdown("""
    <div class="hero-header">
        <span class="badge-pill badge-stage">Visual Analytics &amp; Results</span>
        <h1>📈 Results &amp; Insights</h1>
        <p>Direct computed results and interactive visual graphs for your question.</p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # GUARD: No dataset
    # ----------------------------------------------------------------
    df       = st.session_state.get("df", None)
    filename = st.session_state.get("filename", None)

    if df is None:
        st.warning("⚠️ No dataset loaded. Please go to **Dataset Upload** first.")
        st.stop()

    st.info(f"ℹ️ Active dataset: **{filename}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

    st.markdown("---")

    question = st.session_state.get("user_question", "").strip()

    if not question:
        st.markdown("""
        <div class="placeholder-box">
            <h3>No Question Asked Yet</h3>
            <p>Enter a question below or in the <strong>Ask Dataset &amp; Agent Reasoning</strong> section.</p>
        </div>
        """, unsafe_allow_html=True)
        new_q = st.text_input(
            label="⌨️ Enter your question:",
            placeholder="e.g. Which product has the highest revenue?",
            help="Type your question to view results and graph."
        )
        if new_q.strip():
            st.session_state["user_question"] = new_q.strip()
            st.rerun()
    else:
        # Show active question with optional change toggle
        st.markdown(
            f"""
            <div style="background:#0f172a;border-left:4px solid #3b82f6;padding:1rem 1.4rem;border-radius:10px;margin-bottom:1rem;">
                <p style="margin:0;color:#94a3b8;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;">Active Question</p>
                <h4 style="margin:0.25rem 0 0 0;color:#f1f5f9;font-size:1.15rem;font-weight:600;">{question}</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("✏️ Ask a different question (optional)", expanded=False):
            change_q = st.text_input("Enter new question:", value=question, key="results_change_q")
            if st.button("Update Question", key="btn_update_results_q"):
                if change_q.strip():
                    st.session_state["user_question"] = change_q.strip()
                    st.rerun()

        # Run pipeline
        nlu_result = parse_question(question, df.columns.tolist())
        plan       = build_plan(nlu_result, df.columns.tolist())
        result     = execute_plan(df, plan)
        final_answer, _, _ = generate_answer_and_insights(result, plan)

        # -------------------------------------------------------------
        # ONLY RESULT AND GRAPH
        # -------------------------------------------------------------
        st.subheader("💡 Result")

        if result["status"] == "error":
            st.error(f"❌ {final_answer}")
        else:
            answer_clean = final_answer.replace("**", "")
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#1e3a8a,#2563eb);
                     color:#fff;padding:1.4rem 1.8rem;border-radius:12px;
                     margin-bottom:1.2rem;">
                    <p style="margin:0 0 0.4rem 0;color:#bfdbfe;
                       font-size:0.8rem;text-transform:uppercase;
                       letter-spacing:0.06em;">Answer</p>
                    <h3 style="margin:0;font-size:1.35rem;font-weight:600;
                       line-height:1.4;">{answer_clean}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Scalar metric if applicable
            if result.get("result_value") is not None:
                op_label  = (result.get("operation") or "").upper()
                col_label = result.get("target_column") or ""
                rv = result['result_value']
                rv_display = f"{int(rv):,}" if isinstance(rv, float) and rv == int(rv) else f"{rv:,}" if isinstance(rv, int) else f"{rv:,.2f}" if isinstance(rv, float) else str(rv)
                st.markdown(
                    f"""
                    <div style="display:inline-block;background:#0f172a;border:1px solid #334155;
                         border-radius:10px;padding:1rem 1.6rem;margin-bottom:1.2rem;min-width:240px;">
                        <p style="margin:0 0 0.2rem 0;color:#64748b;
                           font-size:0.75rem;text-transform:uppercase;
                           letter-spacing:0.06em;">{op_label} of {col_label}</p>
                        <h2 style="margin:0;font-size:2.2rem;font-weight:800;
                           color:#38bdf8;">{rv_display}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Tabular result if applicable
            if result.get("result_df") is not None and not result["result_df"].empty:
                rdf = result["result_df"]
                st.markdown(f"**Tabular Result ({len(rdf):,} rows)**")
                st.dataframe(rdf, use_container_width=True, hide_index=True)

            # Visual Graph with user graph selection
            st.markdown("---")
            st.subheader("📊 Graph")

            c_select_col, _ = st.columns([2, 2])
            with c_select_col:
                chart_type = st.selectbox(
                    "Select Graph Type:",
                    options=["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Area Chart", "Histogram", "Box Plot"],
                    index=0
                )

            try:
                fig = generate_chart(result, df, chart_type=chart_type)
            except TypeError:
                importlib.reload(visualisation_engine)
                fig = visualisation_engine.generate_chart(result, df, chart_type=chart_type)

            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ A graph could not be generated for this query with the selected type.")

    # ------------------------------------------------------------------
    # RESET BUTTON — always visible at the bottom of Results & Insights
    # Clears the entire session and sends the user back to Dataset Upload
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 0.5rem 0;">
        <p style="color:#94a3b8;font-size:0.9rem;margin-bottom:0.75rem;">
            Want to start over with a different dataset or question?
        </p>
    </div>
    """, unsafe_allow_html=True)

    _r_col1, _r_col2, _r_col3 = st.columns([2, 2, 2])
    with _r_col2:
        if st.button("🔄 Reset & Upload New Dataset", use_container_width=True, type="primary", key="reset_btn_results"):
            # Clear all session data and navigate back to Dataset Upload
            st.session_state["df"]           = None
            st.session_state["filename"]      = None
            st.session_state["user_question"] = ""
            st.session_state["current_page"]  = "Dataset Upload"
            st.rerun()
