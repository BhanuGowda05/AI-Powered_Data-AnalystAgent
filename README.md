# Data analyst agent 📊

> An autonomous, explainable data analysis assistant built entirely with Python and Streamlit — no external LLMs, no paid APIs, no internet required at runtime.

---

## 1. Project Title

**Data analyst agent** — Modular, rule-based NLP pipeline for structured tabular data analysis.

---

## 2. Project Description

The **Data analyst agent** is a full-stack, multi-page Streamlit web application that enables users to upload structured datasets (CSV or Excel) and ask plain-English questions about their data. The system interprets the question using a custom keyword-based NLU engine, builds a transparent analysis plan, executes the calculations using pandas, generates interactive Plotly charts, and returns a final natural-language answer — all without any external AI APIs, cloud services, or LLMs.

Every number shown to the user is computed directly from their dataset using pandas. Nothing is invented or hallucinated.

---

## 3. Features

| Feature | Description |
|:--------|:------------|
| 📁 Multi-format Upload | Supports `.csv`, `.xlsx`, `.xls` file formats |
| 📋 Dataset Overview | Auto-profiling: shape, data types, missing values, duplicates, statistics |
| 💬 Natural Language Queries | Ask questions in plain English about your data |
| 🧠 Transparent Reasoning | Step-by-step analysis plan shown before any calculation |
| ⚙️ Pandas Execution Engine | Mean, sum, max, min, count, median, groupby, value_counts |
| 📈 Interactive Charts | Plotly bar charts and histograms auto-generated from results |
| 💡 Final Answer + Insights | Templated English answers derived 100% from actual data |
| 🔒 Fully Local | No external APIs, no data leaves your machine |
| 🎨 Dark Theme UI | Professional glassmorphism-style dark interface |
| ✅ Error Handling | Graceful messages for bad files, unknown columns, unsupported queries |

---

## 4. Technology Stack

| Layer | Technology |
|:------|:-----------|
| **Frontend / UI** | Streamlit, Custom CSS (`assets/style.css`) |
| **NLU Engine** | Pure Python — keyword matching, regex tokenisation |
| **Data Processing** | pandas ≥ 2.0, NumPy ≥ 1.26 |
| **Visualisation** | Plotly Express |
| **File Parsing** | openpyxl (`.xlsx`), xlrd (`.xls`) |
| **Language** | Python 3.10+ |
| **Deployment** | Streamlit Community Cloud |

> ⚠️ No LangChain, Groq, OpenAI, or any external LLM is used at any stage.

---

## 5. Project Structure

```
AI_Data_Analyst_Agent/
│
├── app.py                    # Main Streamlit app — 7-page navigation + pipeline routing
├── question_parser.py        # Stage 5: NLU — keyword detection + column matching
├── reasoning_engine.py       # Stage 6: Reasoning — builds structured analysis plan
├── analysis_engine.py        # Stage 7: Execution — pandas calculations
├── visualisation_engine.py   # Stage 8: Chart generation using Plotly
├── insights_engine.py        # Stage 9: Final English answer + key insights
│
├── assets/
│   └── style.css             # Dark theme CSS — cards, metrics, hero banner, sidebar
│
├── requirements.txt          # All Python dependencies with version constraints
├── test_stage10_runner.py    # Automated test suite — 25 tests across all modules
└── README.md                 # This file
```

---

## 6. Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Step-by-step

```bash
# 1. Clone or download the project
git clone https://github.com/<your-username>/AI_Data_Analyst_Agent.git
cd AI_Data_Analyst_Agent

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## 7. How to Run

```bash
streamlit run app.py
```

The app will automatically open in your default browser at:
```
http://localhost:8501
```

To run the automated test suite:
```bash
python test_stage10_runner.py
```
Expected output: `TOTAL: 25  PASSED: 25  FAILED: 0`

---

## 8. How to Use

### Step 1 — Upload Your Dataset
1. Navigate to **Dataset Upload** in the sidebar.
2. Click **Browse files** and select a `.csv` or `.xlsx` file.
3. The system will validate and load the data into session memory.

### Step 2 — Explore Your Data
1. Go to **Dataset Overview** to see:
   - Row/column count, duplicate rows
   - Column types (numerical, categorical, datetime)
   - Missing value report
   - Statistical summary (mean, std, min, max, quartiles)

### Step 3 — Ask a Question
1. Go to **Ask About Your Dataset**, **Agent Reasoning**, **Data Analysis Engine**, or **Results & Insights**.
2. Type a plain-English question, e.g.:
   - `"What is the average salary?"`
   - `"Which product has the highest revenue?"`
   - `"What is the total sales?"`
   - `"Which department has the highest average salary?"`
   - `"How many customers are there?"`
3. The system will interpret, plan, calculate, and display the answer.

### Tips for Best Results
- Use words that appear in your column names.
- Include an operation keyword: **average, highest, lowest, total, count, sum, list**.
- For groupby queries, include both a categorical and a numerical column name.

---

## 9. System Workflow

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5 — question_parser.py  (NLU Engine)                    │
│  • Tokenises the question                                       │
│  • Detects operation keyword (average, highest, count…)         │
│  • Matches column names using word overlap + CamelCase split    │
│  • Assigns roles: numerical col = target, categorical = group   │
│  • Returns: operation, target_column, group_column, confidence  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 6 — reasoning_engine.py  (Plan Builder)                 │
│  • Maps operation → pandas aggregation function                 │
│  • Determines analysis type:                                    │
│      single_aggregate / groupby_aggregate /                     │
│      value_counts / full_summary                                │
│  • Sets sort direction (ascending/descending) and limit (top N) │
│  • Builds numbered reasoning steps (shown to user)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 7 — analysis_engine.py  (Pandas Execution)              │
│  • Validates column existence                                   │
│  • Coerces to numeric (handles "1,000" or "$500" strings)       │
│  • Drops null rows appropriately                                │
│  • Executes: .mean() / .sum() / .max() / .min() /               │
│              .count() / .groupby().agg() / .value_counts()      │
│  • Returns: result_value (scalar) + result_df (table)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 8 — visualisation_engine.py  (Plotly Charts)            │
│  • groupby_aggregate → Plotly bar chart (colour by value)       │
│  • value_counts → bar chart or histogram (if numeric + >10 uniq)│
│  • single_aggregate / full_summary → no chart (metric display)  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 9 — insights_engine.py  (Natural Language Answer)       │
│  • Fills English sentence templates with actual computed values  │
│  • Computes supporting stats: avg, max, min, std, count         │
│  • Returns: final_answer (str) + insights (list) + stats (list) │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
              Streamlit UI (app.py) renders:
              • Confidence badge
              • Reasoning steps
              • Final answer card
              • Big-number metric
              • Supporting statistics
              • Plotly chart + data table
              • Key insights bullets
```

---

## 10. Future Enhancements

| Enhancement | Description |
|:------------|:------------|
| 🔍 Filter Support | Handle filter conditions like "WHERE salary > 50000" |
| 📅 Date/Time Queries | Time-series aggregations (monthly averages, trends) |
| 📤 Export Results | Download answers as PDF, CSV, or Excel |
| 🗄️ Multi-Sheet Excel | Let user choose which sheet to load |
| 📊 More Chart Types | Pie charts, scatter plots, line charts |
| 🧩 Plugin Architecture | Add new analysis types without changing core code |
| 🌐 Multi-language UI | Support for non-English column names |
| 🔐 File Password Support | Handle password-protected Excel files |

---

## Deployment on Streamlit Community Cloud

### Prerequisites
- A [GitHub](https://github.com) account
- A [Streamlit Community Cloud](https://streamlit.io/cloud) account (free)

### Step 1 — Push to GitHub
```bash
# Initialise git (if not already done)
git init
git add .
git commit -m "Stage 10: Tested and deployment-ready"

# Create a new repo on GitHub, then push:
git remote add origin https://github.com/<your-username>/AI_Data_Analyst_Agent.git
git branch -M main
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
2. Click **"New app"**.
3. Select your GitHub repository.
4. Set **Main file path** to: `app.py`
5. Click **"Deploy!"**

Streamlit Cloud will automatically:
- Install all packages from `requirements.txt`
- Launch `app.py` as the entry point
- Provide a public URL like `https://<your-username>-ai-data-analyst-agent.streamlit.app`

> **Note:** No secrets or environment variables are required — this app is fully self-contained with no external API keys.

---

## Project Architecture & Data Flow (Viva Guide)

### Q: What is the overall architecture of this project?

The project follows a **Modular Pipeline Architecture** with strict **Separation of Concerns**. The system is divided into 5 independent Python modules that form a sequential pipeline:

```
question_parser.py → reasoning_engine.py → analysis_engine.py
                                                    ↓
                        app.py ← insights_engine.py ← visualisation_engine.py
```

Each module has a single responsibility and communicates via Python dictionaries (structured data contracts), not function calls with many arguments.

---

### Q: How does the NLU engine work without an LLM?

The `question_parser.py` module uses a **rule-based keyword matching** approach:

1. **Tokenisation**: The question is lowercased and stripped of punctuation, then split into word tokens.
2. **Operation Detection**: A dictionary maps words like `"average" → "average"`, `"highest" → "maximum"`, `"total" → "sum"` to canonical operation names.
3. **Column Matching**: Each DataFrame column name is also tokenised (and CamelCase-split, e.g. `CustomerID → customer, id`). If any token from the question overlaps with column tokens (including plural stemming), that column is matched.
4. **Role Assignment**: Matched columns are classified by heuristic: columns with names like "salary", "revenue" are **target** (the thing being measured); columns with names like "product", "department" are **group** (the dimension to group by).
5. **Confidence Scoring**: A score of HIGH/MEDIUM/LOW is assigned based on how many of {operation, target_column, group_column} were successfully identified.

---

### Q: Why is reasoning separated from calculation?

This follows the **Separation of Concerns** principle — a fundamental software engineering concept:

| Module | Role | Benefit |
|:-------|:-----|:--------|
| `reasoning_engine.py` | WHAT to do (the plan) | Explainability — user sees the plan before numbers appear |
| `analysis_engine.py` | HOW to do it (execution) | Modularity — execution can be changed without touching reasoning |

This mirrors how enterprise AI agent frameworks (LangChain, AutoGPT) separate intent classification from tool execution.

---

### Q: What pandas operations does the system support?

| Analysis Type | Example Question | pandas Code |
|:-------------|:-----------------|:------------|
| `single_aggregate` | "What is the average salary?" | `df["Salary"].mean()` |
| `groupby_aggregate` | "Which product has highest revenue?" | `df.groupby("Product")["Revenue"].sum().sort_values()` |
| `value_counts` | "How many orders per category?" | `df["Category"].value_counts()` |
| `full_summary` | "Show me all employees" | `df[["Employee"]]` |

---

### Q: How are errors handled?

The system has three layers of error handling:
1. **Upload layer**: Validates file type, checks for empty files and missing columns.
2. **NLU layer**: Returns LOW confidence and explains what it couldn't detect.
3. **Execution layer**: Every function returns a standard dict with `status: "error"` and an `error_message` — never crashes with an unhandled exception.

---

### Q: Is this deployment-ready? Why?

Yes. The system:
- Has **no external API dependencies** (no API keys, no internet at runtime)
- Has **all dependencies listed** in `requirements.txt` with version constraints
- Has a **25-test automated suite** with 100% pass rate
- Has **graceful error handling** at every stage
- Runs on **Streamlit Community Cloud** with zero configuration
