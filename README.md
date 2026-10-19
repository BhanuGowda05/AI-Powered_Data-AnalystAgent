# 🤖 AI-Powered Data Analyst Agent

 An autonomous, explainable data analysis assistant built entirely with Python and Streamlit — no external LLMs, no paid APIs, no internet required at runtime.
An AI-powered data analysis application designed to simplify the process of analyzing datasets. The system allows users to upload a dataset and automatically performs data understanding, cleaning, exploratory data analysis, visualization, and generates meaningful insights
---

## 📌 Overview

The **AI-Powered Data Analyst Agent** allows users to upload a CSV or Excel dataset and instantly get:
- Automated data understanding and cleaning
- Exploratory data analysis with statistics
- Interactive visualizations (Plotly charts)
- Plain-English answers to natural language questions about the data

Users simply upload a dataset, ask questions like *"Which product has the highest revenue?"*, and the system interprets, plans, calculates, and answers — all using deterministic Python logic with zero external AI APIs.

---

## ✨ Key Features

| Feature | Description |
|:--------|:------------|
| 📂 Multi-format Upload | Supports `.csv`, `.xlsx`, `.xls` file formats |
| 📋 Dataset Overview | Auto-profiling: shape, types, missing values, duplicates |
| 💬 Natural Language Queries | Ask questions in plain English about your data |
| 🧠 Transparent Reasoning | Step-by-step analysis plan shown before any calculation |
| ⚙️ Pandas Execution Engine | Mean, sum, max, min, count, median, groupby, value_counts |
| 📈 Interactive Charts | Plotly charts auto-generated from results |
| 💡 AI-Powered Insights | English answers derived 100% from actual data |
| 🧹 Data Cleaning | Remove duplicates, fill missing values automatically |
| 🔒 Fully Local | No external APIs — data never leaves your machine |
| 🎨 Dark Theme UI | Professional glassmorphism-style dark interface |

---

## 🛠️ Technology Stack

| Layer | Technology |
|:------|:-----------|
| **Frontend / UI** | Streamlit + Custom CSS |
| **NLU Engine** | Pure Python — keyword matching, regex tokenisation |
| **Data Processing** | pandas , NumPy  |
| **Visualisation** | Plotly Express |
| **File Parsing** | openpyxl (`.xlsx`), xlrd (`.xls`) |
| **Language** | Python 3.10+ |
| **Deployment** | Streamlit Community Cloud |

> ⚠️ No LangChain, OpenAI, Groq, or any external LLM is used at any stage.

---

## 🗂️ Project Structure

```
AI_Data_Analyst_Agent/
│
├── app.py                    # Main Streamlit app — multi-page navigation
├── question_parser.py        # NLU — keyword detection + column matching
├── reasoning_engine.py       # Reasoning — builds structured analysis plan
├── analysis_engine.py        # Execution — pandas calculations
├── visualisation_engine.py   # Chart generation using Plotly
├── insights_engine.py        # Final English answer + key insights
│
├── assets/
│   ├── style.css             # Dark theme CSS
│   ├── sample_sales.csv      # Built-in sample dataset
│   ├── sample_hr.csv         # Built-in sample dataset
│   └── sample_ecommerce.csv  # Built-in sample dataset
│
├── .streamlit/config.toml    # Streamlit dark theme config
├── requirements.txt          # Python dependencies
├── test_stage10_runner.py    # Automated test suite (25 tests)
└── README.md
```

---

## ⚙️ How It Works

```
User uploads dataset
        ↓
  Dataset Understanding     → rows, columns, data types, statistics
        ↓
  Data Quality Check        → missing values, duplicate records
        ↓
  Data Cleaning             → fill missing, remove duplicates
        ↓
  Ask a Question (NLU)      → keyword detection + column matching
        ↓
  Reasoning Engine          → builds step-by-step analysis plan
        ↓
  Analysis Engine           → pandas calculations (mean/sum/max/groupby…)
        ↓
  Visualisation Engine      → Plotly charts
        ↓
  Insights Engine           → plain-English answer + key observations
        ↓
  Results Dashboard         → answer card + chart + data table
```

---

## 🚀 Installation & Running

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/bhanushreeg-gowda/AI-Powered_Data-AnalystAgent.git
cd AI-Powered_Data-AnalystAgent

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

App opens at: **http://localhost:8501**

---

## 💬 How to Use

**Step 1 — Upload Dataset**
- Go to **Dataset Upload** in the sidebar
- Upload a `.csv` or `.xlsx` file (or use a built-in sample dataset)

**Step 2 — Explore Your Data**
- Go to **Dataset Overview** to see column types, missing values, and statistics
- Use **Data Cleaning** to remove duplicates and fill missing values

**Step 3 — Ask a Question**
- Go to **Ask Dataset & Agent Reasoning**
- Type a plain-English question, for example:
  - *"What is the average salary?"*
  - *"Which product has the highest revenue?"*
  - *"How many employees are in each department?"*
  - *"What is the total sales?"*

**Step 4 — View Results**
- Go to **Results & Insights** to see the answer, chart, and key observations

> 💡 **Tip:** Use words from your column names and include an operation keyword like `average`, `highest`, `total`, `count`, or `show`.

---

## 🧪 Automated Tests

```bash
python test_stage10_runner.py
```
Expected: `TOTAL: 25  PASSED: 25  FAILED: 0`

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub (already done ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select this repository
4. Set **Main file path** → `app.py`
5. Click **Deploy!**

> No secrets or environment variables needed — fully self-contained.

---

## 🔮 Future Enhancements

- Filter support (e.g. *"WHERE salary > 50000"*)
- Date/time and time-series queries
- PDF/Excel report export
- Multi-sheet Excel support
- Database connectivity
- Advanced predictive analytics
- Multi-language column name support

---

## 📚 Project Type

**College Mini Project**

**Domain**: Data Science / Artificial Intelligence

## 👩‍💻 Developed by:
**Bhanushree G | CSE – Data Science | Nagarjuna College of Engineering and Technology (NCET)**

---

*Built with Python, Streamlit, and pandas. No external AI APIs. Everything computed locally from your data.*
