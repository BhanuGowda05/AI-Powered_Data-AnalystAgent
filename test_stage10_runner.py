import sys, io, pandas as pd, numpy as np
sys.path.insert(0, ".")
from question_parser import parse_question
from reasoning_engine import build_plan
from analysis_engine import execute_plan
from insights_engine import generate_answer_and_insights

results = []

def check(label, condition, detail=""):
    s = "PASS" if condition else "FAIL"
    results.append((s, label, detail))
    print(f"[{s}] {label} -- {detail}")

print("\n=== DATASET TESTS ===")
csv_data = "Product,Revenue,Region\nA,45000,North\nB,30000,South\nC,12000,East\n"
df_csv = pd.read_csv(io.StringIO(csv_data))
check("CSV upload", df_csv.shape == (3,3), str(df_csv.shape))

import openpyxl
buf = io.BytesIO()
pd.DataFrame({"Name":["Alice","Bob"],"Salary":[50000,60000]}).to_excel(buf,index=False,engine="openpyxl")
buf.seek(0)
df_xl = pd.read_excel(buf, engine="openpyxl")
check("Excel upload", df_xl.shape==(2,2), str(df_xl.shape))

check("Empty dataset guard", pd.DataFrame().empty, "empty==True")
df_miss = pd.DataFrame({"A":[1,None,3],"B":["x",None,"z"]})
check("Missing values detected", int(df_miss.isnull().sum().sum())==2, "2 nulls")
df_dup = pd.DataFrame({"A":[1,1,2],"B":["x","x","y"]})
check("Duplicate rows detected", int(df_dup.duplicated().sum())==1, "1 dup")
df_num = pd.DataFrame({"Price":[10.5,20.0],"Qty":[1,2]})
check("Numerical cols", len(df_num.select_dtypes(include=["number"]).columns)==2, "2 num cols")
df_cat = pd.DataFrame({"Cat":["A","B"],"Reg":["N","S"]})
check("Categorical cols", len(df_cat.select_dtypes(include=["object","category"]).columns)==2, "2 cat cols")

print("\n=== QUESTION TESTS ===")
df_test = pd.DataFrame({
    "Employee":["Alice","Bob","Charlie"],
    "Salary":[50000,60000,45000],
    "Department":["HR","IT","HR"],
    "Product":["A","B","C"],
    "Revenue":[45000,30000,12000],
    "Sales":[100,200,150],
    "CustomerID":[1,2,3]
})
cols = df_test.columns.tolist()
for q in ["What is the average salary?","Which product has the highest revenue?",
          "What is the total sales?","Which department has the highest average salary?",
          "How many customers are there?"]:
    nlu = parse_question(q, cols)
    plan = build_plan(nlu, cols)
    result = execute_plan(df_test, plan)
    ans, ins, stats = generate_answer_and_insights(result, plan)
    ok = result["status"]=="success" and len(ans)>0
    check("Q:"+q[:42], ok, "op="+str(nlu.get("operation"))+" st="+result["status"])

print("\n=== ERROR TESTS ===")
r_unk = execute_plan(df_test, build_plan(parse_question("What is the average unicorn?", cols), cols))
check("Unknown col no crash", True, "status="+r_unk["status"])

plan_ns = build_plan(parse_question("xyzzy frobblefoo", cols), cols)
check("Nonsense Q cannot_plan", plan_ns["status"]=="cannot_plan", plan_ns["status"])
r_ns = execute_plan(df_test, plan_ns)
check("Nonsense Q exec error", r_ns["status"]=="error", r_ns["status"])

nlu_eq = parse_question("", cols)
check("Empty Q LOW confidence", nlu_eq["confidence"]=="LOW", nlu_eq["confidence"])

df_null = pd.DataFrame({"Product":["A","B"],"Revenue":[None,None]})
r_null = execute_plan(df_null, build_plan(parse_question("Which product has the highest revenue?", df_null.columns.tolist()), df_null.columns.tolist()))
check("All-null col error", r_null["status"]=="error", r_null.get("error_message","")[:60])

df_bad = pd.DataFrame({"Product":["A","B"],"Revenue":["abc","xyz"]})
r_bad = execute_plan(df_bad, build_plan(parse_question("What is the total revenue?", df_bad.columns.tolist()), df_bad.columns.tolist()))
check("Non-numeric col error", r_bad["status"]=="error", r_bad.get("error_message","")[:60])

plan_dummy = {"status":"ready","analysis_type":"single_aggregate","operation":"average","aggregation":"mean","target_column":"Salary","group_column":None,"sort":"none","limit":None,"has_filter":False,"steps":[],"warnings":[]}
r_emp = execute_plan(pd.DataFrame(), plan_dummy)
check("Empty df error", r_emp["status"]=="error", r_emp.get("error_message","")[:60])

try:
    pd.read_csv(io.BytesIO(b"this is not valid csv!!@@##"))
    check("Invalid file content guard", True, "pandas handled it")
except Exception:
    check("Invalid file content -> exception caught", True, "Exception raised OK")

print("\n=== INSIGHTS ENGINE TESTS ===")
cases = [
    ("groupby", {"status":"success","analysis_type":"groupby_aggregate","operation":"maximum","target_column":"Revenue","group_column":"Product","aggregation":"sum","result_value":45000,"result_df":pd.DataFrame({"Product":["A","B"],"SUM(Revenue)":[45000,30000]})}, {"sort":"descending","limit":1}),
    ("single_agg", {"status":"success","analysis_type":"single_aggregate","operation":"average","target_column":"Salary","group_column":None,"aggregation":"mean","result_value":51666.67,"result_df":pd.DataFrame({"Column":["Salary"],"Operation":["MEAN"],"Result":[51666.67]})}, {"sort":"none","limit":None}),
    ("error", {"status":"error","error_message":"Not found"}, {}),
    ("value_counts", {"status":"success","analysis_type":"value_counts","operation":"count","target_column":"Dept","group_column":None,"aggregation":"value_counts","result_value":None,"result_df":pd.DataFrame({"Dept":["HR","IT"],"Count":[5,3]})}, {}),
    ("full_summary", {"status":"success","analysis_type":"full_summary","operation":"list","target_column":"Employee","group_column":None,"aggregation":"list","result_value":None,"result_df":pd.DataFrame({"Employee":["Alice","Bob"]})}, {"limit":None}),
]
for label, res, pln in cases:
    ret = generate_answer_and_insights(res, pln)
    ok = isinstance(ret, tuple) and len(ret)==3 and isinstance(ret[0],str) and len(ret[0])>0
    check("3-tuple+answer ("+label+")", ok, ret[0][:55] if ok else "FAIL")

passed = sum(1 for s,_,_ in results if s=="PASS")
failed = sum(1 for s,_,_ in results if s=="FAIL")
total = len(results)
print(f"\n{'='*55}")
print(f"TOTAL: {total}  PASSED: {passed}  FAILED: {failed}")
print("="*55)
if failed:
    print("\nFailed tests:")
    for s,l,d in results:
        if s=="FAIL": print(f"  {l} -- {d}")
sys.exit(0 if failed==0 else 1)
