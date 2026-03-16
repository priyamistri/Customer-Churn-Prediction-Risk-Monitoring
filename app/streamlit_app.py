import duckdb
import streamlit as st
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "warehouse" / "warehouse.duckdb"

st.set_page_config(page_title="Churn Dashboard", layout="wide")
st.title("Churn Dashboard (Milestone 2)")

con = duckdb.connect(str(DB_PATH))

latest_score = con.execute("SELECT MAX(score_date) FROM ml.churn_predictions").fetchone()[0]

model = st.selectbox("Model", ["xgboost v1", "logreg_baseline v1"], index=0)
if model == "xgboost v1":
    model_name, model_version = "xgboost", "v1"
else:
    model_name, model_version = "logreg_baseline", "v1"

st.caption(f"Latest score_date: {latest_score} | model: {model_name}/{model_version}")

c1, c2, c3 = st.columns(3)
customers = con.execute("SELECT COUNT(*) FROM stg.customers").fetchone()[0]
label_rate = con.execute("SELECT AVG(churn_label) FROM stg.customers").fetchone()[0]
avg_pred = con.execute(
    "SELECT AVG(churn_probability) FROM ml.churn_predictions WHERE score_date=? AND model_name=? AND model_version=?",
    [latest_score, model_name, model_version]
).fetchone()[0]

c1.metric("Customers", f"{customers:,}")
c2.metric("Snapshot churn rate", f"{label_rate*100:.1f}%")
c3.metric("Avg predicted churn", f"{avg_pred*100:.1f}%")

st.subheader("Top customers by predicted churn")
df = con.execute("""
    SELECT p.customerID, p.churn_probability,
           c.Contract, c.PaymentMethod, c.MonthlyCharges, c.tenure
    FROM ml.churn_predictions p
    JOIN stg.customers c USING(customerID)
    WHERE p.score_date=? AND p.model_name=? AND p.model_version=?
    ORDER BY p.churn_probability DESC
    LIMIT 200
""", [latest_score, model_name, model_version]).df()

st.dataframe(df, width="stretch", hide_index=True)

st.info("Use the Drilldown page (left sidebar) to see SHAP feature drivers per customer (XGBoost only).")