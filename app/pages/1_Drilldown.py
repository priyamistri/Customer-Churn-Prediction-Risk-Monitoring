import json
import duckdb
import pandas as pd
import streamlit as st
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "warehouse" / "warehouse.duckdb"

st.set_page_config(page_title="Customer Drilldown", layout="wide")
st.title("Customer Drilldown (XGBoost + SHAP)")

con = duckdb.connect(str(DB_PATH))

latest_score = con.execute("SELECT MAX(score_date) FROM ml.churn_predictions").fetchone()[0]
st.caption(f"Latest score_date: {latest_score}")

cust_id = st.text_input("customerID", "")
if not cust_id:
    st.stop()

cust = con.execute("SELECT * FROM stg.customers WHERE customerID=?", [cust_id]).df()
if cust.empty:
    st.error("customerID not found.")
    st.stop()

pred = con.execute("""
    SELECT churn_probability
    FROM ml.churn_predictions
    WHERE score_date=? AND customerID=? AND model_name='xgboost' AND model_version='v1'
""", [latest_score, cust_id]).df()

exp = con.execute("""
    SELECT base_value, churn_probability, top_features_json
    FROM ml.churn_explanations
    WHERE score_date=? AND customerID=? AND model_name='xgboost' AND model_version='v1'
""", [latest_score, cust_id]).df()

c1, c2 = st.columns(2)
c1.subheader("Customer profile")
c1.dataframe(cust, width="stretch", hide_index=True)

if not pred.empty:
    c2.metric("Predicted churn (XGBoost)", f"{pred['churn_probability'].iloc[0]*100:.1f}%")
else:
    c2.warning("No XGBoost prediction found for this customer/date.")

st.subheader("Top SHAP feature drivers")
if exp.empty:
    st.warning("No SHAP explanation found. Run: python -m src.ml.explain.generate_shap")
    st.stop()

top = json.loads(exp["top_features_json"].iloc[0])
top_df = pd.DataFrame(top)
top_df["abs_shap"] = top_df["shap"].abs()
top_df = top_df.sort_values("abs_shap", ascending=False).drop(columns=["abs_shap"])

st.dataframe(top_df, width="stretch", hide_index=True)
st.caption("Positive SHAP increases churn risk; negative SHAP decreases churn risk.")