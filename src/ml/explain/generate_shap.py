import json
import joblib
import numpy as np
import pandas as pd
import shap

from src.config import WAREHOUSE_PATH
from src.utils_duckdb import connect


def get_feature_names(preprocessor):
    out = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "named_steps") and "ohe" in trans.named_steps:
            ohe = trans.named_steps["ohe"]
            out.extend(list(ohe.get_feature_names_out(cols)))
        else:
            out.extend(list(cols))
    return out


def main(top_k: int = 8, sample_background: int = 300):
    con = connect(WAREHOUSE_PATH)

    with open("sql/05_explanations_tables.sql", "r", encoding="utf-8") as f:
        con.execute(f.read())

    artifact = joblib.load("artifacts/xgb_v1.joblib")
    pipeline = artifact["model"]

    latest_date = con.execute(
        "SELECT MAX(feature_date) FROM mart.customer_features_daily"
    ).fetchone()[0]

    df = con.execute("""
        SELECT *
        FROM mart.customer_features_daily
        WHERE feature_date = ?
    """, [latest_date]).df()

    preds = con.execute("""
        SELECT customerID, churn_probability
        FROM ml.churn_predictions
        WHERE score_date=? AND model_name='xgboost' AND model_version='v1'
    """, [latest_date]).df()

    df = df.merge(preds, on="customerID", how="left")

    X_raw = df.drop(columns=["churn_label", "feature_date", "customerID"])
    customer_ids = df["customerID"].values
    churn_prob = df["churn_probability"].values

    pre = pipeline.named_steps["pre"]
    xgb = pipeline.named_steps["model"]

    X_trans = pre.transform(X_raw)
    feature_names = get_feature_names(pre)

    n = X_trans.shape[0]
    bg_idx = np.random.RandomState(42).choice(n, size=min(sample_background, n), replace=False)
    X_bg = X_trans[bg_idx]

    explainer = shap.TreeExplainer(xgb, data=X_bg, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_trans)
    base_value = float(np.array(explainer.expected_value).reshape(-1)[0])

    rows = []
    for i in range(n):
        sv = shap_values[i]
        idx = np.argsort(np.abs(sv))[::-1][:top_k]

        top = []
        for j in idx:
            top.append({
                "feature": feature_names[j],
                "shap": float(sv[j]),
                "value": float(X_trans[i, j]) if hasattr(X_trans, "__getitem__") else None
            })

        rows.append({
            "score_date": latest_date,
            "customerID": customer_ids[i],
            "model_name": "xgboost",
            "model_version": "v1",
            "base_value": base_value,
            "churn_probability": float(churn_prob[i]) if churn_prob[i] == churn_prob[i] else None,
            "top_features_json": json.dumps(top),
            "all_features_json": ""
        })

    out_df = pd.DataFrame(rows)

    con.execute("""
        DELETE FROM ml.churn_explanations
        WHERE score_date=? AND model_name='xgboost' AND model_version='v1'
    """, [latest_date])

    con.register("exp_df", out_df)
    con.execute("INSERT INTO ml.churn_explanations SELECT * FROM exp_df")

    print(f"Saved SHAP explanations for {len(out_df):,} customers on {latest_date}")


if __name__ == "__main__":
    main()