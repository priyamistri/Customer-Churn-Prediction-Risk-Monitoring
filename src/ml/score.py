import joblib
import pandas as pd

from src.config import WAREHOUSE_PATH
from src.utils_duckdb import connect

def main():
    con = connect(WAREHOUSE_PATH)

    with open("sql/04_predictions_tables.sql", "r", encoding="utf-8") as f:
        con.execute(f.read())

    artifact = joblib.load("artifacts/logreg_baseline.joblib")
    model = artifact["model"]

    latest_date = con.execute(
        "SELECT MAX(feature_date) FROM mart.customer_features_daily"
    ).fetchone()[0]

    df = con.execute("""
        SELECT *
        FROM mart.customer_features_daily
        WHERE feature_date = ?
    """, [latest_date]).df()

    X = df.drop(columns=["churn_label", "feature_date"])
    proba = model.predict_proba(X.drop(columns=["customerID"]))[:, 1]

    preds = pd.DataFrame({
        "score_date": latest_date,
        "customerID": X["customerID"],
        "churn_probability": proba,
        "model_name": "logreg_baseline",
        "model_version": "v1"
    })

    # keep reruns deterministic: delete prior scores for same date/model
    con.execute("""
        DELETE FROM ml.churn_predictions
        WHERE score_date = ? AND model_name = 'logreg_baseline' AND model_version = 'v1'
    """, [latest_date])

    con.register("preds_df", preds)
    con.execute("INSERT INTO ml.churn_predictions SELECT * FROM preds_df")

    n = con.execute("""
        SELECT COUNT(*) FROM ml.churn_predictions
        WHERE score_date = ? AND model_name='logreg_baseline' AND model_version='v1'
    """, [latest_date]).fetchone()[0]

    print(f"Wrote {n:,} predictions for {latest_date}")

if __name__ == "__main__":
    main()