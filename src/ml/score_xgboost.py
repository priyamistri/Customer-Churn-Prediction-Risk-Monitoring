import joblib
import pandas as pd

from src.config import WAREHOUSE_PATH
from src.utils_duckdb import connect
from utils.export_powerbi import export_powerbi_tables


def main():
    con = connect(WAREHOUSE_PATH)

    artifact = joblib.load("artifacts/xgb_v1.joblib")
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

    # ---- predictions table written to DuckDB ----
    preds = pd.DataFrame({
        "score_date": latest_date,
        "customerID": X["customerID"],
        "churn_probability": proba,
        "model_name": "xgboost",
        "model_version": "v1",
    })

    con.execute("""
        DELETE FROM ml.churn_predictions
        WHERE score_date=? AND model_name='xgboost' AND model_version='v1'
    """, [latest_date])

    con.register("preds_df", preds)
    con.execute("INSERT INTO ml.churn_predictions SELECT * FROM preds_df")

    n = con.execute("""
        SELECT COUNT(*) FROM ml.churn_predictions
        WHERE score_date=? AND model_name='xgboost' AND model_version='v1'
    """, [latest_date]).fetchone()[0]

    # ---- Power BI export ----
    # Map your column names to the schema we chose for Power BI.
    # customers_scored.csv expects: customer_id + churn_proba (+ optional cols)
    preds_for_bi = pd.DataFrame({
        "customer_id": preds["customerID"].astype(str),
        "churn_proba": preds["churn_probability"].astype(float),

        # Optional fields (fill if you have them; leaving None is fine)
        "segment": None,
        "contract_type": None,
        "tenure_months": None,
        "monthly_charges": None,
        "total_charges": None,
        "payment_method": None,
        "internet_service": None,
        "phone_service": None,
        "online_security": None,
        "tech_support": None,
        "paperless_billing": None,
        "region": None,
    })

    export_powerbi_tables(
        preds_df=preds_for_bi,
        metrics_dict=None,
        threshold=0.50,
        model_version="v1",
    )

    print(f"Wrote {n:,} xgboost predictions for {latest_date}")


if __name__ == "__main__":
    main()