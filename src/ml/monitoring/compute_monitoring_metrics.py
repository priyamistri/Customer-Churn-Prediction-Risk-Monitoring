import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from src.config import WAREHOUSE_PATH
from src.utils_duckdb import connect


def psi(expected, actual, bins=10, eps=1e-6):
    """Population Stability Index: compares distribution shift."""
    q = np.linspace(0, 1, bins + 1)
    cuts = np.quantile(expected, q)
    cuts[0] = -np.inf
    cuts[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=cuts)
    act_counts, _ = np.histogram(actual, bins=cuts)

    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)

    exp_pct = np.clip(exp_pct, eps, 1)
    act_pct = np.clip(act_pct, eps, 1)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def main():
    con = connect(WAREHOUSE_PATH)

    with open("sql/07_model_monitoring.sql", "r", encoding="utf-8") as f:
        con.execute(f.read())

    score_date = con.execute("SELECT MAX(score_date) FROM ml.churn_predictions").fetchone()[0]
    model_name, model_version = "xgboost", "v1"

    preds = con.execute("""
        SELECT churn_probability
        FROM ml.churn_predictions
        WHERE score_date=? AND model_name=? AND model_version=?
    """, [score_date, model_name, model_version]).df()

    if preds.empty:
        raise SystemExit("No predictions found for monitoring.")

    p = preds["churn_probability"].astype(float).values
    n_scored = int(len(p))
    avg_proba = float(np.mean(p))
    std_proba = float(np.std(p))
    p95_proba = float(np.quantile(p, 0.95))

    prev = con.execute("""
        SELECT churn_probability
        FROM ml.churn_predictions
        WHERE score_date < ? AND model_name=? AND model_version=?
        ORDER BY score_date DESC
        LIMIT 7043
    """, [score_date, model_name, model_version]).df()

    psi_val = None
    if not prev.empty:
        psi_val = psi(prev["churn_probability"].astype(float).values, p)

    miss = con.execute("""
        SELECT
          AVG(CASE WHEN TotalCharges IS NULL THEN 1 ELSE 0 END) AS missing_totalcharges_rate,
          AVG(CASE WHEN MonthlyCharges IS NULL THEN 1 ELSE 0 END) AS missing_monthlycharges_rate
        FROM mart.customer_features_daily
        WHERE feature_date=(SELECT MAX(feature_date) FROM mart.customer_features_daily)
    """).df().iloc[0]

    row = pd.DataFrame([{
        "score_date": score_date,
        "model_name": model_name,
        "model_version": model_version,
        "n_scored": n_scored,
        "avg_proba": avg_proba,
        "std_proba": std_proba,
        "p95_proba": p95_proba,
        "psi_vs_prev": psi_val,
        "missing_totalcharges_rate": float(miss["missing_totalcharges_rate"]),
        "missing_monthlycharges_rate": float(miss["missing_monthlycharges_rate"]),
        "created_at": datetime.utcnow(),
    }])

    con.execute("""
        DELETE FROM ml.model_monitoring_daily
        WHERE score_date=? AND model_name=? AND model_version=?
    """, [score_date, model_name, model_version])

    con.register("row_df", row)
    con.execute("INSERT INTO ml.model_monitoring_daily SELECT * FROM row_df")

    # ---- Power BI metrics export (append) ----
    ART = Path("artifacts")
    ART.mkdir(exist_ok=True)

    run_ts = datetime.now(timezone.utc).isoformat()
    threshold = 0.50

    metrics_row = pd.DataFrame([{
        "run_ts": run_ts,
        "model_version": model_version,
        "threshold": threshold,
        # You do not compute these here, so keep them blank
        "auc": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        # These you DO compute
        "customers_scored": n_scored,
        "high_risk_count": int((p >= threshold).sum()),
        "avg_churn_proba": avg_proba,
    }])

    metrics_path = ART / "run_metrics.csv"
    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        old = pd.read_csv(metrics_path)
        metrics_row = pd.concat([old, metrics_row], ignore_index=True)

    metrics_row.to_csv(metrics_path, index=False)

    print("Wrote monitoring row:", row.to_dict(orient="records")[0])
    print("Appended Power BI metrics row to artifacts/run_metrics.csv")


if __name__ == "__main__":
    main()