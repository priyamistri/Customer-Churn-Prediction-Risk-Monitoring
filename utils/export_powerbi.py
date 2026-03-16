from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

def export_powerbi_tables(
    preds_df: pd.DataFrame,
    metrics_dict: dict | None = None,
    threshold: float = 0.50,
    model_version: str = "v1",
    artifacts_dir: str = "artifacts",
) -> None:
    ART = Path(artifacts_dir)
    ART.mkdir(exist_ok=True)

    run_ts = datetime.now(timezone.utc).isoformat()

    preds_out = preds_df.copy()
    preds_out["run_ts"] = run_ts
    preds_out["threshold"] = threshold
    preds_out["model_version"] = model_version

    if "churn_label" not in preds_out.columns:
        preds_out["churn_label"] = (preds_out["churn_proba"] >= threshold).astype(int)

    for c in ["segment","contract_type","tenure_months","monthly_charges","total_charges",
              "payment_method","internet_service","phone_service","online_security",
              "tech_support","paperless_billing","region"]:
        if c not in preds_out.columns:
            preds_out[c] = None

    preds_out = preds_out[[
        "run_ts","customer_id","churn_proba","churn_label","threshold","model_version",
        "segment","contract_type","tenure_months","monthly_charges","total_charges",
        "payment_method","internet_service","phone_service","online_security",
        "tech_support","paperless_billing","region"
    ]]

    preds_out.to_csv(ART / "customers_scored.csv", index=False)

    m = metrics_dict or {}
    metrics_row = pd.DataFrame([{
        "run_ts": run_ts,
        "model_version": model_version,
        "threshold": threshold,
        "auc": m.get("auc"),
        "accuracy": m.get("accuracy"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "f1": m.get("f1"),
        "customers_scored": int(len(preds_out)),
        "high_risk_count": int((preds_out["churn_proba"] >= threshold).sum()),
        "avg_churn_proba": float(preds_out["churn_proba"].mean()) if len(preds_out) else 0.0,
    }])

    metrics_path = ART / "run_metrics.csv"
    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        old = pd.read_csv(metrics_path)
        metrics_row = pd.concat([old, metrics_row], ignore_index=True)

    metrics_row.to_csv(metrics_path, index=False)