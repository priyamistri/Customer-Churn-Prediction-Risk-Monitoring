import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

from xgboost import XGBClassifier
import mlflow

from src.config import WAREHOUSE_PATH
from src.utils_duckdb import connect
from src.ml.mlflow_utils import setup_mlflow


def main():
    setup_mlflow()
    con = connect(WAREHOUSE_PATH)

    latest_date = con.execute(
        "SELECT MAX(feature_date) FROM mart.customer_features_daily"
    ).fetchone()[0]

    df = con.execute("""
        SELECT *
        FROM mart.customer_features_daily
        WHERE feature_date = ?
    """, [latest_date]).df()

    y = df["churn_label"].astype(int)
    X = df.drop(columns=["churn_label", "feature_date", "customerID"])

    cat_cols = ["Contract", "PaymentMethod", "PaperlessBilling", "InternetService", "TechSupport"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]), num_cols),
        ]
    )

    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        n_jobs=4
    )

    model = Pipeline([("pre", pre), ("model", xgb)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="xgboost_v1") as run:
        mlflow.log_param("feature_date", str(latest_date))
        mlflow.log_params({
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0
        })

        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)

        mlflow.log_metric("roc_auc", float(roc_auc))
        mlflow.log_metric("pr_auc", float(pr_auc))

        Path("artifacts").mkdir(exist_ok=True)
        local_path = Path("artifacts/xgb_v1.joblib")
        joblib.dump(
            {"model": model, "feature_date": str(latest_date), "roc_auc": float(roc_auc), "run_id": run.info.run_id},
            local_path
        )

        mlflow.log_artifact(str(local_path), artifact_path="model_artifacts")

        print(f"Trained XGBoost on {latest_date} | ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f}")
        print(f"MLflow run_id: {run.info.run_id}")


if __name__ == "__main__":
    main()