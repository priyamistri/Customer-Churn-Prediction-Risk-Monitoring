import joblib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import WAREHOUSE_PATH
from src.utils_duckdb import connect

def main():
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

    clf = Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(
        {"model": clf, "feature_date": str(latest_date), "roc_auc": float(auc)},
        "artifacts/logreg_baseline.joblib"
    )

    print(f"Trained logreg baseline on {latest_date} | ROC-AUC={auc:.4f}")

if __name__ == "__main__":
    main()