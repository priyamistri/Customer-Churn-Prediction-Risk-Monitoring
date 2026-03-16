from src.config import WAREHOUSE_PATH
from src.utils_duckdb import connect

def assert_eq(con, name, sql, expected):
    val = con.execute(sql).fetchone()[0]
    ok = (val == expected)
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}: {val} (expected {expected})")
    if not ok:
        raise AssertionError(f"{name} failed: got {val}, expected {expected}")

def assert_zero(con, name, sql):
    assert_eq(con, name, sql, 0)

def main():
    con = connect(WAREHOUSE_PATH)

    assert_eq(con, "raw snapshot count", "SELECT COUNT(*) FROM raw.telco_customers_snapshot", 7043)
    assert_eq(con, "stg customers count", "SELECT COUNT(*) FROM stg.customers", 7043)

    assert_zero(con, "null customerIDs", "SELECT COUNT(*) FROM stg.customers WHERE customerID IS NULL")
    assert_zero(con, "invalid churn_label", "SELECT COUNT(*) FROM stg.customers WHERE churn_label NOT IN (0,1)")

    assert_eq(
        con,
        "features rows on latest date",
        "SELECT COUNT(*) FROM mart.customer_features_daily WHERE feature_date=(SELECT MAX(feature_date) FROM mart.customer_features_daily)",
        7043
    )

    assert_eq(
        con,
        "xgboost preds rows on latest score_date",
        "SELECT COUNT(*) FROM ml.churn_predictions WHERE score_date=(SELECT MAX(score_date) FROM ml.churn_predictions) AND model_name='xgboost' AND model_version='v1'",
        7043
    )

    print("All data quality checks passed.")

if __name__ == "__main__":
    main()