from src.config import WAREHOUSE_PATH
from src.utils_duckdb import connect

def main():
    con = connect(WAREHOUSE_PATH)

    with open("sql/03_customer_features_daily.sql", "r", encoding="utf-8") as f:
        con.execute(f.read())

    n = con.execute("SELECT COUNT(*) FROM mart.customer_features_daily").fetchone()[0]
    maxd = con.execute("SELECT MAX(feature_date) FROM mart.customer_features_daily").fetchone()[0]
    mind = con.execute("SELECT MIN(feature_date) FROM mart.customer_features_daily").fetchone()[0]
    print(f"Built mart.customer_features_daily: {n:,} rows ({mind} -> {maxd})")

if __name__ == "__main__":
    main()