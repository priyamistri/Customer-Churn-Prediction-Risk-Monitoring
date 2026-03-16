from src.config import WAREHOUSE_PATH
from src.utils_duckdb import connect

def main():
    con = connect(WAREHOUSE_PATH)

    with open("sql/02_staging_views.sql", "r", encoding="utf-8") as f:
        con.execute(f.read())

    # Better schema check in DuckDB
    schemas = con.execute("SELECT schema_name FROM information_schema.schemata ORDER BY 1").fetchall()
    print("Schemas:", [s[0] for s in schemas])

    # Row count check
    n = con.execute("SELECT COUNT(*) FROM stg.customers").fetchone()[0]
    print(f"stg.customers rows: {n:,}")

    # Null check on TotalCharges (expected some NULLs from blanks)
    nulls = con.execute("SELECT SUM(CASE WHEN TotalCharges IS NULL THEN 1 ELSE 0 END) FROM stg.customers").fetchone()[0]
    print(f"stg.customers TotalCharges NULLs: {nulls:,}")

if __name__ == "__main__":
    main()