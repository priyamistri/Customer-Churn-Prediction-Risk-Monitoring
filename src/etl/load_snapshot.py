from src.config import WAREHOUSE_PATH, DATA_RAW
from src.utils_duckdb import connect

def main():
    con = connect(WAREHOUSE_PATH)

    # Create raw table
    with open("sql/01_raw_tables.sql", "r", encoding="utf-8") as f:
        con.execute(f.read())

    csv_path = DATA_RAW / "telco_churn.csv"

    # Deterministic reruns
    con.execute("DELETE FROM raw.telco_customers_snapshot")

    con.execute("""
        INSERT INTO raw.telco_customers_snapshot
        SELECT * FROM read_csv_auto(?, HEADER=TRUE)
    """, [str(csv_path)])

    count = con.execute("SELECT COUNT(*) FROM raw.telco_customers_snapshot").fetchone()[0]
    print(f"Loaded {count:,} rows into raw.telco_customers_snapshot")

    # sanity: churn distribution
    dist = con.execute("""
        SELECT Churn, COUNT(*) AS n
        FROM raw.telco_customers_snapshot
        GROUP BY 1
        ORDER BY 1
    """).fetchall()
    print("Churn distribution:", dist)

if __name__ == "__main__":
    main()