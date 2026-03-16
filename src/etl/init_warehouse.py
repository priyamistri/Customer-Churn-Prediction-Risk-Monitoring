from src.config import WAREHOUSE_PATH
from src.utils_duckdb import connect

def main():
    con = connect(WAREHOUSE_PATH)

    with open("sql/00_schemas.sql", "r", encoding="utf-8") as f:
        con.execute(f.read())

    schemas = [row[0] for row in con.execute("SHOW SCHEMAS").fetchall()]
    print(f"Warehouse created at: {WAREHOUSE_PATH}")
    print("Schemas:", schemas)

if __name__ == "__main__":
    main()