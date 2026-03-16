from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
WAREHOUSE_PATH = PROJECT_ROOT / "warehouse" / "warehouse.duckdb"

TELCO_CSV_URL = (
    "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/"
    "chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)