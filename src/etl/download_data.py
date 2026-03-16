import pandas as pd
from src.config import DATA_RAW, TELCO_CSV_URL

def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out = DATA_RAW / "telco_churn.csv"

    df = pd.read_csv(TELCO_CSV_URL)
    df.to_csv(out, index=False)

    print(f"Saved: {out}")
    print(f"Rows: {len(df):,} | Cols: {len(df.columns)}")

if __name__ == "__main__":
    main()