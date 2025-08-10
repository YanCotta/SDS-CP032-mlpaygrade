import json
from pathlib import Path
import pandas as pd

RAW = "submissions/team-members/yan-cotta/archive/salaries.csv"
OUT = Path("submissions/team-members/yan-cotta/outputs/with_cpi.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Minimal CPI/COLA table (example; replace with better source later)
CPI = {
    2020: 1.00,
    2021: 1.06,
    2022: 1.13,
    2023: 1.19,
    2024: 1.23,
}

if __name__ == "__main__":
    df = pd.read_csv(RAW)
    if "work_year" not in df.columns:
        raise ValueError("Missing work_year in data")
# Load CPI/COLA table from CSV for reliable data
def load_cpi_table(cpi_csv_path):
    cpi_df = pd.read_csv(cpi_csv_path)
    if "year" not in cpi_df.columns or "cpi_factor" not in cpi_df.columns:
        raise ValueError("CPI CSV must contain 'year' and 'cpi_factor' columns")
    return dict(zip(cpi_df["year"], cpi_df["cpi_factor"]))

CPI_CSV = "submissions/team-members/yan-cotta/archive/cpi.csv"

if __name__ == "__main__":
    df = pd.read_csv(RAW)
    if "work_year" not in df.columns:
        raise ValueError("Missing work_year in data")
    cpi_table = load_cpi_table(CPI_CSV)
    missing_years = set(df["work_year"]) - set(cpi_table.keys())
    if missing_years:
        raise ValueError(f"CPI data missing for years: {sorted(missing_years)}")
    df["cpi_factor"] = df["work_year"].map(cpi_table)
    # Example: real_salary normalized to 2020 baseline
    df["real_salary_usd"] = df["salary_in_usd"] / df["cpi_factor"].astype(float)
    df.to_csv(OUT, index=False)
    print(json.dumps({"rows": int(len(df)), "cols": int(df.shape[1]), "output": str(OUT)}, indent=2))
