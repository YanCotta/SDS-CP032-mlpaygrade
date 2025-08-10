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
    df["cpi_factor"] = df["work_year"].map(CPI)
    # Example: real_salary normalized to 2020 baseline
    df["real_salary_usd"] = df["salary_in_usd"] / df["cpi_factor"].astype(float)
    df.to_csv(OUT, index=False)
    print(json.dumps({"rows": int(len(df)), "cols": int(df.shape[1]), "output": str(OUT)}, indent=2))
