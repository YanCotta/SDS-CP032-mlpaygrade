import re
from typing import Optional
import pandas as pd

CATEGORY_MAP = [
    ("MANAGEMENT", [r"\b(head|director|manager|lead|vp|chief|principal)\b"]),
    ("MACHINE_LEARNING", [r"\bmachine learning\b", r"\bml\b", r"\bm l \b"]),
    ("DATA_ENGINEERING", [r"\bdata engineer\b", r"\banalytics engineer\b", r"\bplatform\b"]),
    ("DATA_SCIENCE", [r"\bdata scientist\b", r"\bresearch scientist\b", r"\bapplied scientist\b"]),
    ("DATA_ANALYSIS", [r"\bdata analyst\b", r"\bbusiness intelligence\b", r"\bbusiness analyst\b"]),
    ("SPECIALIZED", [r"\bmle ops\b", r"\bmlo(ps)?\b", r"\bdevops\b", r"\bai\b", r"\bnlp\b", r"\bvision\b", r"\brecsys\b"]),
]

# Minimal ISO country alpha-2 to continent mapping; unmapped -> OTHER
CONTINENT_MAP = {
    "US":"NORTH_AMERICA","CA":"NORTH_AMERICA","MX":"NORTH_AMERICA",
    "GB":"EUROPE","DE":"EUROPE","FR":"EUROPE","ES":"EUROPE","IT":"EUROPE","NL":"EUROPE","PL":"EUROPE","SE":"EUROPE","DK":"EUROPE","CH":"EUROPE","LU":"EUROPE","IE":"EUROPE","PT":"EUROPE","NO":"EUROPE","FI":"EUROPE","AT":"EUROPE","BE":"EUROPE","CZ":"EUROPE",
    "IN":"ASIA_PACIFIC","CN":"ASIA_PACIFIC","JP":"ASIA_PACIFIC","KR":"ASIA_PACIFIC","SG":"ASIA_PACIFIC","AU":"ASIA_PACIFIC","NZ":"ASIA_PACIFIC","PH":"ASIA_PACIFIC","ID":"ASIA_PACIFIC","VN":"ASIA_PACIFIC","MY":"ASIA_PACIFIC","TH":"ASIA_PACIFIC",
    "BR":"OTHER","AR":"OTHER","CL":"OTHER","CO":"OTHER","PE":"OTHER","UY":"OTHER","PY":"OTHER","BO":"OTHER","EC":"OTHER","VE":"OTHER",
    "ZA":"OTHER","NG":"OTHER","EG":"OTHER","MA":"OTHER","KE":"OTHER","GH":"OTHER","TN":"OTHER","DZ":"OTHER","ET":"OTHER",
    "IL":"OTHER","TR":"OTHER","AE":"OTHER","QA":"OTHER","SA":"OTHER"
}

def infer_job_category(job_title: str) -> str:
    title = (job_title or "").lower().strip()
    for label, patterns in CATEGORY_MAP:
        for pat in patterns:
            if re.search(pat, title):
                return label
    # Fallback heuristics
    if "engineer" in title: return "DATA_ENGINEERING"
    if "scientist" in title: return "DATA_SCIENCE"
    if "analyst" in title: return "DATA_ANALYSIS"
    return "SPECIALIZED"

def country_to_continent(code: Optional[str]) -> str:
    if not code or not isinstance(code, str):
        return "OTHER"
    code = code.strip().upper()
    return CONTINENT_MAP.get(code, "OTHER")

def add_job_category(df: pd.DataFrame) -> pd.DataFrame:
    if "job_category" not in df.columns and "job_title" in df.columns:
        df = df.copy()
        df["job_category"] = df["job_title"].apply(infer_job_category)
    return df

def add_continent(df: pd.DataFrame, from_col: str = "company_location") -> pd.DataFrame:
    if "continent" not in df.columns and from_col in df.columns:
        df = df.copy()
        df["continent"] = df[from_col].apply(country_to_continent)
    return df

def ensure_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = add_job_category(df)
    df = add_continent(df, from_col="company_location" if "company_location" in df.columns else "employee_residence")
    required = ["job_category","continent","experience_level","employment_type","company_size","work_year","salary_in_usd"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[ensure_group_columns] DataFrame is missing required columns: {missing}. "
            "Please ensure your DataFrame contains these columns before calling this function."
        )
    return df
