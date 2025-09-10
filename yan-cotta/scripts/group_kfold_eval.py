import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from utils.feature_engineering import ensure_group_columns

RAW = "submissions/team-members/yan-cotta/archive/salaries.csv"
OUTDIR = Path("submissions/team-members/yan-cotta/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

CATS = ["job_category","continent","experience_level","employment_type","company_size"]
NUMS = ["remote_ratio","work_year"]
TARGET = "salary_in_usd"

# Build group key independent of year to simulate unseen categorical combos
GROUP_KEY = lambda df: df["job_category"] + "|" + df["continent"] + "|" + df["experience_level"] + "|" + df["employment_type"] + "|" + df["company_size"]

if __name__ == "__main__":
    df = pd.read_csv(RAW)
    df = ensure_group_columns(df).drop_duplicates()
    df["group_key"] = GROUP_KEY(df)

    X = df[CATS + NUMS]
    y = df[TARGET]
    groups = df["group_key"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATS),
        ("num", "passthrough", NUMS),
    ])

    xgb = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    pipe = Pipeline([("pre", pre), ("xgb", xgb)])

    gkf = GroupKFold(n_splits=5)
    maes = []
    r2s = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        pipe.fit(X_tr, y_tr)
        y_hat = pipe.predict(X_te)
        maes.append(mean_absolute_error(y_te, y_hat))
        r2s.append(r2_score(y_te, y_hat))

    res = {"gkf_mae_mean": float(np.mean(maes)), "gkf_mae_std": float(np.std(maes)),
           "gkf_r2_mean": float(np.mean(r2s)), "gkf_r2_std": float(np.std(r2s))}
    print(json.dumps(res, indent=2))
    with open(OUTDIR / "group_kfold_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
