"""
MLPayGrade XGBoost Training Pipeline - PRODUCTION VERSION

This script implements the final, leak-proof training pipeline using:
- sklearn.pipeline.Pipeline with ColumnTransformer (fitted on train only)
- Temporal splits: 2020-2022 train, 2023 val, 2024 test
- Deduplication to prevent data leakage
- Deterministic feature engineering

DEPRECATED ARTIFACTS (DO NOT USE):
- preprocessed_mlpaygrade_data.csv (legacy preprocessing)
- salary_encoders.pkl (legacy encoders) 
- salary_scaler.pkl (legacy scalers)

Use only the Pipeline object saved by this script for training/inference.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
from utils.feature_engineering import ensure_group_columns

RAW = "submissions/team-members/yan-cotta/archive/salaries.csv"
OUTDIR = Path("submissions/team-members/yan-cotta/models")
OUTDIR.mkdir(parents=True, exist_ok=True)

CATS = ["job_category","continent","experience_level","employment_type","company_size"]
NUMS = ["remote_ratio","work_year"]
TARGET = "salary_in_usd"
TIME = "work_year"

def temporal_split(df: pd.DataFrame):
    train = df[df[TIME].between(2020, 2022)]
    val   = df[df[TIME] == 2023]
    test  = df[df[TIME] == 2024]
    return train, val, test

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:submissions/team-members/yan-cotta/mlruns")
    mlflow.set_experiment("mlpaygrade-no-leak")

    df = pd.read_csv(RAW)
    df = ensure_group_columns(df)
    df = df.drop_duplicates()

    train_df, val_df, test_df = temporal_split(df)
    X_train, y_train = train_df[CATS+NUMS], train_df[TARGET]
    X_val,   y_val   = val_df[CATS+NUMS],   val_df[TARGET]
    X_test,  y_test  = test_df[CATS+NUMS],  test_df[TARGET]

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

    with mlflow.start_run(run_name="xgb_temporal_no_leak"):
        pipe.fit(X_train, y_train)

        metrics = {}
        for split, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
            pred = pipe.predict(X)
            metrics[f"{split}_mae"] = float(mean_absolute_error(y, pred))
            metrics[f"{split}_r2"] = float(r2_score(y, pred))

        print(json.dumps(metrics, indent=2))
        for k,v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(pipe, "model")
        import joblib
        joblib.dump(pipe, OUTDIR / "xgb_pipeline.pkl")
        with open(OUTDIR / "xgb_metrics.json","w") as f:
            json.dump(metrics, f, indent=2)
