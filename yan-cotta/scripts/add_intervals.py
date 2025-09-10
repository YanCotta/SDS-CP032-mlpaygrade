import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from mapie.regression import MapieRegressor
from sklearn.metrics import mean_absolute_error
from utils.feature_engineering import ensure_group_columns

RAW = "submissions/team-members/yan-cotta/archive/salaries.csv"
MODELDIR = Path("submissions/team-members/yan-cotta/models")

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
    pipe = joblib.load(MODELDIR / "xgb_pipeline.pkl")

    df = pd.read_csv(RAW)
    df = ensure_group_columns(df).drop_duplicates()
    train_df, val_df, test_df = temporal_split(df)

    X_train, y_train = train_df[CATS+NUMS], train_df[TARGET]
    X_test,  y_test  = test_df[CATS+NUMS],  test_df[TARGET]

    # Use K-fold residuals for better-calibrated intervals with jackknife+ style
    mapie = MapieRegressor(estimator=pipe, method="plus", cv=5, random_state=42)
    mapie.fit(X_train, y_train)

    y_pred, y_int = mapie.predict(X_test, alpha=0.1)  # 90% intervals
    coverage = float(np.mean((y_test.values >= y_int[:,0]) & (y_test.values <= y_int[:,1])))
    width = float(np.mean(y_int[:,1] - y_int[:,0]))
    mae = float(mean_absolute_error(y_test, y_pred))

    print(json.dumps({"test_mae": mae, "coverage_90": coverage, "avg_width": width}, indent=2))
    joblib.dump(mapie, MODELDIR / "xgb_mapie.pkl")
    with open(MODELDIR / "xgb_intervals.json","w") as f:
        json.dump({"test_mae": mae, "coverage_90": coverage, "avg_width": width}, f, indent=2)
