import json
from pathlib import Path
import pandas as pd
import shap
import joblib

RAW = "submissions/team-members/yan-cotta/archive/salaries.csv"
MODELDIR = Path("submissions/team-members/yan-cotta/models")
OUTDIR = Path("submissions/team-members/yan-cotta/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

CATS = ["job_category","continent","experience_level","employment_type","company_size"]
NUMS = ["remote_ratio","work_year"]
TARGET = "salary_in_usd"

if __name__ == "__main__":
    pipe = joblib.load(MODELDIR / "xgb_pipeline.pkl")
    df = pd.read_csv(RAW)
    from utils.feature_engineering import ensure_group_columns
    df = ensure_group_columns(df).drop_duplicates()

    # Take a small sample for SHAP to keep it fast
    sample = df.sample(n=min(2000, len(df)), random_state=42)
    X = sample[CATS + NUMS]

    # Use the pipeline's preprocessor to get feature names after one-hot
    pre = pipe.named_steps["pre"]
    pre.fit(X)
    X_enc = pre.transform(X)
    try:
        feature_names = list(pre.get_feature_names_out())
    except Exception:
        feature_names = [f"f{i}" for i in range(X_enc.shape[1])]
        warnings.warn(
            "Could not extract feature names from the preprocessor. "
            "Falling back to generic names, which may reduce SHAP interpretability."
        )
        # Try to use original column names if possible
        if hasattr(X, 'columns'):
            feature_names = [f"{col}_{i}" for i, col in enumerate(X.columns)]
            # If number of columns doesn't match, fallback to f{i}
            if len(feature_names) != X_enc.shape[1]:
                feature_names = [f"f{i}" for i in range(X_enc.shape[1])]
        else:
            feature_names = [f"f{i}" for i in range(X_enc.shape[1])]
    model = pipe.named_steps["xgb"]
    explainer = shap.Explainer(model, X_enc, feature_names=feature_names)
    shap_values = explainer(X_enc)

    # Summarize mean(|SHAP|) per feature
    mean_abs = shap_values.abs.mean(0).values.tolist()
    importance = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)

    out = {"top_20": importance[:20]}
    print(json.dumps(out, indent=2))
    with open(OUTDIR / "shap_top20.json", "w") as f:
        json.dump(out, f, indent=2)
