import json
import numpy as np
import pandas as pd
from pathlib import Path
from utils.feature_engineering import ensure_group_columns

RAW = "submissions/team-members/yan-cotta/archive/salaries.csv"
OUTDIR = Path("submissions/team-members/yan-cotta/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

GROUP_COLS = ["job_category","continent","experience_level","employment_type","company_size","work_year"]
TARGET = "salary_in_usd"

def leave_one_out_group_mae(df: pd.DataFrame) -> dict:
    g = df.groupby(GROUP_COLS)[TARGET].agg(["sum","count"]).rename(columns={"sum":"g_sum","count":"g_n"})
    df = df.merge(g, on=GROUP_COLS, how="left")
    global_mean = df[TARGET].mean()
    loo = np.where(df["g_n"]>1, (df["g_sum"] - df[TARGET])/(df["g_n"]-1), global_mean)
    abs_err = np.abs(df[TARGET] - loo)
    res = {
        "n_rows": int(len(df)),
        "n_groups": int(len(g)),
        "global_mae": float(np.abs(df[TARGET]-global_mean).mean()),
        "group_loo_mae_mean": float(abs_err.mean()),
        "group_loo_mae_median": float(np.median(abs_err)),
        "group_loo_mae_p90": float(np.percentile(abs_err, 90)),
    }
    # Save group stats for inspection
    gstats = df[GROUP_COLS + [TARGET]].groupby(GROUP_COLS).agg(["count","mean","std"]).reset_index()
    gstats.to_csv(OUTDIR / "group_stats.csv", index=False)
    with open(OUTDIR / "group_ceiling_metrics.json","w") as f:
        json.dump(res, f, indent=2)
    return res

if __name__ == "__main__":
    df = pd.read_csv(RAW)
    df = ensure_group_columns(df)
    res = leave_one_out_group_mae(df)
    print(json.dumps(res, indent=2))
