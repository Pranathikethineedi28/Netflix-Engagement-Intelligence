"""Label definition and feature selection for the engagement-risk model.

The label reproduces the notebook: a user is "at risk" if they are in the bottom 30% for
completion rate or recommendation click-through, or the top 30% for content fatigue or time
between sessions. The four columns that define the label are removed from the inputs,
otherwise the model would just re-learn the rule (target leakage).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MASTER_TABLE = ROOT / "data" / "user_master_table.csv"

LABEL = "engagement_risk"
ID_COL = "user_id"
LEAKAGE_COLS = ["avg_completion", "recommendation_ctr", "content_fatigue_signal", "avg_time_gap_hours"]
# is_active is a different target (the failed first framing), so it is never an input either.
EXCLUDED = [ID_COL, "is_active", LABEL, *LEAKAGE_COLS]


def load_master(path: str | Path = MASTER_TABLE) -> pd.DataFrame:
    return pd.read_csv(path)


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[LABEL] = (
        (out["avg_completion"] < out["avg_completion"].quantile(0.30))
        | (out["recommendation_ctr"] < out["recommendation_ctr"].quantile(0.30))
        | (out["content_fatigue_signal"] > out["content_fatigue_signal"].quantile(0.70))
        | (out["avg_time_gap_hours"] > out["avg_time_gap_hours"].quantile(0.70))
    ).astype(int)
    return out


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return model inputs plus the categorical and numeric column names."""
    X = df.drop(columns=[c for c in EXCLUDED if c in df.columns])
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    X = X.copy()
    for col in categorical:
        X[col] = X[col].astype(str)
    return X, categorical, numeric


def risk_level(probability: float) -> str:
    """Same bands as the dashboard: Low < 1/3 <= Medium < 2/3 <= High."""
    if probability >= 2 / 3:
        return "High"
    if probability >= 1 / 3:
        return "Medium"
    return "Low"
