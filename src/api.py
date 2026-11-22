"""FastAPI service that serves the engagement-risk model.

Endpoints:
    GET  /health                 model name, training time and test metrics
    POST /predict                score one user from their behavioral features
    GET  /users/{user_id}/risk   look up a known user in the master table and score them

Run locally:
    python -m src.train --no-mlflow          # creates models/engagement_risk_pipeline.joblib
    uvicorn src.api:app --reload             # docs at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import ID_COL, MASTER_TABLE, risk_level, split_features
from src.train import MODEL_PATH

app = FastAPI(
    title="Netflix Engagement Risk API",
    description="Scores how likely a user is to disengage, using the engagement-risk model.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    features: dict[str, float | int | str | None] = Field(
        ..., description="Feature name to value. Missing features are imputed the same way as in training."
    )


class PredictResponse(BaseModel):
    risk_probability: float
    risk_level: str
    model: str
    missing_features: list[str]


def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", MODEL_PATH))


@lru_cache(maxsize=1)
def load_model():
    path = model_path()
    if not path.exists():
        raise RuntimeError(f"Model not found at {path}. Run `python -m src.train --no-mlflow` first.")
    return joblib.load(path), json.loads(path.with_suffix(".json").read_text())


@lru_cache(maxsize=1)
def load_master_table() -> pd.DataFrame:
    return pd.read_csv(os.environ.get("MASTER_TABLE", MASTER_TABLE)).set_index(ID_COL)


def score(features: dict) -> PredictResponse:
    pipeline, meta = load_model()
    expected = meta["features"]
    unknown = sorted(set(features) - set(expected))
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown features: {unknown}")
    categorical = set(meta["categorical_features"])
    for name, value in features.items():
        if name not in categorical and isinstance(value, str):
            raise HTTPException(status_code=422, detail=f"Feature '{name}' must be numeric")

    missing = [f for f in expected if features.get(f) is None]
    row = {f: features.get(f) for f in expected}
    frame = pd.DataFrame([row], columns=expected)
    for col in categorical:  # same treatment as training: categories as strings, NaN for missing
        frame[col] = frame[col].astype(object).where(frame[col].notna(), math.nan)
        frame[col] = frame[col].map(lambda v: v if isinstance(v, float) and math.isnan(v) else str(v))
    numeric = [f for f in expected if f not in categorical]
    frame[numeric] = frame[numeric].apply(pd.to_numeric)

    probability = float(pipeline.predict_proba(frame)[0, 1])
    return PredictResponse(
        risk_probability=round(probability, 4),
        risk_level=risk_level(probability),
        model=meta["model"],
        missing_features=missing,
    )


@app.get("/health")
def health() -> dict:
    try:
        _, meta = load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "model": meta["model"], "trained_at": meta["trained_at"], "test_metrics": meta["test"]}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    return score(request.features)


@app.get("/users/{user_id}/risk", response_model=PredictResponse)
def user_risk(user_id: str) -> PredictResponse:
    table = load_master_table()
    if user_id not in table.index:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    X, _, _ = split_features(table.loc[[user_id]].reset_index())
    record = X.iloc[0].to_dict()
    return score({k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in record.items()})
