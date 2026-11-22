import json

import pytest
from fastapi.testclient import TestClient

from src.features import EXCLUDED, LABEL, LEAKAGE_COLS, add_label, load_master, risk_level, split_features
from src.train import train


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train a small, fast model once for the whole test session (no MLflow, temp location)."""
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    meta = train(n_estimators=20, use_mlflow=False, model_path=path)
    return path, meta


@pytest.fixture(scope="session")
def client(trained, monkeypatch_session):
    path, _ = trained
    monkeypatch_session.setenv("MODEL_PATH", str(path))
    from src import api

    api.load_model.cache_clear()
    return TestClient(api.app)


@pytest.fixture(scope="session")
def monkeypatch_session():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_label_matches_the_notebook_definition():
    df = add_label(load_master())
    assert df[LABEL].isin([0, 1]).all()
    assert 0.55 < df[LABEL].mean() < 0.62  # the notebook reported 58.6% at risk


def test_label_defining_columns_are_never_model_inputs():
    X, _, _ = split_features(add_label(load_master()))
    assert not set(LEAKAGE_COLS) & set(X.columns)
    assert not set(EXCLUDED) & set(X.columns)


def test_risk_levels():
    assert risk_level(0.9) == "High" and risk_level(0.5) == "Medium" and risk_level(0.1) == "Low"


def test_training_beats_chance_and_writes_metadata(trained):
    path, meta = trained
    assert path.exists()
    saved = json.loads(path.with_suffix(".json").read_text())
    assert saved["model"] in {"random_forest", "logistic_regression"}
    assert saved["test"]["roc_auc"] > 0.8  # far above 0.5 (chance)
    assert set(saved["leakage_cols_removed"]) == set(LEAKAGE_COLS)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and "roc_auc" in body["test_metrics"]


def test_user_lookup(client):
    body = client.get("/users/user_00001/risk").json()
    assert 0 <= body["risk_probability"] <= 1
    assert body["risk_level"] in {"Low", "Medium", "High"}


def test_unknown_user_is_404(client):
    assert client.get("/users/nobody/risk").status_code == 404


def test_predict_with_partial_features_reports_missing(client):
    body = client.post("/predict", json={"features": {"age": 30, "subscription_plan": "Premium", "total_sessions": 12}}).json()
    assert 0 <= body["risk_probability"] <= 1
    assert "monthly_spend" in body["missing_features"]


def test_predict_rejects_unknown_and_leakage_features(client):
    r = client.post("/predict", json={"features": {"avg_completion": 0.2}})
    assert r.status_code == 422 and "avg_completion" in r.json()["detail"]


def test_predict_rejects_text_in_numeric_feature(client):
    r = client.post("/predict", json={"features": {"age": "thirty"}})
    assert r.status_code == 422
