"""Reproducible training run for the engagement-risk model, tracked with MLflow.

Steps:
1. Build the label and drop leakage-prone columns (src/features.py).
2. Hold out 20% of users as a final test set, stratified by the label.
3. Compare Logistic Regression and Random Forest with 5-fold stratified CV on the training split.
4. Pick the model with the best mean CV ROC-AUC and score it once on the held-out test set.
5. Refit the chosen model on all users and save it for the API, with a metadata file.

Every run logs parameters, CV metrics (mean and std), test metrics and the model to MLflow.

Usage:
    python -m src.train                    # full run, logged to ./mlflow.db
    python -m src.train --no-mlflow        # same run without tracking (used in the Docker build)
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse runs at http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import LABEL, LEAKAGE_COLS, MASTER_TABLE, ROOT, add_label, load_master, split_features

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "engagement_risk_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "engagement_risk_pipeline.json"
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
SEED = 42


def make_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def make_models(n_estimators: int) -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators, random_state=SEED, class_weight="balanced", n_jobs=-1
        ),
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def train(n_estimators: int = 200, use_mlflow: bool = True, model_path: Path = MODEL_PATH,
          data_path: Path = MASTER_TABLE, tracking_uri: str = "sqlite:///mlflow.db") -> dict:
    df = add_label(load_master(data_path))
    X, categorical, numeric = split_features(df)
    y = df[LABEL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    if use_mlflow:
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("engagement-risk")

    cv_results = {}
    for name, estimator in make_models(n_estimators).items():
        pipe = Pipeline([("preprocessor", make_preprocessor(categorical, numeric)), ("model", estimator)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=METRICS)
        summary = {m: {"mean": float(np.mean(scores[f"test_{m}"])), "std": float(np.std(scores[f"test_{m}"]))}
                   for m in METRICS}
        cv_results[name] = summary
        if use_mlflow:
            with mlflow.start_run(run_name=f"cv-{name}"):
                mlflow.log_params({"model": name, "n_estimators": n_estimators, "cv_folds": 5,
                                   "leakage_cols_removed": ",".join(LEAKAGE_COLS),
                                   "data_sha256": file_sha256(Path(data_path)), "n_features": X.shape[1]})
                for m, s in summary.items():
                    mlflow.log_metric(f"cv_{m}_mean", s["mean"])
                    mlflow.log_metric(f"cv_{m}_std", s["std"])

    best_name = max(cv_results, key=lambda n: cv_results[n]["roc_auc"]["mean"])
    best = Pipeline([("preprocessor", make_preprocessor(categorical, numeric)),
                     ("model", make_models(n_estimators)[best_name])])
    best.fit(X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = {
        "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred), "roc_auc": roc_auc_score(y_test, proba),
    }
    test_metrics = {k: float(v) for k, v in test_metrics.items()}

    # Final artifact: refit on every user so the served model uses all available data.
    final = Pipeline([("preprocessor", make_preprocessor(categorical, numeric)),
                      ("model", make_models(n_estimators)[best_name])])
    final.fit(X, y)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, model_path)

    metadata = {
        "model": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_sha256": file_sha256(Path(data_path)),
        "n_users": int(len(df)),
        "positive_rate": float(y.mean()),
        "features": list(X.columns),
        "categorical_features": categorical,
        "leakage_cols_removed": LEAKAGE_COLS,
        "cv": cv_results,
        "test": test_metrics,
    }
    model_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))

    if use_mlflow:
        import mlflow  # noqa: PLC0415

        with mlflow.start_run(run_name=f"final-{best_name}"):
            mlflow.log_params({"model": best_name, "n_estimators": n_estimators, "selected_by": "cv_roc_auc_mean"})
            for m, v in test_metrics.items():
                mlflow.log_metric(f"test_{m}", v)
            mlflow.log_artifact(str(model_path.with_suffix(".json")))
            log_sklearn_model(mlflow, final)
    return metadata


def log_sklearn_model(mlflow, model) -> None:
    """Log the pipeline to MLflow.

    MLflow 3 serializes scikit-learn models with skops and refuses tree models unless their
    internal types are explicitly trusted. We trained this model ourselves, so we trust exactly
    the two types it reports. Older MLflow versions don't have the parameter, hence the fallback.
    """
    trusted = ["sklearn.tree._tree.Tree", "numpy.dtype"]
    try:
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=trusted)
    except TypeError:
        mlflow.sklearn.log_model(model, artifact_path="model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    meta = train(n_estimators=args.n_estimators, use_mlflow=not args.no_mlflow)
    print(f"Selected {meta['model']} (best mean CV ROC-AUC)")
    for name, res in meta["cv"].items():
        print(f"  CV {name:20s} " + "  ".join(f"{m} {s['mean']:.3f}±{s['std']:.3f}" for m, s in res.items()))
    print("  Held-out test        " + "  ".join(f"{m} {v:.3f}" for m, v in meta["test"].items()))
    print(f"Saved {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
