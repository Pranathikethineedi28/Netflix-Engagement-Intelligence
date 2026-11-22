# 🎬 Netflix Engagement Intelligence Platform

 **Live Demo:** https://netflix-engagement-intelligence-fnfnw9drvxvaxz97ypc9sz.streamlit.app/

📂 **GitHub:** https://github.com/Pranathikethineedi28/Netflix-Engagement-Intelligence

An end-to-end machine learning platform that predicts user engagement risk, performs behavioral segmentation, and generates personalized retention strategies from Netflix-style streaming data.

## Features

- User-level feature engineering
- Engagement risk prediction using Random Forest
- K-Fold Cross Validation
- User segmentation using KMeans
- Business recommendation engine
- Explainable AI layer
- Interactive Streamlit dashboard

## Tech Stack

Python • Pandas • Scikit-learn • Streamlit • Matplotlib • KMeans • Random Forest • Git • GitHub

## Data

The project uses a **synthetic, Netflix-style dataset** (generated users, titles, watch history, searches, reviews and recommendation logs). It is not real Netflix data. Raw tables are in `data/`, and `final_user_intelligence_table.csv` is the one-row-per-user feature table that powers the dashboard.

## Modeling Approach and Results

All metrics are 5-fold stratified cross-validation means.

**1. Predicting `is_active` directly was not informative.** The Random Forest reached 0.86 accuracy, but only because 85% of users are active: its ROC-AUC was 0.55, close to random, and on a held-out split it identified just 12 of 305 inactive users. So I reframed the problem.

**2. Engagement-risk model (the deployed model).** A user is labeled *at risk* if they fall in the bottom 30% for completion rate or recommendation click-through, or the top 30% for content fatigue or time between sessions (59% of users). The four columns that define the label are excluded from the model's inputs to avoid target leakage.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.800 | 0.840 | 0.814 | 0.827 | 0.867 |
| Logistic Regression | 0.786 | 0.843 | 0.780 | 0.810 | 0.864 |

The trained pipeline is saved as `models/engagement_risk_model.pkl`, and KMeans segments users by behavior for the retention recommendations shown in the dashboard.

## Run Locally

```
pip install -r requirements.txt
streamlit run app.py
```

The full analysis is in `notebooks/Netflix_Engagement_Intelligence.ipynb`.

## Production Pipeline: Training, Tracking and Serving

The notebook was turned into a reproducible pipeline in `src/`:

| Piece | What it does |
|---|---|
| `src/features.py` | Builds the engagement-risk label and removes the four label-defining columns (target leakage) |
| `src/train.py` | 80/20 stratified split, 5-fold CV on the training split, model selection by mean CV ROC-AUC, one final score on the held-out test set, then a refit on all users for serving |
| MLflow tracking | Every run logs parameters, CV metrics (mean and std), test metrics, a hash of the training data and the model itself |
| `src/api.py` | FastAPI service with `/health`, `/predict` and `/users/{user_id}/risk` |
| `Dockerfile` | Trains the model during the image build, then serves it with uvicorn as a non-root user |
| `.github/workflows/ci.yml` | Runs the tests, builds the Docker image and smoke-tests the running container on every push |

### Results (5-fold CV on the training split, then the held-out 20%)

| Model | CV ROC-AUC | CV F1 | Held-out ROC-AUC | Held-out F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.864 ± 0.006 | 0.809 ± 0.009 | | |
| Random Forest (selected) | 0.865 ± 0.005 | 0.820 ± 0.007 | 0.861 | 0.819 |

**A trade-off worth noting:** the Random Forest wins by 0.001 ROC-AUC, well inside one standard deviation, while being far larger (about 48 MB) and harder to explain than Logistic Regression. In production I would likely serve the Logistic Regression; the selection rule is kept simple here so it's easy to audit.

### Run it

```
pip install -r requirements-api.txt

python -m src.train                                  # train + log to MLflow (sqlite:///mlflow.db)
mlflow ui --backend-store-uri sqlite:///mlflow.db    # compare runs at http://127.0.0.1:5000
uvicorn src.api:app --reload                         # API docs at http://127.0.0.1:8000/docs
pytest -q

docker build -t engagement-risk-api .
docker run -p 8000:8000 engagement-risk-api
curl http://localhost:8000/users/user_00001/risk
```

Example request:

```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"features": {"age": 25, "subscription_plan": "Basic", "total_sessions": 3}}'
```

Missing features are imputed exactly as in training and listed in the response under `missing_features`. Unknown features, and the label-defining leakage columns, are rejected with HTTP 422.
