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
