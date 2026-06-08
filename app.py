import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/final_user_intelligence_table.csv")
# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Netflix Engagement Intelligence",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🎬 Netflix AI Dashboard")
st.sidebar.markdown("""
### Modules
- 👤 User Intelligence
- 📊 Business Dashboard
- 🤖 Model Insights
""")

# -----------------------------
# Header
# -----------------------------
st.title("🎬 Netflix Engagement Intelligence Platform")
st.caption("ML-powered user engagement risk prediction, segmentation, and business action engine")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "👤 User Intelligence",
    "📊 Business Dashboard",
    "🤖 Model Insights"
])

# -----------------------------
# Tab 1: User Intelligence
# -----------------------------
with tab1:
    st.subheader("User Lookup")

    user_id = st.text_input(
        "Enter User ID",
        "user_00001"
    )

    if user_id not in df["user_id"].values:
        st.error("User ID not found. Try user_00001, user_00003, etc.")
    else:
        user = df[df["user_id"] == user_id].iloc[0]

        st.subheader("User Risk Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Risk Probability", f"{user['risk_probability']:.2%}")
        col3.metric("Segment", user["segment_name"])
        col4.metric("Subscription", user["subscription_plan"])

        if user["risk_level"] == "High":
            col2.error("🔴 HIGH RISK")
        elif user["risk_level"] == "Medium":
            col2.warning("🟡 MEDIUM RISK")
        else:
            col2.success("🟢 LOW RISK")

        st.subheader("Engagement Risk Gauge")
        st.progress(float(user["risk_probability"]))
        st.write(f"Risk Probability: **{user['risk_probability']:.2%}**")

        st.subheader("Risk Explanation")
        st.write(user["risk_explanation"])

        st.subheader("Recommended Business Action")
        st.success(user["recommended_action"])

        st.subheader("User Profile")

        profile_cols = [
            "age",
            "gender",
            "country",
            "state_province",
            "monthly_spend",
            "primary_device",
            "household_size"
        ]

        st.table(user[profile_cols].astype(str).to_frame("Value"))

        st.subheader("Behavior Metrics")

        metric_cols = [
            "total_sessions",
            "total_watches",
            "total_watch_time",
            "avg_watch_time",
            "avg_completion",
            "recommendation_ctr",
            "recommendation_ignore_rate",
            "search_count",
            "avg_sentiment_score",
            "recommendation_trust_score",
            "content_fatigue_signal"
        ]

        st.table(user[metric_cols].astype(str).to_frame("Value"))

# -----------------------------
# Tab 2: Business Dashboard
# -----------------------------
with tab2:
    st.subheader("Executive KPIs")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Users", len(df))
    c2.metric("High Risk Users", (df["risk_level"] == "High").sum())
    c3.metric("Average Risk", f"{df['risk_probability'].mean():.2%}")
    c4.metric("Power Watchers", (df["segment_name"] == "Power Watchers").sum())

    st.subheader("Risk Level Distribution")

    fig, ax = plt.subplots(figsize=(5, 5))
    df["risk_level"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax
    )
    ax.set_ylabel("")
    ax.set_title("Users by Risk Level")
    st.pyplot(fig)

    st.subheader("User Segment Distribution")

    segment_counts = df["segment_name"].value_counts()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    segment_counts.plot(kind="bar", ax=ax2)
    ax2.set_xlabel("Segment")
    ax2.set_ylabel("Number of Users")
    ax2.set_title("Users by Segment")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.subheader("Average Risk by Segment")

    avg_risk = (
        df.groupby("segment_name")["risk_probability"]
        .mean()
        .sort_values(ascending=False)
    )

    st.dataframe(avg_risk.to_frame("Average Risk Probability"))

# -----------------------------
# Tab 3: Model Insights
# -----------------------------
with tab3:
    st.subheader("Project Summary")

    st.write("""
    This platform analyzes Netflix-style user behavior data to identify engagement risk,
    segment users, and recommend business actions.
    """)

    st.markdown("""
    **End-to-End ML Pipeline:**
    - Raw data ingestion
    - Data cleaning
    - Feature engineering
    - User master table creation
    - Engagement risk classification
    - K-Fold cross validation
    - Random Forest model training
    - KMeans user segmentation
    - Risk scoring engine
    - Business explanation layer
    - Streamlit dashboard
    """)

    st.subheader("Model Performance")

    st.markdown("""
    **Final validated model:** Random Forest  
    **Accuracy:** 80.0%  
    **Precision:** 84.0%  
    **Recall:** 81.4%  
    **F1 Score:** 82.7%  
    **ROC-AUC:** 86.7%
    """)

    st.subheader("Business Value")

    st.markdown("""
    This system helps a streaming platform:
    - Identify users at engagement risk
    - Understand why users may disengage
    - Segment users into behavioral groups
    - Recommend targeted business actions
    - Support retention and personalization strategy
    """)