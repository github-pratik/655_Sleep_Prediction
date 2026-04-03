#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd
import streamlit as st
import joblib
import altair as alt

MODEL_PATH = Path("models/mobile_champion.pkl") if Path("models/mobile_champion.pkl").exists() else Path("models/rf.pkl")
DATA_PATH = Path("dataset/model_data.csv")
TARGET_COL = "fatigue_label"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "label_date" in df.columns:
        df["label_date"] = pd.to_datetime(df["label_date"]).dt.date
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict(df_features, model):
    feature_cols = [c for c in df_features.columns if c != TARGET_COL]
    preds = model.predict(df_features[feature_cols])
    proba = model.predict_proba(df_features[feature_cols]) if hasattr(model, "predict_proba") else None
    return preds, proba


def main():
    st.title("Next-Day Fatigue Prediction")
    st.write("Uses Apple Watch sleep + physiology features with a mobile-focused prediction model.")

    df = load_data()
    model = load_model()
    model_name = type(model.named_steps["model"]).__name__ if hasattr(model, "named_steps") else type(model).__name__
    st.caption(f"Loaded model: `{MODEL_PATH}` ({model_name})")

    feature_cols = [c for c in df.columns if c != TARGET_COL]

    st.subheader("Upload custom model_data.csv (optional)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "label_date" in df.columns:
            df["label_date"] = pd.to_datetime(df["label_date"]).dt.date
        st.success("Custom data loaded")

    st.subheader("Select date for prediction")
    date_options = df["label_date" if "label_date" in df.columns else "night_date"].astype(str).unique()
    chosen_date = st.selectbox("Label date", sorted(date_options))
    row = df[(df["label_date"].astype(str) == chosen_date) if "label_date" in df.columns else (df["night_date"].astype(str) == chosen_date)]
    preds, proba = predict(row, model)
    st.metric("Prediction", preds[0])
    if proba is not None:
        st.write("Class probabilities:")
        st.json({str(cls): float(p) for cls, p in zip(model.classes_, proba[0])})

    st.subheader("Fatigue over time")
    chart_df = df.copy()
    chart_df["date"] = chart_df["label_date" if "label_date" in df.columns else "night_date"]
    chart_df["prediction"], _ = predict(chart_df, model)
    base = alt.Chart(chart_df).mark_line(point=True).encode(
        x="date:T",
        y="prediction:Q",
        color=alt.value("#4c78a8"),
        tooltip=["date", "prediction"]
    )
    st.altair_chart(base, use_container_width=True)

    st.subheader("Top factors (RF feature importances if available)")
    fi_path = Path("reports/feature_importance.png")
    if fi_path.exists():
        st.image(str(fi_path))
    else:
        st.info("Feature importance plot not found (only available for Random Forest).")


if __name__ == "__main__":
    main()
