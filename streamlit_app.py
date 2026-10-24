import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from features import forecast

st.set_page_config(page_title="Energy Forecaster", page_icon="⚡")
st.title("⚡ Energy Consumption Forecaster")
st.write("By ~ Ashish Ranjan.....................")
st.write("Forecasts hourly household electricity use with a LightGBM model.")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


model = load_model()
sample = pd.read_csv("sample_history.csv", index_col=0, parse_dates=True)["kwh"].asfreq("h")

uploaded = st.file_uploader("Optional: your own hourly CSV (columns: timestamp, kwh)", type="csv")
horizon = st.slider("Hours to forecast", 1, 72, 24)

history = sample
if uploaded is not None:
    history = pd.read_csv(uploaded, index_col=0, parse_dates=True)["kwh"].asfreq("h").interpolate()

if len(history) < 200:
    st.error("Need at least 200 hourly rows of history.")
else:
    pred = forecast(model, history, horizon)

    fig, ax = plt.subplots(figsize=(9, 4))
    history.iloc[-168:].plot(ax=ax, label="Recent usage (last 7 days)")
    pred.plot(ax=ax, label="Forecast")
    ax.set_ylabel("kWh")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    table = pred.round(3).rename_axis("timestamp").reset_index()
    table["timestamp"] = table["timestamp"].astype(str)
    st.dataframe(table)
