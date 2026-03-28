import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# 🔁 Auto refresh every 10 sec
st_autorefresh(interval=10000, key="refresh")

# Page config
st.set_page_config(page_title="Advanced Weather App", layout="wide")

st.title("🌦️ Advanced Live Weather Dashboard")

# 🔑 API KEY
API_KEY = "e5dd5925b2e785093d814134ba52c691"

# 🌍 City selector
city = st.selectbox("📍 Select City", ["Palampur", "Shimla", "New Delhi", "Patiala"])

# 🌐 API URL
url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"

# Fetch data
data = requests.get(url).json()

# Session state for storing history
if "weather_data" not in st.session_state:
    st.session_state.weather_data = pd.DataFrame(columns=["Time", "Temperature", "Humidity"])

if "main" in data:
    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]

    # Add new data point
    new_row = pd.DataFrame({
        "Time": [datetime.now()],
        "Temperature": [temp],
        "Humidity": [humidity]
    })

    st.session_state.weather_data = pd.concat(
        [st.session_state.weather_data, new_row],
        ignore_index=True
    )

    df = st.session_state.weather_data

    # 📊 Metrics
    col1, col2 = st.columns(2)
    col1.metric("🌡️ Temperature", f"{temp:.2f} °C")
    col2.metric("💧 Humidity", f"{humidity:.2f} %")

    st.divider()

    # 📈 Graphs
    st.subheader("📈 Temperature Trend")
    st.line_chart(df.set_index("Time")["Temperature"])

    st.subheader("💧 Humidity Trend")
    st.line_chart(df.set_index("Time")["Humidity"])

    st.divider()

    # 🤖 ML Prediction
    st.subheader("🤖 AI Temperature Prediction")

    if len(df) > 5:
        df["Time_num"] = df["Time"].map(pd.Timestamp.timestamp)

        X = df["Time_num"].values.reshape(-1, 1)
        y = df["Temperature"].values

        model = LinearRegression()
        model.fit(X, y)

        future_time = np.array([[X[-1][0] + 60]])
        predicted_temp = model.predict(future_time)

        st.success(f"🔮 Predicted Temp (Next Minute): {predicted_temp[0]:.2f} °C")
    else:
        st.info("Collecting data for prediction...")

    st.divider()

    # 🚨 Alerts
    st.subheader("🚨 Weather Alerts")

    if temp > 35:
        st.error("🔥 High Temperature Alert!")
    elif temp < 5:
        st.warning("❄️ Cold Weather Alert!")
    else:
        st.success("✅ Weather Normal")

    # 📋 Data table
    with st.expander("📋 Show Data"):
        st.dataframe(df.tail(20))

else:
    st.error("❌ Error fetching weather data")