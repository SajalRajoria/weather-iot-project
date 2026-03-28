import streamlit as st
import pandas as pd
import time
import os
import numpy as np
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="Weather Dashboard", layout="wide")

st.title("🌦️ Smart IoT Weather Monitoring Dashboard")

# Placeholder for auto refresh
placeholder = st.empty()

while True:
    with placeholder.container():
        try:
            # ✅ Check if file exists
            if not os.path.exists("weather_data.csv"):
                st.warning("⚠️ No data found. Run weather_iot.py first.")
            else:
                # Load data
                data = pd.read_csv("weather_data.csv")

                # Convert time column
                data["Time"] = pd.to_datetime(data["Time"])
                data.set_index("Time", inplace=True)

                # 🟢 Latest values
                latest_temp = data["Temperature"].iloc[-1]
                latest_hum = data["Humidity"].iloc[-1]

                # 📊 Metrics
                col1, col2 = st.columns(2)
                col1.metric("🌡️ Temperature", f"{latest_temp:.2f} °C")
                col2.metric("💧 Humidity", f"{latest_hum:.2f} %")

                # 📈 Graphs
                st.subheader("📈 Temperature Trend")
                st.line_chart(data["Temperature"])

                st.subheader("💧 Humidity Trend")
                st.line_chart(data["Humidity"])

                # 🤖 ML Prediction
                st.subheader("🤖 Temperature Prediction")

                df = data.reset_index()

                # Convert time to numeric
                df["Time_num"] = df["Time"].map(pd.Timestamp.timestamp)

                X = df["Time_num"].values.reshape(-1, 1)
                y = df["Temperature"].values

                if len(X) > 5:  # ensure enough data
                    model = LinearRegression()
                    model.fit(X, y)

                    # Predict next 60 seconds
                    future_time = np.array([[X[-1][0] + 60]])
                    predicted_temp = model.predict(future_time)

                    st.success(f"🔮 Predicted Temperature (Next Minute): {predicted_temp[0]:.2f} °C")
                else:
                    st.info("Collecting more data for prediction...")

        except Exception as e:
            st.error(f"❌ Error: {e}")

    time.sleep(5)  # refresh every 5 seconds