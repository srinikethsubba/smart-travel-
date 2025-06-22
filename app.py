import streamlit as st
import pandas as pd
import sqlite3
import joblib
import matplotlib.pyplot as plt
import datetime

st.title("Smart Travel Planner & Price Predictor")
st.markdown("Built by Sriniketh")

model = joblib.load("models/price_predictor.joblib")

conn = sqlite3.connect("db/travel.sqlite")
df = pd.read_sql("SELECT * FROM hotel_prices", conn)
conn.close()

df["date"] = pd.to_datetime(df["date"])

st.sidebar.title("Filter Options")
selected_hotel = st.sidebar.selectbox("Select Hotel Type", df["hotel"].unique())
date_range = st.sidebar.date_input("Select Date Range", [df["date"].min(), df["date"].max()])
predict_date = st.sidebar.date_input("Predict Price For Date", value=datetime.date.today(), min_value=datetime.date.today())

filtered_df = df[
   (df["hotel"] == selected_hotel) &
   (df["date"] >= pd.to_datetime(date_range[0])) &
   (df["date"] <= pd.to_datetime(date_range[1]))
]

st.subheader("Hotel Prices Over Time")
st.write(f"Showing **{selected_hotel}** prices from {date_range[0]} to {date_range[1]}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(filtered_df["date"], filtered_df["price"], marker="o")
ax.set_title("Hotel Prices")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
st.pyplot(fig) 

timestamp = pd.to_datetime(predict_date).value // 10**9
predicted_price = model.predict([[timestamp]])[0]

st.write(f"selected day: {predict_date.strftime('%A')}")

st.subheader("Predicted Price")
st.write(f"Estimated price for **{selected_hotel}** on {predict_date}: **${predicted_price:.2f}**")

st.markdown("Note: Chart shows past data (2015-2017), but predictions support any future date")
