import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from prophet import Prophet

# App title
st.title("📊 Retail Sales Forecasting Dashboard")

# Load data
df = pd.read_csv("retail_sales.csv")

# Convert columns
df['data'] = pd.to_datetime(df['data'])
df = df[['data', 'venda']]
df = df.rename(columns={'data': 'ds', 'venda': 'y'})

# Show raw data
st.subheader("📂 Raw Sales Data")
st.dataframe(df.head())

# Plot historical sales
st.subheader("📈 Historical Sales Trend")
fig1, ax1 = plt.subplots()
ax1.plot(df['ds'], df['y'])
ax1.set_xlabel("Date")
ax1.set_ylabel("Sales")
st.pyplot(fig1)

# Train model
model = Prophet()
model.fit(df)

# Forecast period selector
periods = st.slider("Select number of days to forecast:", 7, 90, 30)

future = model.make_future_dataframe(periods=periods)
forecast = model.predict(future)

# Forecast plot
st.subheader("🔮 Sales Forecast")
fig2 = model.plot(forecast)
st.pyplot(fig2)

# Components
st.subheader("📊 Trend & Seasonality")
fig3 = model.plot_components(forecast)
st.pyplot(fig3)

st.success("✅ Forecast generated successfully!")
