import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import io
from prophet.diagnostics import performance_metrics
from prophet.models import StanBackendEnum

# -----------------------------------
# STREAMLIT APP CONFIG
# -----------------------------------
st.set_page_config(page_title="📈 Stock Prophet Forecast Dashboard", layout="wide")

st.title("📈 Stock Prophet Forecast Dashboard")
st.markdown("""
Upload your Excel file with a **date/time column** and a **numeric stock price column**.
This dashboard trains a Prophet model and shows:
- Historical Stock Price Trends  
- Model Performance Metrics  
- 5-Year Long-Term Forecast  
- Short-Term (Days to Months) Forecast
""")

# -----------------------------------
# FILE UPLOAD
# -----------------------------------
uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded {uploaded_file.name} with {len(df)} rows")

        # Column selection
        target_col = st.selectbox("Select stock price column", [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)])
        date_col = st.selectbox("Select date/time column", [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64) or 'date' in c.lower() or 'time' in c.lower()])

        df = df[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"}).dropna()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds")

        # Sidebar config
        st.sidebar.header("⚙️ Model Settings")
        yearly = st.sidebar.checkbox("Yearly seasonality", True)
        weekly = st.sidebar.checkbox("Weekly seasonality", True)
        daily = st.sidebar.checkbox("Daily seasonality", False)
        test_size = st.sidebar.slider("Test size (% of last rows)", 5, 50, 10)

        # Prophet model
        st.sidebar.subheader("🔮 Forecast Settings")
        forecast_freq = st.sidebar.text_input("Data frequency (e.g., D, M)", "D")

        # Train model
        st.info("Training Prophet model... Please wait ⏳")
        split_idx = int(len(df) * (1 - test_size / 100))
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

        # Force Prophet to use cmdstanpy backend globally
        model = Prophet(yearly_seasonality=yearly, weekly_seasonality=weekly, daily_seasonality=daily)
        model.stan_backend = StanBackendEnum.CMDSTANPY

        # Fit model (no backend argument here!)
        model.fit(train_df)

        # Predict on test data
        future = model.make_future_dataframe(periods=len(test_df), freq=forecast_freq)
        forecast = model.predict(future)

        # Calculate metrics
        y_true = test_df["y"].values
        y_pred = forecast["yhat"].iloc[-len(test_df):].values
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # -----------------------------------
        # MULTI-TAB DASHBOARD
        # -----------------------------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Stock Overview",
            "🤖 Model & Metrics",
            "📈 5-Year Long-Term Forecast",
            "📅 Short-Term Forecast"
        ])

        # -----------------
        # TAB 1: STOCK OVERVIEW
        # -----------------
        with tab1:
            st.subheader("Historical Stock Price Overview")
            st.line_chart(df.set_index("ds")["y"])
            st.write("**Basic Stats:**")
            st.dataframe(df.describe())

        # -----------------
        # TAB 2: MODEL & METRICS
        # -----------------
        with tab2:
            st.subheader("Model Performance & Explanation")
            st.markdown(f"""
            - **MAE:** {mae:.2f}  
            - **RMSE:** {rmse:.2f}  
            - **Train Size:** {len(train_df)}  
            - **Test Size:** {len(test_df)}  
            - **Model:** Prophet (Trend + Seasonality)
            """)
            st.markdown("""
            ### 🔍 How Prophet Helps:
            Prophet is designed for time-series with strong seasonal effects and trends.
            It automatically detects:
            - Daily, weekly, and yearly patterns  
            - Long-term growth trends  
            - Holidays and anomalies (if configured)
            """)

            st.subheader("Forecast vs Actual (Test Period)")
            test_fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(test_df["ds"], test_df["y"], label="Actual", color="orange")
            ax.plot(test_df["ds"], forecast["yhat"].iloc[-len(test_df):], label="Predicted", color="green", linestyle="--")
            ax.legend()
            st.pyplot(test_fig)

        # -----------------
        # TAB 3: LONG-TERM FORECAST
        # -----------------
        with tab3:
            st.subheader("📈 Prophet 5-Year Forecast")
            long_future = model.make_future_dataframe(periods=1825, freq="D")
            long_forecast = model.predict(long_future)
            st.plotly_chart(plot_plotly(model, long_forecast), use_container_width=True)

            out = io.BytesIO()
            long_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_excel(out, index=False)
            st.download_button("📥 Download 5-Year Forecast", out.getvalue(), "5_year_forecast.xlsx")

        # -----------------
        # TAB 4: SHORT-TERM FORECAST
        # -----------------
        with tab4:
            st.subheader("📅 Custom Short-Term Forecast")
            short_days = st.slider("Forecast Days Ahead", 7, 365, 90)
            short_future = model.make_future_dataframe(periods=short_days, freq=forecast_freq)
            short_forecast = model.predict(short_future)

            st.plotly_chart(plot_plotly(model, short_forecast), use_container_width=True)

            st.write(f"**Forecasting next {short_days} {forecast_freq}-periods.**")
            short_out = io.BytesIO()
            short_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_excel(short_out, index=False)
            st.download_button("📥 Download Short-Term Forecast", short_out.getvalue(), "short_term_forecast.xlsx")

    except Exception as e:
        st.error(f"❌ Error: {e}")
