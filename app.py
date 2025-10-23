import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Prophet Forecast App (Excel-Compatible)", layout="wide")

st.title("📈 Prophet Forecast App (Excel-Compatible)")
st.markdown("Upload an Excel file with a date/time column and a numeric target column. "
            "The app will train a Facebook Prophet model, show forecasts, components, and evaluation metrics.")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded {uploaded_file.name} with {len(df)} rows")
        st.dataframe(df.head())

        target_col = st.selectbox("Select target column", [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)])
        date_col = st.selectbox("Select date/time column", [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64) or 'date' in c.lower() or 'time' in c.lower()])

        # Settings
        st.sidebar.header("Model Settings")
        forecast_periods = st.sidebar.number_input("Forecast periods (steps ahead)", value=30, step=1)
        freq = st.sidebar.text_input("Frequency (e.g. D, M, H)", value="D")
        test_size = st.sidebar.slider("Test size (% of last rows)", min_value=5, max_value=50, value=10)

        run_forecast = st.button("🚀 Run Forecast")

        if run_forecast:
            df = df[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"}).dropna()
            df["ds"] = pd.to_datetime(df["ds"])
            df = df.sort_values("ds")

            if len(df) > 5000:
                st.warning("Large dataset detected. Sampling 5000 rows for faster training.")
                df = df.sample(5000).sort_values("ds")

            split_idx = int(len(df) * (1 - test_size / 100))
            train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(train_df)

            future = model.make_future_dataframe(periods=forecast_periods, freq=freq)
            forecast = model.predict(future)

            # Plot forecast
            st.subheader("Forecast Plot")
            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            # Plot components
            st.subheader("Forecast Components")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)

            # Evaluate
            y_true = test_df["y"].values
            y_pred = forecast["yhat"].iloc[-len(test_df):].values
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            st.write(f"**MAE:** {mae:.2f} | **RMSE:** {rmse:.2f}")

            # Download results
            out = io.BytesIO()
            forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_excel(out, index=False)
            st.download_button("📥 Download Forecast Results (Excel)", out.getvalue(), "forecast_results.xlsx")

    except Exception as e:
        st.error(f"❌ Error: {e}")

