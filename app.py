import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------
# Streamlit page config
# ------------------------------
st.set_page_config(page_title="📈 Stock Prophet Forecast Dashboard", layout="wide")
st.title("📈 Stock Prophet Forecast Dashboard")
st.caption("Upload an Excel file, clean & explore the data, engineer lag features, then forecast with Prophet.")

# ------------------------------
# Helpers
# ------------------------------

@st.cache_data(show_spinner=False)
def load_excel(file, sheet_name=None):
    data = pd.read_excel(file, sheet_name=sheet_name if sheet_name else 0)
    # if user forgot to specify sheet_name and there are multiple sheets
    if isinstance(data, dict):
        first_sheet = list(data.keys())[0]
        st.warning(f"Multiple sheets detected. Using first sheet: '{first_sheet}'")
        data = data[first_sheet]
    return data

def coerce_datetime(df, col):
    ser = pd.to_datetime(df[col], errors="coerce")
    return ser

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)[: len(y_true)]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-12, None))) * 100.0)
    return mae, rmse, mape

def build_lags(df, target_col, max_lag=5):
    out = df.copy()
    for k in range(1, max_lag + 1):
        out[f"{target_col}_lag{k}"] = out[target_col].shift(k)
    return out

def df_to_excel_bytes(df):
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf

# Keep state across tabs
ss = st.session_state
if "df_raw" not in ss: ss.df_raw = None
if "df_clean" not in ss: ss.df_clean = None
if "df_final" not in ss: ss.df_final = None
if "date_col" not in ss: ss.date_col = None
if "target_col" not in ss: ss.target_col = None

# ------------------------------
# File upload + column selection
# ------------------------------
st.sidebar.header("📤 Upload & Settings")
uploaded = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"])
sheet = st.sidebar.text_input("Sheet name (optional)", value="")
test_pct = st.sidebar.slider("Test size (% of last rows)", 5, 50, 10)
yearly = st.sidebar.checkbox("Yearly seasonality", True)
weekly = st.sidebar.checkbox("Weekly seasonality", True)
daily = st.sidebar.checkbox("Daily seasonality", False)
freq = st.sidebar.text_input("Data frequency (e.g., D, M)", "D")

# Tabs (4 for EDA/cleaning + 4 for forecasting)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🧹 Raw & Missing",
    "🧯 Correlation Heatmap",
    "🧩 Lag Features (1–5)",
    "✅ Final Prepared Data",
    "📊 Stock Overview",
    "🤖 Model & Metrics",
    "📈 5-Year Forecast",
    "📅 Short-Term Forecast",
])

# ------------------------------
# Tab 1: Raw & Missing
# ------------------------------
with tab1:
    st.subheader("🧹 Raw Data & Missing Values")
    if uploaded:
        try:
            df = load_excel(uploaded, sheet_name=sheet if sheet.strip() else None)
            ss.df_raw = df.copy()
            st.success(f"Loaded **{uploaded.name}** with **{len(df):,}** rows and **{df.shape[1]}** columns.")
            st.dataframe(df.head(50), use_container_width=True)

            # Guess columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            dt_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            dt_detected = []
            for c in df.columns:
                # Try actual datetime type or convertible
                try:
                    if pd.api.types.is_datetime64_any_dtype(df[c]) or pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.75:
                        dt_detected.append(c)
                except Exception:
                    pass
            dt_suggestions = list(dict.fromkeys(dt_candidates + dt_detected))  # preserve order, unique

            ss.target_col = st.selectbox("Select target (stock price) column", numeric_cols or df.columns.tolist(), index=0 if numeric_cols else 0)
            ss.date_col = st.selectbox("Select date/time column", dt_suggestions or df.columns.tolist(), index=0)

            # Missing summary
            st.markdown("### Missing Values (count per column)")
            miss = df.isna().sum().to_frame("missing_count")
            st.dataframe(miss, use_container_width=True)

            # Cleaning options
            st.markdown("### Cleaning")
            method = st.radio("Choose NaN handling", ["Drop rows with any NaN", "Forward fill (ffill)", "Backward fill (bfill)", "Fill NaN with 0"], index=0, horizontal=True)

            apply_btn = st.button("Apply Cleaning")
            if apply_btn:
                df2 = df.copy()
                # Ensure datetime for selected date column
                df2[ss.date_col] = coerce_datetime(df2, ss.date_col)
                df2 = df2.sort_values(ss.date_col)
                # Apply missing strategy
                if method == "Drop rows with any NaN":
                    df2 = df2.dropna()
                elif method == "Forward fill (ffill)":
                    df2 = df2.ffill()
                elif method == "Backward fill (bfill)":
                    df2 = df2.bfill()
                else:
                    df2 = df2.fillna(0)
                # Drop rows where date is NaT
                df2 = df2[df2[ss.date_col].notna()]
                ss.df_clean = df2.reset_index(drop=True)
                st.success(f"Cleaned data has **{len(ss.df_clean):,}** rows.")
                st.dataframe(ss.df_clean.head(50), use_container_width=True)

                # Download cleaned
                st.download_button(
                    "📥 Download Cleaned Excel",
                    data=df_to_excel_bytes(ss.df_clean),
                    file_name="cleaned_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except Exception as e:
            st.error(f"Failed to read Excel: {e}")
    else:
        st.info("Upload an Excel file to begin.")

# ------------------------------
# Tab 2: Correlation Heatmap
# ------------------------------
with tab2:
    st.subheader("🧯 Correlation Heatmap (Numeric Columns)")
    if ss.df_clean is not None:
        num = ss.df_clean.select_dtypes(include=[np.number])
        if num.shape[1] >= 2:
            corr = num.corr(numeric_only=True)
            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap",
                aspect="auto",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least two numeric columns for a heatmap.")
    else:
        st.info("Please prepare cleaned data in Tab 1 first.")

# ------------------------------
# Tab 3: Lag Features (1–5)
# ------------------------------
with tab3:
    st.subheader("🧩 Build Lag Features (1–5)")
    if ss.df_clean is not None and ss.target_col and ss.date_col:
        max_lag = st.slider("Max lag (k)", 1, 10, 5)
        build_btn = st.button("Build Lag Features")
        if build_btn:
            df = ss.df_clean.copy()
            # Keep only date + numeric (to avoid trouble)
            if ss.target_col not in df.columns:
                st.error(f"Target column '{ss.target_col}' not found in cleaned data.")
            else:
                # Ensure datetime index for convenience
                df = df.sort_values(ss.date_col)
                df_lagged = build_lags(df[[ss.date_col, ss.target_col]], ss.target_col, max_lag=max_lag)
                # Drop initial NaN rows introduced by shifting
                df_lagged = df_lagged.dropna().reset_index(drop=True)
                ss.df_final = df_lagged.rename(columns={ss.date_col: "ds", ss.target_col: "y"})
                # Keep original date column name for reference if needed elsewhere
                st.success(f"Lag features created up to lag {max_lag}. Final rows: {len(ss.df_final):,}")
                st.dataframe(ss.df_final.head(50), use_container_width=True)

                st.download_button(
                    "📥 Download Lagged Final Excel",
                    data=df_to_excel_bytes(ss.df_final),
                    file_name="final_lagged_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
    else:
        st.info("Make sure cleaned data exists and target/date columns are selected in Tab 1.")

# ------------------------------
# Tab 4: Final Prepared Data
# ------------------------------
with tab4:
    st.subheader("✅ Final Prepared Data (used for forecasting)")
    if ss.df_final is not None:
        st.dataframe(ss.df_final.head(100), use_container_width=True)
        st.write(f"Shape: {ss.df_final.shape[0]:,} rows × {ss.df_final.shape[1]} columns")
        st.download_button(
            "📥 Download Final Prepared Excel",
            data=df_to_excel_bytes(ss.df_final),
            file_name="final_prepared_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Build lag features in Tab 3 to finalize the dataset.")

# ------------------------------
# Tab 5: Stock Overview
# ------------------------------
with tab5:
    st.subheader("📊 Stock Overview")
    src = ss.df_final if ss.df_final is not None else (ss.df_clean if ss.df_clean is not None else ss.df_raw)
    if src is not None:
        # pick price column (y if final exists, else target_col)
        if "y" in src.columns and "ds" in src.columns:
            plot_df = src[["ds", "y"]].rename(columns={"ds": "Date", "y": "Price"})
        elif ss.target_col and ss.date_col and ss.target_col in src.columns and ss.date_col in src.columns:
            plot_df = src[[ss.date_col, ss.target_col]].rename(columns={ss.date_col: "Date", ss.target_col: "Price"})
        else:
            st.warning("Could not find a suitable date/price pair to plot.")
            plot_df = None

        if plot_df is not None:
            fig = px.line(plot_df, x="Date", y="Price", title="Historical Stock Price")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("### Basic Stats")
            st.dataframe(plot_df["Price"].describe().to_frame(), use_container_width=True)
    else:
        st.info("Upload and prepare data first.")

# ------------------------------
# Tab 6: Model & Metrics
# ------------------------------
with tab6:
    st.subheader("🤖 Model & Metrics (Prophet)")
    data_for_model = ss.df_final
    if data_for_model is None:
        st.info("Using cleaned data (no lags). For best results, finalize data in Tab 3.")
        # Fallback to cleaned
        if ss.df_clean is not None and ss.date_col and ss.target_col:
            temp = ss.df_clean.copy()
            temp = temp[[ss.date_col, ss.target_col]].dropna()
            temp = temp.rename(columns={ss.date_col: "ds", ss.target_col: "y"})
            temp["ds"] = pd.to_datetime(temp["ds"], errors="coerce")
            temp = temp.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
            data_for_model = temp

    if data_for_model is not None and {"ds", "y"}.issubset(data_for_model.columns):
        # Train/Test split
        split_idx = int(len(data_for_model) * (1 - test_pct / 100))
        train_df = data_for_model.iloc[:split_idx].copy()
        test_df = data_for_model.iloc[split_idx:].copy()

        # Prophet model
        with st.spinner("Training Prophet model..."):
            model = Prophet(
                yearly_seasonality=yearly,
                weekly_seasonality=weekly,
                daily_seasonality=daily
            )
            model.fit(train_df)

        # Predict on test horizon only
        future = model.make_future_dataframe(periods=len(test_df), freq=freq)
        forecast = model.predict(future)

        # Compute metrics on overlapping test period
        y_true = test_df["y"].values
        y_pred = forecast["yhat"].iloc[-len(test_df):].values
        mae, rmse, mape = metrics(y_true, y_pred)

        st.success(f"**MAE:** {mae:.4f} | **RMSE:** {rmse:.4f} | **MAPE:** {mape:.2f}%")
        st.markdown("**Train size:** {:,} | **Test size:** {:,}".format(len(train_df), len(test_df)))

        # Plot: test vs predicted
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_df["ds"], y=y_true, name="Actual (Test)", mode="lines"))
        fig.add_trace(go.Scatter(x=test_df["ds"], y=y_pred, name="Predicted", mode="lines"))
        fig.update_layout(title="Test Period: Actual vs Predicted", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # Store model & last forecast for other tabs
        ss._model_trained = model
        ss._last_data = data_for_model
    else:
        st.info("Prepare your final dataset in Tab 3 or at least clean data in Tab 1.")

# ------------------------------
# Tab 7: 5-Year Forecast
# ------------------------------
with tab7:
    st.subheader("📈 5-Year Long-Term Forecast")
    if "_model_trained" in ss and ss._model_trained is not None and ss._last_data is not None:
        model = ss._model_trained
        # 5 years ≈ 1825 days
        future = model.make_future_dataframe(periods=1825, freq="D")
        long_fcst = model.predict(future)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ss._last_data["ds"], y=ss._last_data["y"], name="History", mode="lines"))
        fig.add_trace(go.Scatter(x=long_fcst["ds"], y=long_fcst["yhat"], name="Forecast", mode="lines"))
        fig.add_trace(go.Scatter(
            x=long_fcst["ds"], y=long_fcst["yhat_upper"], name="Upper", mode="lines",
            line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=long_fcst["ds"], y=long_fcst["yhat_lower"], name="Lower", mode="lines",
            fill="tonexty", line=dict(width=0), showlegend=False
        ))
        fig.update_layout(title="5-Year Forecast", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # Download
        st.download_button(
            "📥 Download 5-Year Forecast (Excel)",
            data=df_to_excel_bytes(long_fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]]),
            file_name="forecast_5y.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Train the model in **Model & Metrics** tab first.")

# ------------------------------
# Tab 8: Short-Term Forecast (Days–Months)
# ------------------------------
with tab8:
    st.subheader("📅 Short-Term Forecast (Days–Months)")
    if "_model_trained" in ss and ss._model_trained is not None and ss._last_data is not None:
        horizon = st.slider("Horizon (days)", 7, 365, 90)
        model = ss._model_trained
        future = model.make_future_dataframe(periods=horizon, freq=freq)
        short_fcst = model.predict(future)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ss._last_data["ds"], y=ss._last_data["y"], name="History", mode="lines"))
        fig.add_trace(go.Scatter(x=short_fcst["ds"], y=short_fcst["yhat"], name="Forecast", mode="lines"))
        fig.add_trace(go.Scatter(
            x=short_fcst["ds"], y=short_fcst["yhat_upper"], name="Upper", mode="lines",
            line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=short_fcst["ds"], y=short_fcst["yhat_lower"], name="Lower", mode="lines",
            fill="tonexty", line=dict(width=0), showlegend=False
        ))
        fig.update_layout(title=f"Short-Term Forecast ({horizon} days)", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "📥 Download Short-Term Forecast (Excel)",
            data=df_to_excel_bytes(short_fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]]),
            file_name=f"forecast_{horizon}d.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Train the model in **Model & Metrics** tab first.")
