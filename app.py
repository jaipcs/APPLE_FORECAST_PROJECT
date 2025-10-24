# ============================================================
# 📈 Stock Prophet Forecast Dashboard
# Cleaned & Colab-ready version (Streamlit)
# ============================================================


import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ------------------------------ #
# Streamlit page config
# ------------------------------ #
st.set_page_config(page_title="📈 Stock Prophet Forecast Dashboard", layout="wide")
st.title("📈 Stock Prophet Forecast Dashboard")
st.caption("Upload an Excel file → clean & explore → engineer lag features → forecast with Prophet.")


# ------------------------------ #
# Helpers
# ------------------------------ #
@st.cache_data(show_spinner=False)
def load_excel(file, sheet_name=None):
    data = pd.read_excel(file, sheet_name=sheet_name if sheet_name else 0)
    if isinstance(data, dict):
        # multiple sheets detected
        first_sheet = list(data.keys())[0]
        st.warning(f"Multiple sheets detected — using first sheet: {first_sheet}")
        data = data[first_sheet]
    return data

    
def coerce_datetime(df, col):
    return pd.to_datetime(df[col], errors="coerce")

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)[: len(y_true)]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = float(np.mean(np.abs((y_true - y_pred) /
               np.clip(np.abs(y_true), 1e-12, None))) * 100.0)
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

# ------------------------------ #
# Session state
# ------------------------------ #
ss = st.session_state
for k in ["df_raw","df_clean","df_final","date_col","target_col"]:
    ss.setdefault(k, None)

# ------------------------------ #
# Sidebar
# ------------------------------ #
st.sidebar.header("📤 Upload & Settings")
uploaded = st.sidebar.file_uploader("Upload Excel file", type=["xlsx","xls"])
sheet = st.sidebar.text_input("Sheet name (optional)", value="")
test_pct = st.sidebar.slider("Test size (% of last rows)", 5, 50, 10)
yearly = st.sidebar.checkbox("Yearly seasonality", True)
weekly = st.sidebar.checkbox("Weekly seasonality", True)
daily = st.sidebar.checkbox("Daily seasonality", False)
freq = st.sidebar.text_input("Data frequency (e.g., D, M)", "D")

# ------------------------------ #
# Tabs
# ------------------------------ #
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

# ============================================================
# TAB 1 — Raw & Missing
# ============================================================
with tab1:
    st.subheader("🧹 Raw Data & Missing Values")
    if uploaded:
        try:
            df = load_excel(uploaded, sheet_name=sheet if sheet.strip() else None)
            ss.df_raw = df.copy()
            st.success(f"Loaded **{uploaded.name}** — {len(df):,} rows × {df.shape[1]} columns.")
            st.dataframe(df.head(50), width='stretch')

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            dt_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            dt_detected = [c for c in df.columns if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.75]
            dt_suggestions = list(dict.fromkeys(dt_candidates + dt_detected))

            ss.target_col = st.selectbox("Select target (price) column", numeric_cols or df.columns.tolist())
            ss.date_col = st.selectbox("Select date/time column", dt_suggestions or df.columns.tolist())

            st.markdown("### Missing Values (count per column)")
            miss = df.isna().sum().to_frame("missing_count")
            st.dataframe(miss, width='stretch')

            st.markdown("### Cleaning")
            method = st.radio(
                "NaN handling",
                ["Drop rows", "Forward fill", "Backward fill", "Fill with 0"],
                index=0, horizontal=True)

            if st.button("Apply Cleaning"):
                df2 = df.copy()
                df2[ss.date_col] = coerce_datetime(df2, ss.date_col)
                df2 = df2.sort_values(ss.date_col)
                if method == "Drop rows": df2 = df2.dropna()
                elif method == "Forward fill": df2 = df2.ffill()
                elif method == "Backward fill": df2 = df2.bfill()
                else: df2 = df2.fillna(0)
                df2 = df2[df2[ss.date_col].notna()].reset_index(drop=True)
                ss.df_clean = df2
                st.success(f"Cleaned data — {len(df2):,} rows.")
                st.dataframe(df2.head(50), width='stretch')
                st.download_button("📥 Download Cleaned Excel",
                    data=df_to_excel_bytes(df2),
                    file_name="cleaned_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"❌ Failed to read Excel: {e}")
    else:
        st.info("Upload an Excel file to begin.")

# ============================================================
# TAB 2 — Correlation Heatmap
# ============================================================
with tab2:
    st.subheader("🧯 Correlation Heatmap")
    if ss.df_clean is not None:
        num = ss.df_clean.select_dtypes(include=[np.number])
        if num.shape[1] >= 2:
            corr = num.corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=True,
                            color_continuous_scale="RdBu_r",
                            title="Correlation Heatmap")
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Need ≥2 numeric columns.")
    else:
        st.info("Clean data first in Tab 1.")

# ============================================================
# TAB 3 — Lag Features
# ============================================================
with tab3:
    st.subheader("🧩 Build Lag Features")
    if ss.df_clean is not None and ss.target_col and ss.date_col:
        k = st.slider("Max lag", 1, 10, 5)
        if st.button("Build Lags"):
            df = ss.df_clean.copy()
            df = df.sort_values(ss.date_col)
            lagged = build_lags(df[[ss.date_col, ss.target_col]], ss.target_col, k).dropna().reset_index(drop=True)
            ss.df_final = lagged.rename(columns={ss.date_col: "ds", ss.target_col: "y"})
            st.success(f"Created lag 1-{k}. Rows: {len(ss.df_final):,}")
            st.dataframe(ss.df_final.head(50), width='stretch')
            st.download_button("📥 Download Lagged Excel",
                data=df_to_excel_bytes(ss.df_final),
                file_name="final_lagged_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Prepare cleaned data first.")

# ============================================================
# TAB 4 — Final Data
# ============================================================
with tab4:
    st.subheader("✅ Final Prepared Data")
    if ss.df_final is not None:
        st.dataframe(ss.df_final.head(100), width='stretch')
        st.write(f"Shape: {ss.df_final.shape}")
        st.download_button("📥 Download Final Excel",
            data=df_to_excel_bytes(ss.df_final),
            file_name="final_prepared_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Build lags in Tab 3.")

# ============================================================

# Tab 5: Stock Overview
# ------------------------------
with tab5:
    st.subheader("📊 Stock Overview")

    # ✅ Guard: prevent multi-sheet dict issue
    if isinstance(ss.df_raw, dict):
        st.error("❌ Your uploaded Excel contains multiple sheets. Please clean or select one sheet first.")
        st.stop()

    # ✅ Choose the most processed version of data
    src = ss.get("df_final") or ss.get("df_clean") or ss.get("df_raw")

    if src is not None:
        if not isinstance(src, pd.DataFrame):
            st.error("❌ Invalid data format (expected DataFrame). Please reload a single-sheet Excel file.")
            st.stop()

        # ✅ Handle column mapping for either cleaned or final dataset
        if {"y", "ds"}.issubset(src.columns):
            plot_df = src.rename(columns={"ds": "Date", "y": "Price"})
        elif ss.date_col and ss.target_col and {ss.date_col, ss.target_col}.issubset(src.columns):
            plot_df = src.rename(columns={ss.date_col: "Date", ss.target_col: "Price"})
        else:
            plot_df = None

        # ✅ Plot only if columns are valid
        if plot_df is not None:
            fig = px.line(plot_df, x="Date", y="Price", title="Historical Stock Price")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("### Basic Stats")
            st.dataframe(plot_df["Price"].describe().to_frame(), use_container_width=True)
        else:
            st.warning("⚠️ Could not find matching Date and Price columns to plot.")
    else:
        st.info("ℹ️ Please upload and prepare your data first in previous tabs.")


# ============================================================
# TAB 6 — Model & Metrics
# ============================================================
with tab6:
    st.subheader("🤖 Prophet Model & Metrics")

    # ✅ Guard: prevent multi-sheet dict issue
    if isinstance(ss.df_raw, dict):
        st.error("❌ Your uploaded Excel contains multiple sheets. Please clean or select one sheet first.")
        st.stop()

    # ---------------- Validate data ----------------
    data = ss.df_final
    if data is None and ss.df_clean is not None:
        try:
            tmp = ss.df_clean[[ss.date_col, ss.target_col]].dropna()
            data = tmp.rename(columns={ss.date_col: "ds", ss.target_col: "y"})
            data["ds"] = pd.to_datetime(data["ds"], errors="coerce")
            data = data.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
        except Exception as e:
            st.error(f"⚠️ Could not prepare data: {e}")
            st.stop()

    # ---------------- Check data validity ----------------
    if data is not None and isinstance(data, pd.DataFrame) and {"ds", "y"}.issubset(data.columns):
        data["ds"] = pd.to_datetime(data["ds"], errors="coerce")
        data = data.dropna(subset=["ds"]).reset_index(drop=True)

        if len(data) < 20:
            st.error("❌ Not enough rows for training. Need ≥ 20 valid records.")
        else:
            try:
                # ---------------- Train/Test Split ----------------
                split = int(len(data) * (1 - test_pct / 100))
                train, test = data.iloc[:split], data.iloc[split:]

                # ---------------- Train Prophet ----------------
                with st.spinner("Training Prophet model..."):
                    model = Prophet(
                        yearly_seasonality=yearly,
                        weekly_seasonality=weekly,
                        daily_seasonality=daily
                    )
                    model.fit(train)

                # ---------------- Forecast ----------------
                freq_use = freq.strip().upper() if freq.strip() else "D"
                future = model.make_future_dataframe(periods=len(test), freq=freq_use)
                fcst = model.predict(future)

                # ---------------- Metrics ----------------
                y_true = test["y"].values
                y_pred = fcst["yhat"].iloc[-len(test):].values
                mae, rmse, mape = metrics(y_true, y_pred)
                st.success(f"✅ MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")

                # ---------------- Plot ----------------
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=test["ds"], y=y_true, name="Actual", mode="lines"))
                fig.add_trace(go.Scatter(x=test["ds"], y=y_pred, name="Predicted", mode="lines"))
                fig.update_layout(
                    title="Actual vs Predicted (Test Period)",
                    xaxis_title="Date",
                    yaxis_title="Price"
                )
                st.plotly_chart(fig, use_container_width=True)

                # ---------------- Save Model State ----------------
                ss._model_trained = model
                ss._last_data = data

            except Exception as e:
                st.error(f"❌ Prophet training failed: {e}")

    else:
        st.info("ℹ️ Please prepare your dataset in Tab 3 (Lag Features) or Tab 1 (Cleaned Data) first.")


# ============================================================
# TAB 7 — 5-Year Forecast
# ============================================================
with tab7:
    st.subheader("📈 5-Year Forecast")
    if "_model_trained" in ss and ss._model_trained:
        model = ss._model_trained
        future = model.make_future_dataframe(periods=1825, freq="D")
        fcst = model.predict(future)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ss._last_data["ds"], y=ss._last_data["y"], name="History"))
        fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat"], name="Forecast"))
        fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat_upper"], fill=None, mode="lines",
                                 line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat_lower"], fill="tonexty", mode="lines",
                                 line=dict(width=0), showlegend=False))
        fig.update_layout(title="5-Year Forecast", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, width='stretch')
        st.download_button("📥 Download 5-Year Forecast",
            data=df_to_excel_bytes(fcst[["ds","yhat","yhat_lower","yhat_upper"]]),
            file_name="forecast_5y.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Train model first in Tab 6.")

# ============================================================
# TAB 8 — Short-Term Forecast
# ============================================================
with tab8:
    st.subheader("📅 Short-Term Forecast")
    if "_model_trained" in ss and ss._model_trained:
        horizon = st.slider("Horizon (days)", 7, 365, 90)
        model = ss._model_trained
        future = model.make_future_dataframe(periods=horizon, freq=freq)
        fcst = model.predict(future)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ss._last_data["ds"], y=ss._last_data["y"], name="History"))
        fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat"], name="Forecast"))
        fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat_upper"], fill=None, mode="lines",
                                 line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat_lower"], fill="tonexty", mode="lines",
                                 line=dict(width=0), showlegend=False))
        fig.update_layout(title=f"Short-Term Forecast ({horizon} days)",
                          xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, width='stretch')
        st.download_button("📥 Download Short-Term Forecast",
            data=df_to_excel_bytes(fcst[["ds","yhat","yhat_lower","yhat_upper"]]),
            file_name=f"forecast_{horizon}d.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Train model first in Tab 6.")


