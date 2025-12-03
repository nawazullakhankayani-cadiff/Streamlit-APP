# Streamlit app: EDA + Train Linear Regression + Predict
# Save and run with: streamlit run app_streamlit_final.py

import os
import time
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- Config - edit if you want ----------------
STUDENT_NAME = "Nawazullakhan kayani"
STUDENT_ID = "st20329043"
MODEL_FOLDER = f"models_{STUDENT_ID}"
# --- FIX APPLIED HERE ---
# Changed the local CSV name to match the file you uploaded: Cleaned_indian_air_data (15).csv
LOCAL_CSV = "Cleaned_indian_air_data (15).csv"

# --- FIX: Ensure model folder exists right at the start ---
os.makedirs(MODEL_FOLDER, exist_ok=True)

# --- CORRECTED: Removed "Final" from page title/config ---
st.set_page_config(page_title="AirLens", layout="wide")
st.title("AirLens — Air Quality Explorer")

# ---------------- Helpers ----------------
@st.cache_data
def load_csv_file(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # coerce numeric columns
    for col in ['PM2.5','PM10','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene','AQI']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # ensure Date is datetime
    if 'Date' in df.columns:
        # --- FIX: Corrected date format to YYYY-MM-DD to match the data ---
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        df['YEAR'] = df['Date'].dt.year
        df['MONTH'] = df['Date'].dt.month
    return df

def smooth_city_ts(df, metric, window_days=7):
    df2 = df[['Date','City', metric]].dropna().sort_values(['City','Date'])
    if df2.empty:
        return pd.DataFrame()
    # rolling using time-based window by city
    out = []
    for city, grp in df2.groupby('City'):
        s = grp.set_index('Date')[metric].rolling(f'{window_days}D').median()
        tmp = s.reset_index()
        tmp['City'] = city
        tmp = tmp.rename(columns={metric: 'value'})
        out.append(tmp)
    if not out:
        return pd.DataFrame()
    result = pd.concat(out, ignore_index=True)
    return result

def prepare_features_and_target(df):
    features = [c for c in ['PM2.5','PM10','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene'] if c in df.columns]
    target = 'AQI'
    return features, target

def train_linear(df, features, target):
    X = df[features]
    y = df[target]
    mask = (~y.isna()) & (X.notna().sum(axis=1) > 0)
    Xf = X.loc[mask]
    yf = y.loc[mask]
    if Xf.shape[0] < 30:
        raise ValueError("Not enough rows (>=30) available for training after filtering.")
    X_train, X_test, y_train, y_test = train_test_split(Xf, yf, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    # --- FIX: Removed 'squared=False' argument and used np.sqrt for RMSE ---
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    
    r2 = r2_score(y_test, preds)
    ts = int(time.time())
    model_path = os.path.join(MODEL_FOLDER, f"linear_aqi_{STUDENT_ID}_{ts}.pkl")
    joblib.dump(pipeline, model_path)
    try:
        coef = pipeline.named_steps['lr'].coef_.tolist()
        intercept = float(pipeline.named_steps['lr'].intercept_)
        coef_map = dict(zip(features, coef))
    except Exception:
        coef_map, intercept = {}, None
    return {
        "pipeline": pipeline,
        "model_path": model_path,
        "rmse": float(rmse),
        "r2": float(r2),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "coefficients": coef_map,
        "intercept": intercept
    }

# ---------------- Sidebar (data + model) ----------------
st.sidebar.header("Project & Files")
st.sidebar.write(f"Student: **{STUDENT_NAME}** — ID: **{STUDENT_ID}**")

uploaded_csv = st.sidebar.file_uploader("Upload cleaned CSV (optional)", type=["csv"])
use_local = st.sidebar.checkbox(f"Use local CSV ({LOCAL_CSV}) if available", value=True)

# optional header image upload & preview
uploaded_image = st.sidebar.file_uploader("Upload a header image (optional)", type=["png","jpg","jpeg"])
if uploaded_image is not None:
    st.image(uploaded_image, use_column_width=True)

# ---------------- Load CSV ----------------
if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
    df.columns = [c.strip() for c in df.columns]
    # convert numeric columns
    for col in ['PM2.5','PM10','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene','AQI']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # ensure Date is datetime
    if 'Date' in df.columns:
        # --- FIX: Corrected date format to YYYY-MM-DD to match the data ---
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        df['YEAR'] = df['Date'].dt.year
        df['MONTH'] = df['Date'].dt.month
else:
    if use_local and os.path.exists(LOCAL_CSV):
        df = load_csv_file(LOCAL_CSV)
    else:
        df = pd.DataFrame()

# ---------------- Load saved models ----------------
# --- NOTE: We read this once outside the page logic to ensure consistency ---
model_files = sorted([f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pkl') or f.endswith('.joblib')], reverse=True)
selected_model_name = st.sidebar.selectbox("Load a saved model (optional)", options=["<none>"] + model_files)
loaded_pipeline = None

# If a model is selected in the sidebar, load it immediately
if selected_model_name != "<none>":
    try:
        loaded_pipeline = joblib.load(os.path.join(MODEL_FOLDER, selected_model_name))
        st.sidebar.success(f"Loaded model: {selected_model_name}")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

# ---------------- Main pages ----------------
page = st.sidebar.radio("Choose page", 
    ["🏠 Home", "📊 Data", "🔍 EDA", "🧠 Train ML Model", "🔮 Predict", "⬇️ Download"]
)

# ----------------- Home -----------------
if page == "🏠 Home":
    st.header("AirLens — Home")
    st.write("Interactive app for exploring India air quality data, training a Linear Regression model to predict AQI, and making manual predictions.")
    if df.empty:
        st.info("No dataset loaded. Upload a cleaned CSV (recommended) or place it as the local CSV and check the 'Use local CSV' checkbox.")
    else:
        st.markdown(f"**Dataset:** {len(df)} rows × {df.shape[1]} columns")
        cols = st.columns(4)
        cols[0].metric("Rows", df.shape[0])
        cols[1].metric("Columns", df.shape[1])
        cols[2].metric("Unique Cities", int(df['City'].nunique()) if 'City' in df.columns else "N/A")
        if 'Date' in df.columns:
            cols[3].metric("Date Range", f"{df['Date'].min().date()} → {df['Date'].max().date()}")

# ----------------- Data -----------------
elif page == "📊 Data":
    st.header("Data Preview & Descriptive Statistics")

    # --- NEW: Dataset Column Overview (User Request) ---
    st.subheader("📘 Dataset Column Overview (India Air Quality Dataset)")
    st.markdown("""
This dataset records air-pollution measurements collected from different cities across India. Each file contains the same set of columns, making it easy to combine them into one dataset. The main columns included are:
* **City** – Identifies the location where the air-quality reading was taken.
* **Date** – The actual calendar date of the observation, from which Year, Month, and Day were later extracted.
* **PM2.5 & PM10** – Two types of particulate matter, representing fine and coarse pollution particles.
* **SO₂, NO₂, CO, O₃** – Four major gaseous pollutants measured in micrograms per cubic meter.
* **YEAR, MONTH, day** – Time-based features derived from the Date column to support trend analysis.
All files follow this same structure, and when they are merged together, they form a complete dataset covering different cities and years.
""")
    
    # --- Existing: Summary of Descriptive Statistics (Based on User Input) ---
    st.subheader("📊 Summary of Descriptive Statistics")
    
    st.markdown("### 1. Count")
    actual_count = len(df)
    st.write(f"The dataset currently has **{actual_count}** entries (rows).")
    st.markdown("*(Based on the full, original data, every pollutant and parameter had about **29,531** measurements, showing the data is substantially complete.)*")

    st.markdown("### 3. Minimum & Maximum Values")
    st.markdown("""
* **PM2.5:** 0.04 → 949.99 µg/m³ (Shows extreme range from very clean to extremely polluted days.)
* **PM10:** 0.01 → 1000 µg/m³ (Large fluctuations, possibly indicating pollution spikes.)
* **CO:** 0 → 175.81 mg/m³ (Mostly low, but some high peaks.)
* **AQI:** 0 → 2049 (A very high maximum likely indicates an outlier or an extreme pollution event.)
""")

    st.markdown("### 4. Quartiles (25%, 50%, 75%)")
    st.markdown("""
These divide the data into four equal parts, showing what is "typical" versus "extreme."

**PM2.5 Example:**
* **25% of readings ≤ 28.99** → The air is cleaner most of the time.
* **Median (50%) = 49.2** → Half of the time, PM2.5 ≤ 49.2 µg/m³.
* **75% = 78.62** → One-quarter of the time, PM2.5 is high.
""")

    st.markdown("### 5. Standard Deviation (Spread of Values)")
    st.markdown("""
* **PM2.5 std = 63.9** → Pollution levels change a lot from day to day.
* **NOx std = 31.6** → Nitrogen oxide also varies a lot.
* **O₃ std = 21.98** → Ozone is somewhat consistent.
""")
    # --- End of Descriptive Stats Summary ---

    if df.empty:
        st.warning("No dataset loaded.")
    else:
        st.subheader("Preview (first 10 rows)")
        st.dataframe(df.head(10))
        
        st.subheader("Missing values (top columns)")
        miss = df.isnull().sum().sort_values(ascending=False)
        miss_pct = 100 * miss / len(df)
        miss_tbl = pd.concat([miss, miss_pct], axis=1)
        miss_tbl.columns = ["Missing", "% Missing"]
        st.dataframe(miss_tbl.head(30))

        # --- NEW: Total missing per City (Based on User Input) ---
        st.subheader("Total Missing Data Points per City")
        
        # User's list of City and Missing Count (parsed from the raw text)
        missing_data = {
            "Ahmedabad": 10175, "Aizawl": 159, "Amaravati": 1135, "Amritsar": 1735,
            "Bengaluru": 3446, "Bhopal": 961, "Brajrajnagar": 4038, "Chandigarh": 43,
            "Chennai": 5265, "Coimbatore": 638, "Delhi": 1085, "Ernakulam": 271,
            "Gurugram": 6705, "Guwahati": 1032, "Hyderabad": 1555, "Jaipur": 1400,
            "Jorapokhar": 8333, "Kochi": 354, "Kolkata": 790, "Lucknow": 6058,
            "Mumbai": 12959, "Patna": 6011, "Shillong": 1014, "Talcher": 3814,
            "Thiruvananthapuram": 3774, "Visakhapatnam": 2156
        }
        
        # Create a DataFrame for better display
        df_missing_city = pd.DataFrame(
            list(missing_data.items()),
            columns=['City', 'Total Missing Data Points']
        ).sort_values(by='Total Missing Data Points', ascending=False).reset_index(drop=True)
        
        st.markdown(
            "*(This table shows the total number of missing data points across **all** pollutant columns for each city, highlighting which areas had the most incomplete records.)*"
        )
        st.dataframe(df_missing_city)

# ----------------- EDA -----------------
elif page == "🔍 EDA":
    st.header("Interactive EDA (smoothed & aggregated)")
    if df.empty:
        st.warning("No data loaded.")
    else:
        cities = sorted(df['City'].dropna().unique()) if 'City' in df.columns else []
        sel_cities = st.multiselect("Select cities (empty = all)", options=cities, default=cities[:6])
        
        if 'YEAR' in df.columns:
            # --- FIX: Added robustness check to handle NaN years (prevents ValueError) ---
            valid_years = df['YEAR'].dropna()
            
            if not valid_years.empty:
                yr_min, yr_max = int(valid_years.min()), int(valid_years.max())
                yr_range = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))
                df = df[(df['YEAR'] >= yr_range[0]) & (df['YEAR'] <= yr_range[1])]
            else:
                st.info("No valid 'Date' entries found to determine a year range.")

        filt = df.copy()
        if sel_cities:
            filt = filt[filt['City'].isin(sel_cities)]

        st.subheader("1. Smoothed time series (7-day rolling median)")
        metric_options = [c for c in ['AQI','PM2.5','PM10','NO2','SO2','CO','O3'] if c in filt.columns]
        if metric_options:
            metric = st.selectbox("Metric", options=metric_options, index=0)
            ts_sm = smooth_city_ts(filt, metric, window_days=7)
            if ts_sm.empty:
                st.info("No time series data for selected metric.")
            else:
                # --- EDA IMPROVEMENT: Apply 'seaborn' template and better line look ---
                fig = px.line(ts_sm, x='Date', y='value', color='City', 
                              title=f"Time Series: {metric} (7-day Median)",
                              template="seaborn")
                fig.update_traces(line=dict(width=3)) 
                fig.update_layout(height=520, margin=dict(t=50, b=50, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("2. Top cities by pollutant (bar)")
        pollutant_list = [c for c in ['PM2.5','PM10','NO2','SO2','CO','O3'] if c in filt.columns]
        if pollutant_list:
            poll = st.selectbox("Pollutant to rank", options=pollutant_list, index=0)
            n = st.slider("Top N cities", 3, min(30, filt['City'].nunique()), 8)
            agg = filt.groupby('City')[poll].mean().reset_index().sort_values(by=poll, ascending=False).head(n)
            # --- EDA IMPROVEMENT: Apply 'seaborn' template, color by pollutant value for visual hierarchy ---
            fig2 = px.bar(agg, x='City', y=poll, text=agg[poll].round(2), 
                          title=f"Top {n} cities by mean {poll}",
                          color=poll,
                          color_continuous_scale=px.colors.sequential.Plotly3,
                          template="seaborn")
            fig2.update_traces(textposition='outside')
            fig2.update_layout(xaxis_tickangle=-45, height=420, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("3. Distribution per city (violin) and boxplots of city averages")
        numeric_cols = filt.select_dtypes(include=[np.number]).columns.tolist()
        default_cols = [c for c in ['PM2.5','AQI'] if c in numeric_cols]
        chosen = st.multiselect("Choose numeric columns (for distr.)", options=numeric_cols, default=default_cols)
        if chosen:
            col0 = chosen[0]
            small = filt[['City', col0]].dropna()
            if small.empty:
                st.info("Not enough data.")
            else:
                # --- EDA IMPROVEMENT: Apply 'seaborn' template, color by city for distinction ---
                fig3 = px.violin(small, x='City', y=col0, box=True, points='outliers', 
                                 title=f"Distribution of {col0} by City",
                                 color='City',
                                 template="seaborn")
                fig3.update_layout(xaxis_tickangle=-45, height=480)
                st.plotly_chart(fig3, use_container_width=True)

            avg_city = filt.groupby('City')[chosen].mean().reset_index()
            if not avg_city.empty:
                avg_melt = avg_city.melt(id_vars='City', var_name='Pollutant', value_name='Avg')
                # --- EDA IMPROVEMENT: Apply 'seaborn' template ---
                fig4 = px.box(avg_melt, x='Pollutant', y='Avg', color='Pollutant', 
                              title="Boxplots of City-wise Averages",
                              template="seaborn")
                fig4.update_layout(height=420)
                st.plotly_chart(fig4, use_container_width=True)

        st.subheader("4. Correlation heatmap")
        corr_select = st.multiselect("Pick numeric columns", options=numeric_cols, default=[c for c in numeric_cols if c in ['PM2.5','PM10','AQI']][:3])
        if len(corr_select) >= 2:
            corr = filt[corr_select].corr().round(2)
            # --- EDA IMPROVEMENT: Use a diverging color scale for better correlation visualization ---
            fig5 = px.imshow(corr, text_auto=True, title="Correlation matrix", 
                             aspect="auto",
                             color_continuous_scale=px.colors.diverging.RdBu,
                             template="seaborn")
            fig5.update_layout(height=420)
            st.plotly_chart(fig5, use_container_width=True)

# ----------------- Train Model -----------------
elif page == "🧠 Train ML Model":
    st.header("Train Linear Regression model to predict AQI")
    if df.empty:
        st.warning("Upload or load dataset first.")
    else:
        features, target = prepare_features_and_target(df)
        st.write("Detected features:", features)
        st.write("Target:", target)
        st.write("Note: training requires at least 30 rows with target+feature data.")
        if st.button("Train Linear Regression model now"):
            try:
                res = train_linear(df, features, target)
                st.success("Model trained and saved.")
                st.write(f"Model path: `{res['model_path']}`")
                st.write(f"RMSE: {res['rmse']:.3f}    R²: {res['r2']:.3f}")
                st.write("Training rows:", res['n_train'], "Test rows:", res['n_test'])
                if res['coefficients']:
                    st.subheader("Model coefficients (LinearRegression pipeline)")
                    st.table(pd.DataFrame.from_dict(res['coefficients'], orient='index', columns=['coef']).sort_values('coef', ascending=False))
                
                # --- NEW FIX: Set the trained model as the currently loaded one immediately ---
                st.session_state['loaded_pipeline_data'] = res['pipeline']
                st.session_state['just_trained_model_name'] = res['model_path'].split(os.path.sep)[-1]
                
            except Exception as e:
                st.error(f"Training failed: {e}")

# ----------------- Predict -----------------
elif page == "🔮 Predict":
    st.header("Manual prediction using a loaded model")
    
    # --- FIX: Check for the immediately trained model first ---
    if 'loaded_pipeline_data' in st.session_state:
        loaded_pipeline = st.session_state['loaded_pipeline_data']
        st.info(f"Using model trained in the current session: **{st.session_state['just_trained_model_name']}**")
    elif loaded_pipeline is None:
        # If no model is selected in the sidebar, check if any saved models exist
        if model_files:
            st.info("Please select one of the saved models from the sidebar (under 'Load a saved model (optional)') to use the prediction feature.")
        else:
             st.info("No model loaded in sidebar. Please go to the '🧠 Train ML Model' page and train a model first.")


    if df.empty:
        st.warning("Load dataset first (so we can provide sensible defaults).")
    elif loaded_pipeline is not None:
        
        features, target = prepare_features_and_target(df)
        if target not in df.columns:
            st.error("AQI column is missing from dataset; you can still predict if you have a trained model loaded.")
        st.write("Features used for prediction:", features)
        defaults = {f: float(df[f].median(skipna=True)) if f in df.columns else 0.0 for f in features}

        cols = st.columns(min(4, max(1, len(features))))
        inputs = {}
        for i, feat in enumerate(features):
            with cols[i % len(cols)]:
                inputs[feat] = st.number_input(feat, min_value=0.0, value=float(defaults[feat]), format="%.2f")

        if st.button("Predict AQI with loaded model"):
            Xnew = pd.DataFrame([inputs])
            try:
                pred = loaded_pipeline.predict(Xnew)[0]
                st.metric("Predicted AQI", f"{pred:.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ----------------- Download -----------------
elif page == "⬇️ Download":
    st.header("Download cleaned preview & saved models")
    if df.empty:
        st.info("No data loaded.")
    else:
        df_clean = df.copy()
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
        df_clean = df_clean.fillna("Unknown")
        csv_bytes = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button("Download cleaned preview CSV", data=csv_bytes, file_name=f"cleaned_preview_{STUDENT_ID}.csv")

    st.subheader("Saved models in models folder")
    if model_files:
        for m in model_files:
            path = os.path.join(MODEL_FOLDER, m)
            with open(path, "rb") as fh:
                st.download_button(m, data=fh, file_name=m)
    else:
        st.write("No saved models found in folder.")
        
