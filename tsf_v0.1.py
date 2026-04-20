#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
#----------------------------------------
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#----------------------------------------
from datetime import datetime, timedelta
#----------------------------------------
from sklearn.model_selection import train_test_split
#----------------------------------------
# Forecast
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
#----------------------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
#----------------------------------------
from prophet import Prophet
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM as KerasLSTM, Dense
#from tensorflow.keras.optimizers import Adam
#----------------------------------------
import warnings
warnings.filterwarnings("ignore")
#----------------------------------------
from tsf_func import load_file
from tsf_func import check_missing_values, handle_numerical_missing_values, handle_categorical_missing_values
from tsf_func import evaluate, color_objective

from ts2_func import (load_file, check_missing_values,run_stationarity_tests, get_decomposition_insights, evaluate, color_objective)
#---------------------------------------------------------------------------------------------------------------------------------
### Title for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Forecasting Studio | v0.2",
                    layout="wide",
                    page_icon="📈",            
                    initial_sidebar_state="auto")
#---------------------------------------------------------------------------------------------------------------------------------
### CSS
#---------------------------------------------------------------------------------------------------------------------------------

# Professional Styling
st.markdown("""
<style>
    /* Global Font */
    .stApp { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #e9ecef; }
    [data-testid="stSidebar"] .stMarkdown h2 { font-size: 1.2rem; color: #495057; }
    
    /* Headers */
    h1, h2, h3 { color: #212529; font-weight: 700; letter-spacing: -0.5px; }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: 700; color: #0d6efd; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #6c757d; font-weight: 600; text-transform: uppercase; }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background-color: #0d6efd;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        padding: 10px 20px;
        transition: all 0.2s;
    }
    .stButton > button:hover { background-color: #0b5ed7; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    
    /* Containers */
    .stContainer { border: 1px solid #e9ecef; border-radius: 8px; padding: 15px; background-color: white; }
    
    /* Hide Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; color: #495057; }
    .stTabs [aria-selected="true"] { color: #0d6efd !important; border-bottom: 2px solid #0d6efd; }
</style>
""", unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------------------------------------
### Description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .title-large {
        text-align: center;font-size: 35px;font-weight: bold;background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;-webkit-text-fill-color: transparent;
    }
    .title-small {
        text-align: center;font-size: 20px;background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;-webkit-text-fill-color: transparent;
    }
    .version-badge {
        text-align: center;display: inline-block;background: linear-gradient(120deg, #0056b3, #0d4a96);
        color: white;padding: 2px 12px;border-radius: 20px;font-size: 1.15rem;margin-top: 8px;font-weight: 600;
        letter-spacing: 0.5px;box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    </style>
    <div style="text-align: center;">
        <div class="title-large">Forecasting Studio</div>
        <div class="version-badge"> Play with Future | v0.2</div>
    </div>
    """,
    unsafe_allow_html=True)

#----------------------------------------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;left: 0;bottom: 0;width: 100%;background-color: #F0F2F6;text-align: center;
        padding: 10px;font-size: 14px;color: #333;z-index: 100;
    }
    .footer p {
        margin: 0;
    }
    .footer .highlight {
        font-weight: bold;color: blue;
    }
    </style>
    <div class="footer">
        <p>© 2026 | Created by : <span class="highlight">Avijit Chakraborty</span> <a href="mailto:avijit.mba18@gmail.com"> 📩 </a> | <span class="highlight">Thank you for visiting the app | Unauthorized uses or copying is strictly prohibited | For best view of the app, please zoom out the browser to 75%.</span> </p>
    </div>
    """,unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

with st.popover("**:blue[:hammer_and_wrench: Hyperparameters]**",disabled=False, use_container_width=True):  

    subcol1, subcol2, subcol3, subcol4, subcol5, subcol6 = st.columns(6)
    with subcol1: 
                
        with st.container(border=True): 
            decom_model_type = st.selectbox("**Model type for decomposition**", ["additive", "multiplicative", ])
                    
        with st.container(border=True):                         
            numerical_strategies = ['interpolate','mean', 'median', 'most_frequent', 'forward fill (ffill)', 'backward fill (bfill)']
            categorical_strategies = ['constant','most_frequent']
            selected_numerical_strategy = st.selectbox("**Missing value treatment : Numerical**", numerical_strategies)
            selected_categorical_strategy = st.selectbox("**Missing value treatment : Categorical**", categorical_strategies)  
                    
    with subcol2:
                
        with st.container(border=True):                    
            apply_resample = st.checkbox("🔁 **Apply Resampling**")
            
            if apply_resample:
                with st.expander("**Resampling Options**"):
                    resample_type = st.radio("**Resampling Type**",options=["Downsampling", "Upsampling"],
                                                    help="Downsampling: Aggregate to lower frequency (e.g., daily → monthly). Upsampling: Increase frequency (e.g., monthly → daily), requires filling method.")
                    st.divider()
                    freq_map = {'D': 'Daily','W': 'Weekly','M': 'Monthly','Q': 'Quarterly','Y': 'Yearly','H': 'Hourly','T': 'Minutely','S': 'Secondly'}
                    if resample_type == "Downsampling":
                        st.write("📅 **Downsampling** – Aggregating to a lower frequency")
                        selected_freq = st.selectbox("**Target Frequency**",options=['W', 'M', 'Q', 'Y'],format_func=lambda x: freq_map[x])
                        method = st.selectbox("**Aggregation Method**", ["mean", "sum", "first", "last"])
                    else:  # Upsampling
                        st.write("📈 **Upsampling** – Increasing to a higher frequency")
                        selected_freq = st.selectbox("**Target Frequency**",options=['D', 'H', 'T', 'S'],format_func=lambda x: freq_map[x])
                        method = st.selectbox("**Fill Method for Upsampling**",["forward fill", "backward fill", "interpolate", "zero"])
           
            else:
                selected_freq = None
                resample_type = None
                method = None                           
                            
        with st.container(border=True):
            apply_log = st.checkbox("♾️ **Apply Log Transform**")
            log_col_name = None
            if apply_log:
                with st.expander("**Log Transform**"):
                    # ✅ Safety check: Only validate if df exists
                    if 'df' in locals() and target_col in df.columns:
                        if (df[target_col] <= 0).any():
                            st.warning("Log transform not possible: non-positive values.")
                        else:
                            log_col_name = f"{target_col}_log"
                            st.success("✅ Log transform will be applied.")
                    else:
                        st.info("ℹ️ Upload a file first to validate log transform.")

        with st.container(border=True):
            apply_fourier = st.checkbox("🔢 **Fourier Terms (Multiple Seasonalities)**")
            if apply_fourier:
                with st.expander("**Fourier Terms**"):
                    period1 = st.number_input("**Seasonal Period 1**", min_value=2, value=7)
                    period2 = st.number_input("**Seasonal Period 2**", min_value=2, value=365)
                    num_terms = st.slider("**Fourier Pairs**", 1, 10, 3)
            else:
                period1 = period2 = num_terms = None                           
                                    
    with subcol3: 
                
        with st.container(border=True): 
            freq_guess = st.number_input("**Seasonal Frequency (e.g. 12 for monthly, 7 for weekly)**", min_value=2, max_value=365, value=12)
                        
        with st.container(border=True):   
            max_diff = st.selectbox("**Maximum Differencing Steps Allowed**", options=[1,2,3,4,5,6], index=2)

        with st.container(border=True):
            lags_val = st.slider("**Select number of lags**", min_value=10, max_value=100, value=40, step=5)
            pacf_method = st.selectbox("**PACF Method**", options=["ywm", "ols", "ldb", "ld", "ywunbiased", "ywadjusted"], index=0) 
                                                                                            
    with subcol4: 
                
        with st.container(border=True):                                    
            train = st.slider("**Train Size (as %)**", 10, 90, 70, 5)
            test = st.slider("**Test Size (as %)**", 10, 50, 30, 5)    
            random_state = st.number_input("**Random State**", 0, 100, 42)
            n_jobs = st.number_input("**Parallel Processing (n_jobs)**", -10, 10, 1) 

    with subcol5: 
                                    
                with st.expander("**📌 Tune | Smoothning**", expanded=False):
                    alpha = st.slider('**Alpha (Smoothing Parameter)**', min_value=0.01, max_value=1.0, value=0.2,key = 'ses_1')
                    beta = st.slider('**Beta (Trend Smoothing Parameter)**', min_value=0.01, max_value=1.0, value=0.2,key = 'ses_2')
                    gamma = st.slider('**Gamma (Seasonality Smoothing Parameter)**', min_value=0.01, max_value=1.0, value=0.2,key = 'ses_3') 
                                    
                                    #order_arima = st.text_input('**ARIMA Order (p,d,q)**:', '1,1,1')
                                    #order_arima = tuple(map(int, order_arima.split(',')))     
                                    #order_sarima = st.text_input('**SARIMA Order (p,d,q,m)**:', '1,1,1,12')
                                    #order_sarima = tuple(map(int, order_sarima.split(',')))    
                                    
                with st.expander("**📌 Tune | ARIMA**"):
                    arima_p = st.slider("**ARIMA p (max)**", 0, 5, 2)
                    arima_d = st.slider("**ARIMA d (max)**", 0, 3, 2)
                    arima_q = st.slider("**ARIMA q (max)**", 0, 5, 2)

                with st.expander("**📌 Tune | Prophet**"):
                    changepoint_prior = st.slider("**Prophet: Changepoint Prior**", 0.001, 1.0, 0.05, 0.01)
                    seasonality_prior = st.slider("**Seasonality Prior**", 0.01, 10.0, 1.0, 0.1)
                    seasonality_mode = st.selectbox("**Seasonality Mode**", ["additive", "multiplicative"])

                with st.expander("**📌 Tune | Random Forest & XGBoost**"):
                    n_estimators = st.slider("**n_estimators**", 50, 200, 100)
                    max_depth = st.slider("**max_depth**", 3, 10, 5)

    with subcol6:

        with st.container(border=True): 
                    model = st.selectbox("**Select Model**", ["l2", "l1", "rbf"], index=0)
                    algo_name = st.selectbox("**Select Algorithm**", ["Binary Segmentation", "Pelt", "Window", "Bottom-Up"], index=0)
                    num_change_points = st.slider("**Number of Change Points**", min_value=1, max_value=20, value=3)     
           

#---------------------------------------------------------------------------------------------------------------------------------
### Main app
#---------------------------------------------------------------------------------------------------------------------------------

#st.divider()

col_inf, col_space, col_det = st.columns((0.15,0.01,0.85))

with col_inf: 
    
    with st.container(border=True):

        uploaded_file = st.file_uploader("**:blue[Choose a file]**",type=["csv", "xls", "xlsx"], accept_multiple_files=False,key="file_upload")
        if not uploaded_file:
            st.info("Please upload a CSV file to begin.")
            st.stop()
        
        if uploaded_file is not None:
            st.success("Data loaded successfully!")
            df = load_file(uploaded_file)
            if df.empty:
                st.error("Failed to load data.")
                st.stop()
    
    with st.container(border=True):
        
        ts_type = st.radio("**Series Type**", ["Univariate", "Multivariate"], horizontal=True)
    
    with st.container(border=True):
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            time_col = st.selectbox("**Time Column**", df.columns)
            try:
                df[time_col] = pd.to_datetime(df[time_col],errors='coerce')
            except Exception as e:
                st.error("Time column parsing failed.")
                st.stop()
        with col_s2:
            target_col = st.selectbox("**Target Variable**", [c for c in df.columns if c != time_col])
        
        feature_cols = [target_col]
        if ts_type == "Multivariate":
            feature_cols = st.multiselect("**Features**", [c for c in df.columns if c != time_col], default=[target_col])

    #st.markdown("---")
    #st.subheader("🔧 Hyperparameters")
    
    with st.expander('**⚙️ Lags**'):
    
        add_lags = st.checkbox("**Add Lagged Features**", value=False)
        if add_lags:
            num_lags = st.number_input("**Number of Lags**", min_value=1, max_value=30, value=3)
    
    with st.container(border=True):
        forecast_steps = st.number_input("**Forecast Horizon (Days)**", 1, 365, 30)
    
    run_btn = st.button("▶️ Run Forecast", type="primary", use_container_width=True)

#----------------------------------------

with col_det:
    
    if run_btn:
        
        with st.spinner("🔄 Processing data, running diagonisting tests, training models and forecasting..."):
        
            st.session_state['target_col'] = target_col
            st.session_state['feature_cols'] = feature_cols
            
            target_col = st.session_state['target_col']
            feature_cols = st.session_state['feature_cols']
            
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df = df.dropna(subset=[time_col]).sort_values(time_col)
            
            # ---------------------------------- Data Prep ---------------------------
            #df = df_raw.copy()           
            
            # ----------------------------------
            df_temp = df.copy()
            transformations = []

            if apply_resample and selected_freq:
                try:
                    df_temp = df_temp.set_index(time_col).resample(selected_freq).mean(numeric_only=True).reset_index()
                    transformations.append("Resampled")
                except:
                    st.error("Resampling failed.")

            # Log transform
            if apply_log and log_col_name:
                df_temp[log_col_name] = np.log(df_temp[target_col])
                target_col = log_col_name
                transformations.append("Log Transform")

            # Fourier terms
            if apply_fourier:
                for i in range(1, num_terms + 1):
                    for period in [period1, period2]:
                        omega = 2 * np.pi * i / period
                        df_temp[f'sin_{period}_{i}'] = np.sin(omega * np.arange(len(df_temp)))
                        df_temp[f'cos_{period}_{i}'] = np.cos(omega * np.arange(len(df_temp)))
                        feature_cols.extend([f'sin_{period}_{i}', f'cos_{period}_{i}'])
                transformations.append("Fourier Terms")

            st.session_state['df_processed'] = df_temp
            st.session_state['transformations'] = transformations
            df = st.session_state['df_processed']
            
            # ----------------------------------        
            # Handle Missing Values
            #df = handle_numerical_missing_values(df, 'interpolate')
            missing = check_missing_values(df)
            df = handle_numerical_missing_values(df, selected_numerical_strategy)

            # ----------------------------------   
            # Stationarity Tests
            
            series = df[target_col].dropna()

            # Run ADF test
            try:
                adf_result = adfuller(series)
                adf_stat = adf_result[0]
                adf_p = adf_result[1]
                adf_lags = adf_result[2]
                adf_nobs = adf_result[3]
                adf_crit = adf_result[4]
            except Exception as e:
                st.error(f"ADF test failed: {e}")
                adf_stat = adf_p = np.nan

            # Run KPSS test
            try:
                kpss_result = kpss(series, regression='c', nlags="auto")
                kpss_stat = kpss_result[0]
                kpss_p = kpss_result[1]
                kpss_lags = kpss_result[2]
                kpss_crit = kpss_result[3]
            except Exception as e:
                st.error(f"KPSS test failed: {e}")
                kpss_stat = kpss_p = np.nan

            # Create results DataFrame
            test_results = pd.DataFrame({
                        "Test": ["ADF (Augmented Dickey-Fuller)", "KPSS (Kwiatkowski-Phillips-Schmidt-Shin)"],
                        "Statistic": [f"{adf_stat:.6f}", f"{kpss_stat:.6f}"],
                        "p-value": [f"{adf_p:.6f}", f"{kpss_p:.6f}"],
                        "Lags Used": [adf_lags, kpss_lags],
                        "Number of Obs": [adf_nobs, len(series)],
                        "Conclusion": [
                            "✅ Series is stationary — no differencing required." if adf_p < 0.05 else "❗Series is **non-stationary** — differencing required.",
                            "✅ Series is stationary — no differencing required." if kpss_p > 0.05 else "❗Series is **non-stationary** — differencing required."
                        ]})
                        
            # ----------------------------------   
            # Differentiation
            d = 0
            diff_series = series.copy()
            max_d = max_diff  # Using the slider value from sidebar
            
            initial_adf_p = adf_p
            initial_kpss_p = kpss_p
            
            # Only run loop if initially non-stationary
            is_non_stationary = (not np.isnan(initial_adf_p) and initial_adf_p >= 0.05) or \
                                (not np.isnan(initial_kpss_p) and initial_kpss_p <= 0.05)
            
            if is_non_stationary:
                while (adf_p >= 0.05 or (kpss_p <= 0.05 and not np.isnan(kpss_p))) and d < max_d:
                    diff_series = diff_series.diff().dropna()
                    try:
                        adf_p = adfuller(diff_series)[1]
                    except:
                        adf_p = 1.0
                    try:
                        kpss_p = kpss(diff_series)[1]
                    except:
                        kpss_p = 0.0
                    d += 1
                #st.info(f"💡 Applied differencing order: **d = {d}**")
            else:
                #st.success("✅ Series is already stationary. No differencing required (`d=0`).")
                diff_series = series
            
            # ----------------------------------
            df_modeling = df.copy()
                                
            if d > 0:
                aligned_diff = pd.Series(diff_series.values, index=diff_series.index)
                aligned_diff = aligned_diff.reindex(df_modeling.index)
                df_modeling[target_col] = aligned_diff
                
                initial_rows = len(df_modeling)
                df_modeling = df_modeling.dropna(subset=[target_col])
                dropped_rows = initial_rows - len(df_modeling)
                
                #if dropped_rows > 0:
                    #st.info(f"⚠️ Dropped **{dropped_rows}** rows with NaN values from differencing.")
                    #st.info(f"✅ Modeling dataset: **{len(df_modeling)}** rows (from original {initial_rows})")
            else:
                st.info("ℹ️ Modeling will use the **original target variable**.")
            
            # ----------------------------------
            st.session_state['d'] = d
            st.session_state['diff_series'] = diff_series
            st.session_state['adf_p'] = adf_p
            st.session_state['kpss_p'] = kpss_p
            st.session_state['series'] = series
            st.session_state['test_results'] = test_results
            # ----------------------------------                      
            # Split Data (NOW SAFE - No NaNs)
            
            split_idx = int(len(df_modeling) * (train/100))
            train, test = df_modeling.iloc[:split_idx], df_modeling.iloc[split_idx:]
            
            y_train = train[target_col].values
            y_test = test[target_col].values
            
            # Final safety check
            if np.isnan(y_train).any():
                st.error("❌ Critical Error: y_train still contains NaNs after cleaning!")
                st.write("Rows with NaN:")
                st.write(train[train[target_col].isnull()])
                st.stop()
                
            if np.isnan(y_test).any():
                st.error("❌ Critical Error: y_test still contains NaNs!")
                st.stop()
            
            st.success(f"✅ Data ready! Train: **{len(y_train)}** rows, Test: **{len(y_test)}** rows")
                
            dates_test = test[time_col]
            dates_full = df[time_col]
            y_full = df[target_col].values
            
            # Stationarity Check (Cached)
            #stat_results = run_stationarity_tests(df[target_col])
            #d_order = 0
            #if stat_results['ADF'] and not stat_results['ADF']['is_stationary']:
                #d_order = 1 # Simplified logic for demo; real app might loop diff
            
            # ----------------------------------      
            # --- Model Training ---

            models_pred = {}
            metrics_list = []

            # 1. Baseline: Moving Average
            ma_window = 5
            ma_pred = np.convolve(y_train, np.ones(ma_window)/ma_window, mode='valid')
            ma_pred = np.concatenate([ma_pred, [ma_pred[-1]] * len(y_test)])[-len(y_test):]
            models_pred['Moving Avg'] = ma_pred
            metrics_list.append(['Moving Avg', *evaluate(ma_pred, y_test)])

            # 2. Naive Forecast (Last value)
            naive_pred = np.full(len(y_test), y_train[-1])
            models_pred['Naive'] = naive_pred
            metrics_list.append(['Naive', *evaluate(naive_pred, y_test)])

            # 3. Seasonal Naive (Last seasonal value)
            seasonal_period = freq_guess
            seasonal_naive_pred = []
            for i in range(len(y_test)):
                idx = len(y_train) - seasonal_period + (i % seasonal_period)
                if idx >= 0:
                    seasonal_naive_pred.append(y_train[idx])
                else:
                    seasonal_naive_pred.append(y_train[-1])
            seasonal_naive_pred = np.array(seasonal_naive_pred)
            models_pred['Seasonal Naive'] = seasonal_naive_pred
            metrics_list.append(['Seasonal Naive', *evaluate(seasonal_naive_pred, y_test)])

            # 4. Exponential Smoothing (Simple)
            try:
                ses = SimpleExpSmoothing(y_train).fit()
                ses_pred = ses.forecast(len(y_test))
                models_pred['Exp Smooth'] = ses_pred
                metrics_list.append(['Exp Smooth', *evaluate(ses_pred, y_test)])
            except: pass

            # 5. ETS (Exponential Smoothing State Space Model)
            try:
                ets = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=seasonal_period).fit()
                ets_pred = ets.forecast(len(y_test))
                models_pred['ETS'] = ets_pred
                metrics_list.append(['ETS', *evaluate(ets_pred, y_test)])
            except Exception as e:
                st.warning(f"ETS failed: {e}")

            # 6. Holt's Linear Trend
            try:
                holt = ExponentialSmoothing(y_train, trend='add', seasonal=None).fit()
                holt_pred = holt.forecast(len(y_test))
                models_pred["Holt's Linear"] = holt_pred
                metrics_list.append(["Holt's Linear", *evaluate(holt_pred, y_test)])
            except: pass

            # 7. Holt-Winters (Triple Exponential Smoothing)
            try:
                hw = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=seasonal_period).fit()
                hw_pred = hw.forecast(len(y_test))
                models_pred['Holt-Winters'] = hw_pred
                metrics_list.append(['Holt-Winters', *evaluate(hw_pred, y_test)])
            except: pass

            # 8. Auto ARIMA
            try:
                auto_arima_model = auto_arima(y_train, seasonal=True, m=seasonal_period, 
                                            trace=False, error_action='ignore', suppress_warnings=True)
                auto_arima_pred = auto_arima_model.predict(n_periods=len(y_test))
                models_pred['Auto ARIMA'] = auto_arima_pred
                metrics_list.append(['Auto ARIMA', *evaluate(auto_arima_pred, y_test)])
            except Exception as e:
                st.warning(f"Auto ARIMA failed: {e}")

            # 9. ARIMA (Manual)
            try:
                arima = ARIMA(y_train, order=(arima_p, d, arima_q)).fit()
                arima_pred = arima.forecast(steps=len(y_test))
                models_pred['ARIMA'] = arima_pred
                metrics_list.append(['ARIMA', *evaluate(arima_pred, y_test)])
            except Exception as e:
                st.warning(f"ARIMA failed: {e}")

            # 10. Linear Regression
            try:
                X_train_lr = np.arange(len(y_train)).reshape(-1, 1)
                X_test_lr = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)
                lr = LinearRegression()
                lr.fit(X_train_lr, y_train)
                lr_pred = lr.predict(X_test_lr)
                models_pred['Linear Regression'] = lr_pred
                metrics_list.append(['Linear Regression', *evaluate(lr_pred, y_test)])
            except: pass

            # 11. Ridge Regression
            try:
                ridge = Ridge(alpha=1.0, random_state=42)
                ridge.fit(X_train_lr, y_train)
                ridge_pred = ridge.predict(X_test_lr)
                models_pred['Ridge'] = ridge_pred
                metrics_list.append(['Ridge', *evaluate(ridge_pred, y_test)])
            except: pass

            # 12. Lasso Regression
            try:
                lasso = Lasso(alpha=0.1, random_state=42)
                lasso.fit(X_train_lr, y_train)
                lasso_pred = lasso.predict(X_test_lr)
                models_pred['Lasso'] = lasso_pred
                metrics_list.append(['Lasso', *evaluate(lasso_pred, y_test)])
            except: pass

            # 13. Elastic Net
            try:
                elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
                elastic.fit(X_train_lr, y_train)
                elastic_pred = elastic.predict(X_test_lr)
                models_pred['Elastic Net'] = elastic_pred
                metrics_list.append(['Elastic Net', *evaluate(elastic_pred, y_test)])
            except: pass

            # 14. Huber Regressor (Robust to outliers)
            try:
                huber = HuberRegressor(epsilon=1.35, max_iter=1000)
                huber.fit(X_train_lr, y_train)
                huber_pred = huber.predict(X_test_lr)
                models_pred['Huber'] = huber_pred
                metrics_list.append(['Huber', *evaluate(huber_pred, y_test)])
            except: pass

            # 15. Random Forest (Using index as feature)
            X_train_idx = np.arange(len(y_train)).reshape(-1, 1)
            X_test_idx = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)

            rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            rf.fit(X_train_idx, y_train)
            rf_pred = rf.predict(X_test_idx)
            models_pred['Random Forest'] = rf_pred
            metrics_list.append(['Random Forest', *evaluate(rf_pred, y_test)])

            # 16. XGBoost
            xgb = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            xgb.fit(X_train_idx, y_train)
            xgb_pred = xgb.predict(X_test_idx)
            models_pred['XGBoost'] = xgb_pred
            metrics_list.append(['XGBoost', *evaluate(xgb_pred, y_test)])

            # 17. Prophet
            try:
                prophet_df = df[[time_col, target_col]].rename(columns={time_col: 'ds', target_col: 'y'})
                m_prophet = Prophet(changepoint_prior_scale=changepoint_prior, 
                                    seasonality_prior_scale=seasonality_prior,
                                    seasonality_mode=seasonality_mode)
                m_prophet.fit(prophet_df.iloc[:split_idx])
                future = m_prophet.make_future_dataframe(periods=len(y_test), freq='D')
                forecast_prophet = m_prophet.predict(future)
                prophet_pred = forecast_prophet.iloc[-len(y_test):]['yhat'].values
                models_pred['Prophet'] = prophet_pred
                metrics_list.append(['Prophet', *evaluate(prophet_pred, y_test)])
            except Exception as e:
                st.warning(f"Prophet failed: {e}")

            # ----------------------------------   
            # Compile Metrics DF
            
            metrics_df = pd.DataFrame(metrics_list, columns=['Model', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R2'])
            best_model_row = metrics_df.loc[metrics_df['RMSE'].idxmin()]
            best_model_name = best_model_row['Model']

            # ---------------------------------- 
            # Generate Future Forecast (Full Data Retrain)

            future_dates = pd.date_range(start=dates_full.iloc[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
            future_preds_dict = {}

            # 1. Moving Average Future
            try:
                last_ma = np.mean(y_full[-ma_window:])
                future_preds_dict['Moving Avg'] = np.full(forecast_steps, last_ma)
            except: pass

            # 2. Naive Forecast Future (Last observed value)
            try:
                naive_value = y_full[-1]
                future_preds_dict['Naive'] = np.full(forecast_steps, naive_value)
            except: pass

            # 3. Seasonal Naive Future (Last seasonal value)
            try:
                seasonal_period = freq_guess
                seasonal_naive_future = []
                for i in range(forecast_steps):
                    idx = len(y_full) - seasonal_period + (i % seasonal_period)
                    if idx >= 0:
                        seasonal_naive_future.append(y_full[idx])
                    else:
                        seasonal_naive_future.append(y_full[-1])
                future_preds_dict['Seasonal Naive'] = np.array(seasonal_naive_future)
            except: pass

            # 4. Exponential Smoothing Future
            try:
                ses_full = SimpleExpSmoothing(y_full).fit()
                future_preds_dict['Exp Smooth'] = ses_full.forecast(forecast_steps)
            except: pass

            # 5. Holt's Linear Trend Future
            try:
                holt_full = ExponentialSmoothing(y_full, trend='add', seasonal=None).fit()
                future_preds_dict["Holt's Linear"] = holt_full.forecast(forecast_steps)
            except: pass

            # 6. Holt-Winters Future (Triple Exponential Smoothing)
            try:
                hw_full = ExponentialSmoothing(y_full, trend='add', seasonal='add', seasonal_periods=seasonal_period).fit()
                future_preds_dict['Holt-Winters'] = hw_full.forecast(forecast_steps)
            except: pass

            # 7. ARIMA Future
            try:
                arima_full = ARIMA(y_full, order=(arima_p, d, arima_q)).fit()
                future_preds_dict['ARIMA'] = arima_full.forecast(steps=forecast_steps)
            except: pass

            # 8. Auto ARIMA Future
            try:
                auto_arima_full = auto_arima(y_full, seasonal=True, m=seasonal_period, 
                                            trace=False, error_action='ignore', suppress_warnings=True)
                future_preds_dict['Auto ARIMA'] = auto_arima_full.predict(n_periods=forecast_steps)
            except: pass

            # 9. Random Forest Future
            try:
                X_future_idx = np.arange(len(y_full), len(y_full) + forecast_steps).reshape(-1, 1)
                rf_full = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                rf_full.fit(np.arange(len(y_full)).reshape(-1, 1), y_full)
                future_preds_dict['Random Forest'] = rf_full.predict(X_future_idx)
            except: pass

            # 10. XGBoost Future
            try:
                X_future_idx = np.arange(len(y_full), len(y_full) + forecast_steps).reshape(-1, 1)
                xgb_full = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                xgb_full.fit(np.arange(len(y_full)).reshape(-1, 1), y_full)
                future_preds_dict['XGBoost'] = xgb_full.predict(X_future_idx)
            except: pass

            # 11. Prophet Future
            try:
                m_prophet_full = Prophet(changepoint_prior_scale=changepoint_prior, 
                                        seasonality_prior_scale=seasonality_prior,
                                        seasonality_mode=seasonality_mode)
                prophet_df_full = df[[time_col, target_col]].rename(columns={time_col:'ds', target_col:'y'})
                m_prophet_full.fit(prophet_df_full)
                future_df = m_prophet_full.make_future_dataframe(periods=forecast_steps, freq='D')
                fcst = m_prophet_full.predict(future_df)
                future_preds_dict['Prophet'] = fcst.iloc[-forecast_steps:]['yhat'].values
            except: pass

            # 12. Linear Regression Future
            try:
                X_future_idx = np.arange(len(y_full), len(y_full) + forecast_steps).reshape(-1, 1)
                lr_full = LinearRegression()
                lr_full.fit(np.arange(len(y_full)).reshape(-1, 1), y_full)
                future_preds_dict['Linear Regression'] = lr_full.predict(X_future_idx)
            except: pass

            # 13. Ridge Regression Future
            try:
                X_future_idx = np.arange(len(y_full), len(y_full) + forecast_steps).reshape(-1, 1)
                ridge_full = Ridge(alpha=1.0, random_state=42)
                ridge_full.fit(np.arange(len(y_full)).reshape(-1, 1), y_full)
                future_preds_dict['Ridge'] = ridge_full.predict(X_future_idx)
            except: pass

            # 14. Elastic Net Future
            try:
                X_future_idx = np.arange(len(y_full), len(y_full) + forecast_steps).reshape(-1, 1)
                elastic_full = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
                elastic_full.fit(np.arange(len(y_full)).reshape(-1, 1), y_full)
                future_preds_dict['Elastic Net'] = elastic_full.predict(X_future_idx)
            except: pass

            # 15. Huber Regressor Future
            try:
                X_future_idx = np.arange(len(y_full), len(y_full) + forecast_steps).reshape(-1, 1)
                huber_full = HuberRegressor(epsilon=1.35, max_iter=1000)
                huber_full.fit(np.arange(len(y_full)).reshape(-1, 1), y_full)
                future_preds_dict['Huber'] = huber_full.predict(X_future_idx)
            except: pass

            # -----------------------------------------------------------------------------
            # 4. Render Dashboard
            # -----------------------------------------------------------------------------
            
            with st.container(border=True):
                
                #st.markdown(f"**Data Range:** {dates_full.min().date()} to {dates_full.max().date()} | **Forecast Horizon:** {forecast_steps} days")
                #st.markdown(" ")
                
                k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
                
                k1.metric(label="📅Start Date", value=dates_full.min().strftime("%Y-%m-%d"))
                k2.metric(label="📅End Date", value=dates_full.max().strftime("%Y-%m-%d"))
                k3.metric(label="🔮Forecast Horizon", value=f"{forecast_steps} Days")                
                k4.metric("🏆Best Model", best_model_name, delta="Lowest RMSE")
                k5.metric("📉Best RMSE", f"{best_model_row['RMSE']:,.2f}", delta="-Optimized", delta_color="inverse")
                k6.metric("📊Test Set Size", f"{len(test)} rows")
                #k7.metric("⚖️Stationarity", "Yes" if stat_results['ADF'] and stat_results['ADF']['is_stationary'] else "No (Diff Applied)")


            #----------------------------------------
            col_basic1, col_basic2, col_basic3 = st.columns((0.3,0.5,0.2))
            
            with col_basic1:   
                         
                with st.popover("**OverView**",disabled=False, use_container_width=True):
                
                    unwanted_substrings = ['unnamed', '0', 'nan', 'deleted']
                    cols_to_delete = [col for col in df.columns if any(sub in col.lower() for sub in unwanted_substrings)]
                    if cols_to_delete:
                        st.warning(f"🗑️ {len(cols_to_delete)} column(s) deleted. | **Showing Top 3 rows for reference.**")
                    else:
                        st.info("✅ No unwanted columns found. Nothing deleted after importing. | **Showing Top 3 rows for reference.**")
                    df= df.drop(columns=cols_to_delete)
                    if add_lags:
                        for lag in range(1, num_lags + 1):
                            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag) 
                    st.dataframe(df.head(3)) 

            with col_basic2: 
                           
                with st.popover("**Statistics**",disabled=False, use_container_width=True):
                    
                    df_describe_table = df.describe(include='all').reset_index().rename(columns={'index': 'Feature'})
                    st.markdown("##### 📊 Descriptive Statistics")
                    st.dataframe(df_describe_table) 

            with col_basic3:   
                         
                with st.popover("**Remarks**",disabled=False, use_container_width=True):
                    
                    decomposition = seasonal_decompose(df[target_col], model='additive', period=12)                                
                    remarks = []
                    if decomposition.trend.dropna().std() > 0.5:
                        remarks.append("🔹 Strong **trend** detected.")
                    else:
                        remarks.append("🔹 Weak or no significant trend.")
                    if decomposition.seasonal.dropna().std() > 0.5:
                        remarks.append("🔹 Clear **seasonality** present.")
                    else:
                        remarks.append("🔹 Little to no seasonality detected.")
                    if decomposition.resid.dropna().std() < 0.2:
                        remarks.append("🔹 Residuals show **low variance** — good model fit.")
                    else:
                        remarks.append("🔹 Residuals are **noisy** — may need model tuning.")
                    st.markdown("##### 📌 Intial Remarks:")
                    for remark in remarks:
                        st.markdown(remark)                     

            #----------------------------------------                    
            #t.divider()
            #tabs = st.tabs(["🔮 Forecast", "📷 Visualizations", "🧪 Diagnostics"])

            # === COLORFUL BOLD TABS CSS ===
            st.markdown("""
            <style>
            .stTabs [data-baseweb="tab-list"] { gap: 10px; padding: 8px; background: #f8f9fa; border-radius: 15px; border: 2px solid #e9ecef; }
            .stTabs [data-baseweb="tab"] { font-weight: 900 !important; font-size: 15px !important; padding: 14px 28px !important; border-radius: 12px !important; background: white !important; border: 2px solid #dee2e6 !important; transition: all 0.25s ease !important; color: #343a40 !important; }
            .stTabs [data-baseweb="tab"]:hover { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; border-color: #764ba2 !important; box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); transform: translateY(-2px); }
            .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; border-color: #764ba2 !important; font-weight: 800 !important; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6); }
            </style>
            """, unsafe_allow_html=True)
            tabs = st.tabs(["🔮 Forecast", "📷 Visualizations", "🧪 Diagnostics"])

            with tabs[0]:
                
                with st.container(border=True):
                    
                    #st.subheader(f"Future Projection ({forecast_steps} Days)")
                    #st.markdown(f"##### Future Projection ({forecast_steps} Days)")
                    
                    fig_future = go.Figure()
                    fig_future.add_trace(go.Scatter(
                        x=dates_full, 
                        y=y_full, 
                        mode='lines', 
                        name='Historical', 
                        line=dict(color='#212529', width=2),
                        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Value</b>: %{y:,.2f}<extra></extra>'
                    ))

                    for name, preds in future_preds_dict.items():
                        fig_future.add_trace(go.Scatter(
                            x=future_dates, 
                            y=preds, 
                            mode='lines', 
                            name=f'{name} Forecast', 
                            line=dict(width=3)
                        ))

                    fig_future.add_vrect(
                        x0=dates_full.iloc[-1], 
                        x1=future_dates[-1], 
                        fillcolor="#0d6efd", 
                        opacity=0.1, 
                        layer="below", 
                        line_width=0,
                        annotation_text="Forecast Period",
                        annotation_position="top right",
                        annotation_font_size=12,
                        annotation_font_color="#0d6efd"
                    )

                    fig_future.update_layout(
                        height=600, 
                        title=f"Future Forecast (Next {forecast_steps} Days)", 
                        template="plotly_white", 
                        hovermode='x unified',
                        xaxis_title="Date",
                        yaxis_title="Value",
                        
                        # === UPDATED LEGEND (Right Side, Vertical) ===
                        legend=dict(
                            orientation="v",              # Vertical orientation
                            x=1.05,                       # Position outside the plot area (right side)
                            y=1,                          # Align to top
                            xanchor="left",               # Anchor point for x positioning
                            yanchor="top",                # Anchor point for y positioning
                            bgcolor='rgba(255,255,255,0.9)',  # Semi-transparent white background
                            bordercolor="#ced4da",        # Light gray border
                            borderwidth=1,                # Border thickness
                            font=dict(size=11, family="Inter, sans-serif"),  # Readable font size
                            traceorder="normal",          # Order of legend items
                            itemclick="toggleothers",     # Click to toggle only this item
                            itemdoubleclick="toggle"      # Double-click to toggle all others
                        ),
                        
                        xaxis=dict(
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=1, label="1m", step="month", stepmode="backward"),
                                    dict(count=6, label="6m", step="month", stepmode="backward"),
                                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                                    dict(count=1, label="1y", step="year", stepmode="backward"),
                                    dict(step="all", label="All")
                                ]),
                                bgcolor="#e9ecef",
                                activecolor="#0d6efd",
                                bordercolor="#ced4da",
                                borderwidth=1,
                                font=dict(size=15, family="Inter, sans-serif", color="#212529")
                            ),
                            rangeslider=dict(
                                visible=True,
                                bgcolor="#f8f9fa",
                                borderwidth=1,
                                bordercolor="#ced4da",
                                thickness=0.15
                            ),
                            type="date",
                            tickfont=dict(size=12)
                        ),
                        yaxis=dict(tickformat=",", tickfont=dict(size=12))
                    )

                    st.plotly_chart(fig_future, use_container_width=True)

                with st.container(border=True):    

                    res_df = pd.DataFrame({"Date": future_dates})
                    res_df['Date'] = res_df['Date'].dt.strftime('%Y-%m-%d')
                    for name, preds in future_preds_dict.items():
                        res_df[name] = preds
                    
                    numeric_cols = [c for c in res_df.columns if c != 'Date']
                    styled_df = res_df.style.format("{:.2f}", subset=numeric_cols)\
                                                .background_gradient(cmap="Blues", subset=numeric_cols)\
                                                .set_properties(**{'text-align': 'center'})
                    st.dataframe(styled_df, height=400)
                    
                    csv = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Forecast CSV", csv, "forecast_results.csv", "text/csv")

            with tabs[1]:
                
                #col_chart, col_table = st.columns([3,1])
                
                #with col_chart:
                    #with st.container(border=True):
                        #st.subheader("Actual vs. Predicted (Test Set)")
                        #st.markdown("##### Actual vs. Predicted (Test Set)")
                        #fig = make_subplots(rows=1, cols=1)
                        #fig.add_trace(go.Scatter(x=dates_test, y=y_test, mode='lines', name='Actual', line=dict(color='#212529', width=2)))
                        #colors = ['#0d6efd', '#198754', '#dc3545', '#ffc107', '#6f42c1']
                        #for i, (name, preds) in enumerate(models_pred.items()):
                            #if len(preds) == len(y_test):
                                #fig.add_trace(go.Scatter(x=dates_test, y=preds, mode='lines', name=name, line=dict(dash='dash', color=colors[i % len(colors)])))
                        #fig.update_layout(height=500, hovermode='x unified', template="plotly_white", legend=dict(orientation="h", y=1.02))
                        #st.plotly_chart(fig, use_container_width=True)

                #with col_table:
                with st.container(border=True):
                        
                    styled_df = metrics_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}%", "R2": "{:.4f}"}) \
                            .apply(lambda x: ['background-color: #d1e7dd; color: #0f5132; font-weight:bold' if x.name == metrics_df['RMSE'].idxmin() else '' for _ in x], axis=1)
                    st.dataframe(styled_df, hide_index=True, use_container_width=True)
                    st.info(f"💡 **Insight:** The **{best_model_name}** achieved the lowest error. Consider using it for final projections.")

                with st.container(border=True):
                        
                    st.markdown("##### 🎢 Rolling Statistics")
                    df.sort_index(inplace=True)
                    rolling_mean = df[target_col].rolling(window=12).mean()
                    rolling_std = df[target_col].rolling(window=12).std()
                                        
                    #fig3, ax2 = plt.subplots(figsize=(20,3))
                    #ax2.plot(df[target_col], label='Original', color='blue')
                    #ax2.plot(rolling_mean, label='Rolling Mean', color='orange')
                    #ax2.plot(rolling_std, label='Rolling Std Dev', color='green')
                    #ax2.set_title(f"Rolling Statistics of {target_col} over Time", fontsize=10)
                    #ax2.set_xlabel(time_col)
                    #ax2.set_ylabel(target_col)
                    #ax2.legend(fontsize=10)
                    #ax2.grid(True)
                    #st.pyplot(fig3, use_container_width=True)    
                                    
                    fig3 = go.Figure()

                    # Original series
                    fig3.add_trace(go.Scatter(
                        x=df.index,
                        y=df[target_col],
                        mode='lines',
                        name='Original',
                        line=dict(color='blue', width=2),
                        opacity=0.7,
                        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Value</b>: %{y:,.2f}<extra></extra>'
                    ))

                    # Rolling Mean
                    fig3.add_trace(go.Scatter(
                        x=df.index,
                        y=rolling_mean,
                        mode='lines',
                        name='Rolling Mean (12)',
                        line=dict(color='orange', width=3),
                        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Rolling Mean</b>: %{y:,.2f}<extra></extra>'
                    ))

                    # Rolling Std Dev
                    fig3.add_trace(go.Scatter(
                        x=df.index,
                        y=rolling_std,
                        mode='lines',
                        name='Rolling Std Dev (12)',
                        line=dict(color='green', width=3),
                        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Rolling Std Dev</b>: %{y:,.2f}<extra></extra>'
                    ))

                    # Update layout
                    fig3.update_layout(
                        title=f"Rolling Statistics of {target_col} over Time",
                        template="plotly_white",
                        hovermode='x unified',
                        height=400,
                        xaxis_title=time_col,
                        yaxis_title=target_col,
                        legend=dict(orientation="h", y=1.02, x=0, bgcolor='rgba(255,255,255,0.8)'),
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(0,0,0,0.1)',
                            rangeselector=dict(buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(step="all", label="All")
                            ]))
                        ),
                        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickformat=",")
                    )

                    st.plotly_chart(fig3, use_container_width=True)
                                    
                with st.container(border=True):
                        
                        st.markdown("##### 🧪 Decomposition ")
                        remarks, decomp_obj = get_decomposition_insights(df[target_col], period=12)
                        if decomp_obj:
                            fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'))
                            fig_decomp.add_trace(go.Scatter(x=dates_full, y=decomp_obj.observed, name="Observed"), row=1, col=1)
                            fig_decomp.add_trace(go.Scatter(x=dates_full, y=decomp_obj.trend, name="Trend"), row=2, col=1)
                            fig_decomp.add_trace(go.Scatter(x=dates_full, y=decomp_obj.seasonal, name="Seasonal"), row=3, col=1)
                            fig_decomp.add_trace(go.Scatter(x=dates_full, y=decomp_obj.resid, name="Residual"), row=4, col=1)
                            fig_decomp.update_layout(height=600, showlegend=False, template="plotly_white")
                            st.plotly_chart(fig_decomp, use_container_width=True)
            
            with tabs[2]:

                with st.container(border=True):
                    
                    st.markdown("##### Transformations ")
                    if st.session_state['transformations']:
                        for t in st.session_state['transformations']:
                            st.write(f"✅ {t}")
                    else:
                        st.write("No transformations applied.")

                with st.container(border=True):
                    
                    st.markdown("##### 🧹 Missing Values ")

                    col1, col2 = st.columns((0.2,0.8))
                    with col1:
                            
                        if missing.empty:
                            st.success("**✅ No missing values found!**")
                                        
                        else:
                            st.warning("**⚠️ Missing values found!**")
                            st.write("**Number of missing values:**")
                            st.table(missing)

                            with col2:   
                                                            
                                st.write("**Missing Values Treatment:**")  
                                #df = handle_categorical_missing_values(df, selected_categorical_strategy)   
                                st.table(df.head(2))

                # --- 3. Stationarity Tests Results (USE PRE-COMPUTED VALUES) ---
                with st.container(border=True):
                    
                    st.markdown("##### 🧪 Stationarity Tests Results")
                    
                    # Display the pre-computed test_results dataframe
                    st.dataframe(test_results, hide_index=True, use_container_width=True)
                    
                    st.divider()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**ADF Test (Null: Non-Stationary)**")
                        if not np.isnan(adf_p) and adf_p < 0.05:
                            st.success("Reject H₀ → ✅ Series is already **stationary**. No differencing needed.")
                        else:
                            st.warning("Fail to reject H₀ → ❗Series is **non-stationary** — differencing required.")
                    with col2:
                        st.write("**KPSS Test (Null: Stationary)**")
                        if not np.isnan(kpss_p) and kpss_p > 0.05:
                            st.success("Fail to reject H₀ → ✅ Series is already **stationary**. No differencing needed.")
                        else:
                            st.warning("Reject H₀ → ❗Series is **non-stationary** — differencing required.")

                # --- 4. Differencing Visualization (USE PRE-COMPUTED d & diff_series) ---
                with st.container(border=True):
                    
                    d = st.session_state.get('d', 0)
                    diff_series = st.session_state.get('diff_series', series)
                    st.markdown(f"##### Required Differencing Order: `d = {d}`")
                    
                    #fig, ax = plt.subplots(figsize=(20, 3))
                    #ax.plot(series.index, series, label="Original", alpha=0.7, color='blue')
                    #if d > 0:
                        #ax.plot(diff_series.index, diff_series, label=f"Differenced (d={d})", color='red', linewidth=2)
                    #ax.legend()
                    #ax.set_title("Original vs Differenced Series", fontsize=10)
                    #ax.grid(True, alpha=0.3)
                    #st.pyplot(fig, use_container_width=True)
                    
                    fig = go.Figure()

                    # Original series
                    fig.add_trace(go.Scatter(
                        x=series.index,
                        y=series.values,
                        mode='lines',
                        name='Original',
                        line=dict(color='blue', width=2),
                        opacity=0.7,
                        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Value</b>: %{y:,.2f}<extra></extra>'
                    ))

                    # Differenced series (if applicable)
                    if d > 0:
                        fig.add_trace(go.Scatter(
                            x=diff_series.index,
                            y=diff_series.values,
                            mode='lines',
                            name=f'Differenced (d={d})',
                            line=dict(color='red', width=2),
                            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Value</b>: %{y:,.2f}<extra></extra>'
                        ))

                    # Update layout
                    fig.update_layout(
                        title="Original vs Differenced Series",
                        template="plotly_white",
                        hovermode='x unified',
                        height=300,  # Compact height similar to matplotlib figsize=(20,3)
                        xaxis_title="Date",
                        yaxis_title="Value",
                        legend=dict(orientation="h", y=1.02, x=0, bgcolor='rgba(255,255,255,0.8)'),
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(0,0,0,0.1)',
                            rangeselector=dict(buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(step="all", label="All")
                            ]))
                        ),
                        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickformat=",")
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    if d > 0:
                        st.info(f"💡 **Note:** First **{d} row(s)** were dropped due to differencing. Modeling uses the stationary differenced series.")
                    else:
                        st.success("✅ Series is stationary — no differencing was applied.")

                # --- 5. Diagnostic Plots on Final Series (USE PRE-COMPUTED diff_series) ---
                with st.container(border=True):
                    st.markdown("##### 📊 Diagnostic Plots on Final Series")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                        
                        # Histogram
                        axes[0, 0].hist(diff_series.dropna(), bins=20, color='skyblue', edgecolor='k', alpha=0.8)
                        axes[0, 0].axvline(diff_series.dropna().mean(), color='red', linestyle='--', label=f"Mean: {diff_series.dropna().mean():.2f}")
                        axes[0, 0].set_title("Histogram of Final Series")
                        axes[0, 0].legend(fontsize=8)
                        axes[0, 0].grid(True, alpha=0.3)
                        
                        # Periodogram (Power Spectral Density)
                        f, Pxx = periodogram(diff_series.dropna())
                        axes[0, 1].semilogy(f, Pxx, color='purple')
                        axes[0, 1].set_title("Periodogram")
                        axes[0, 1].set_xlabel("Frequency")
                        axes[0, 1].set_ylabel("Power")
                        axes[0, 1].grid(True, alpha=0.3)
                        
                        # Q-Q Plot
                        stats.probplot(diff_series.dropna(), dist="norm", plot=axes[1, 0])
                        axes[1, 0].set_title("Q-Q Plot (Normality)")
                        
                        # Clear unused subplot
                        axes[1, 1].axis("off")
                        axes[1, 1].text(0.5, 0.5, "Diagnostics Complete ✓", 
                                    ha='center', va='center', fontsize=10, style='italic', color='green')
                        
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)

                    with col2:
                        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
                        
                        # ACF Plot
                        plot_acf(diff_series.dropna(), ax=ax1, lags=20, alpha=0.05)
                        ax1.set_title("Autocorrelation Function (ACF)", fontsize=10)
                        ax1.grid(True, alpha=0.3)
                        
                        # PACF Plot
                        plot_pacf(diff_series.dropna(), ax=ax2, lags=20, alpha=0.05)
                        ax2.set_title("Partial Autocorrelation Function (PACF)", fontsize=10)
                        ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig2, use_container_width=True)
                        
                        # Interpretation helper
                        with st.expander("📖 How to Interpret ACF/PACF"):
                            st.markdown("""
                            - **ACF**: Significant spikes suggest MA(q) terms. Cut-off after lag q → try MA(q).
                            - **PACF**: Significant spikes suggest AR(p) terms. Cut-off after lag p → try AR(p).
                            - **Both tail off**: May need ARMA(p,q) or differencing.
                            - **Seasonal spikes**: Consider seasonal ARIMA (SARIMA) with period `m`.
                            """)

    else:
        st.info("Click **▶️ Run Forecasting** to start the analysis.")
