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
#---------------------------------------------------------------------------------------------------------------------------------
### Title for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Forecasting | v1.0",
                    layout="wide",
                    page_icon="📈",            
                    initial_sidebar_state="collapsed")
#---------------------------------------------------------------------------------------------------------------------------------
### CSS
#---------------------------------------------------------------------------------------------------------------------------------
st.markdown(
        """
        <style>
        .centered-info {display: flex; justify-content: center; align-items: center; 
                        font-weight: bold; font-size: 15px; color: #007BFF; 
                        padding: 5px; background-color: #FFFFFF;  border-radius: 5px; border: 1px solid #007BFF;
                        margin-top: 0px;margin-bottom: 5px;}
        .stMarkdown {margin-top: 0px !important; padding-top: 0px !important;}                       
        </style>
        """,unsafe_allow_html=True,)

#---------------------------------------------------------------------------------------------------------------------------------
### Description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .title-large {
        text-align: center;
        font-size: 35px;
        font-weight: bold;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .title-small {
        text-align: center;
        font-size: 20px;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    <div class="title-large">Forecasting Studio</div>
    <div class="title-small">Version : 1.0</div>
    """,
    unsafe_allow_html=True)

#----------------------------------------
st.markdown('<div class="centered-info"><span style="margin-left: 10px;">A lightweight streamlit app that help to forecast different time-series problems</span></div>',unsafe_allow_html=True,)
#----------------------------------------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F0F2F6;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
        z-index: 100;
    }
    .footer p {
        margin: 0;
    }
    .footer .highlight {
        font-weight: bold;
        color: blue;
    }
    </style>

    <div class="footer">
        <p>© 2025 | Created by : <span class="highlight">Avijit Chakraborty</span> | <a href="mailto:avijit.mba18@gmail.com"> 📩 </a></p> <span class="highlight">Thank you for visiting the app | Unauthorized uses or copying is strictly prohibited | For best view of the app, please zoom out the browser to 75%.</span>
    </div>
    """,
    unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------
def evaluate(pred, true):
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true - pred) / (true + 1e-8))) * 100
    r2 = r2_score(true, pred)
    return mae, mse, rmse, mape, r2

def color_objective(val):
    color = "#d1ecf1" if val == "Maximize" else "#f8d7da"
    return f"background-color: {color}; color: #004c6d;" if val == "Maximize" else f"background-color: {color}; color: #721c24;"

#---------------------------------------------------------------------------------------------------------------------------------
### Main app
#---------------------------------------------------------------------------------------------------------------------------------
if 'models_pred' not in st.session_state:
    st.session_state.models_pred = None
if 'error_df' not in st.session_state:
    st.session_state.error_df = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'test_dates' not in st.session_state:
    st.session_state.test_dates = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'transformations' not in st.session_state:
    st.session_state.transformations = []
#----------------------------------------
    
col1, col2 = st.columns((0.15,0.85))
with col1:           
    with st.container(border=True):

        uploaded_file = st.file_uploader("**:blue[Choose a file]**",type=["csv", "xls", "xlsx"], accept_multiple_files=False,key="file_upload")
        if not uploaded_file:
            st.info("Please upload a CSV file to begin.")
            st.stop()
        
        if uploaded_file is not None:
            st.success("Data loaded successfully!")
            df = load_file(uploaded_file)        #for filter
                
            st.divider()
            ts_type = st.radio("**Time Series Type**", ["Univariate", "Multivariate"],horizontal=True)

    with st.container(border=True):
        
        time_col = st.selectbox("**Select Time Column**", df.columns.tolist())
        try:
            df[time_col] = pd.to_datetime(df[time_col],errors='coerce')
        except Exception as e:
            st.error("Time column parsing failed.")
            st.stop()
            
        target_col = st.selectbox("**Target Variable**", [c for c in df.columns if c != time_col])
        feature_cols = [target_col]
        if ts_type == "Multivariate":
            feature_cols = st.multiselect("**Feature Variable**", [c for c in df.columns if c != time_col], default=[target_col])
            
    with st.expander('**⚙️ Preprocessing Options**'):
        
        apply_resample = st.checkbox("Apply Resampling")
        if apply_resample:
            with st.expander("Resampling"):
                freq_map = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly', 'Y': 'Yearly'}
                selected_freq = st.selectbox("**Resample Frequency**", list(freq_map.keys()), format_func=lambda x: freq_map[x])
        else:
            selected_freq = None

        apply_log = st.checkbox("Apply Log Transform")
        log_col_name = None
        if apply_log:
             with st.expander("Log Transform"):
                if (df[target_col] <= 0).any():
                    st.warning("Log transform not possible: non-positive values.")
                else:
                    log_col_name = f"{target_col}_log"

        apply_fourier = st.checkbox("Add Fourier Terms (Multiple Seasonalities)")
        if apply_fourier:
             with st.expander("Fourier Terms"):
                period1 = st.number_input("**Seasonal Period 1**", min_value=2, value=7)
                period2 = st.number_input("**Seasonal Period 2**", min_value=2, value=365)
                num_terms = st.slider("**Fourier Pairs**", 1, 10, 3)
        else:
            period1 = period2 = num_terms = None
            
    with st.container(border=True):
        
        forecast_steps = st.slider("**Forecast Steps (days)**", 1, 365, 30)
        #run_forecast = st.button("**▶️ Run Forecasting**")

#----------------------------------------
with col2:             
    # Initialize session state
    if 'df_processed' not in st.session_state or st.session_state.get('uploaded_file') != uploaded_file.name:
        st.session_state['df_processed'] = df.copy()
        st.session_state['transformations'] = []
        st.session_state['uploaded_file'] = uploaded_file.name

    #if run_forecast:
        df_temp = df.copy()
        transformations = []

        if apply_resample and selected_freq:
            try:
                df_temp = df_temp.set_index(time_col).resample(selected_freq).mean(numeric_only=True).reset_index()
                transformations.append("Resampled")
            except:
                st.sidebar.error("Resampling failed.")

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
        st.session_state['target_col'] = target_col
        st.session_state['feature_cols'] = feature_cols
        st.session_state['transformations'] = transformations

        if transformations:
            st.info("✅ Applied: " + ", ".join(transformations))
    #else:
        #st.info("Click **▶️ Run Forecasting** to start.")
        #st.stop()

    df = st.session_state['df_processed']
    target_col = st.session_state['target_col']
    feature_cols = st.session_state['feature_cols']

    with st.popover("**:blue[:hammer_and_wrench: Hyperparameters]**",disabled=False, use_container_width=True):  

                            subcol1, subcol2, subcol3, subcol4, subcol5 = st.columns(5)
                            with subcol1: 
                                decom_model_type = st.selectbox("**Model type for decomposition**", ["additive", "multiplicative", ])
                                st.divider()                           
                                numerical_strategies = ['mean', 'median', 'most_frequent', 'ffill', 'interpolate']
                                categorical_strategies = ['constant','most_frequent']
                                selected_numerical_strategy = st.selectbox("**Missing value treatment : Numerical**", numerical_strategies)
                                selected_categorical_strategy = st.selectbox("**Missing value treatment : Categorical**", categorical_strategies)  
                                st.divider()
                                resample_freq_down = st.selectbox("**Downsampling frequency**",options=["W", "M", "Q", "Y"],format_func=lambda x: {"W": "Weekly", "M": "Monthly", "Q": "Quarterly", "Y": "Yearly"}[x])
                                agg_method = st.selectbox("**Aggregation method**", ["mean", "sum", "median"])
                                resample_freq_up = st.selectbox("**Upsampling frequency**",options=["H", "6H", "D"],format_func=lambda x: {"H": "Hourly", "6H": "Every 6 Hours", "D": "Daily"}[x])
                                interp_method = st.selectbox("**Interpolation method**", ["linear", "spline", "quadratic", "cubic"])
                                
                            with subcol2:  
                                freq_guess = st.number_input("**Seasonal Frequency (e.g. 12 for monthly, 7 for weekly)**", min_value=2, max_value=365, value=12)
                                st.divider()    
                                max_diff = st.selectbox("**Maximum Differencing Steps Allowed**", options=[1,2,3,4,5,6], index=2)
                                st.divider()
                                lags_val = st.slider("**Select number of lags**", min_value=10, max_value=100, value=40, step=5)
                                pacf_method = st.selectbox("**PACF Method**", options=["ywm", "ols", "ldb", "ld", "ywunbiased", "ywadjusted"], index=0) 
                                                                                           
                            with subcol3: 
                                    train = st.slider("**Train Size (as %)**", 10, 90, 70, 5)
                                    test = st.slider("**Test Size (as %)**", 10, 50, 30, 5)    
                                    random_state = st.number_input("**Random State**", 0, 100, 42)
                                    n_jobs = st.number_input("**Parallel Processing (n_jobs)**", -10, 10, 1) 

                            with subcol4: 
                                
                                with st.expander("**📌 Parameters | Smoothning**", expanded=False):
                                    alpha = st.slider('**Alpha (Smoothing Parameter)**', min_value=0.01, max_value=1.0, value=0.2,key = 'ses_1')
                                    beta = st.slider('**Beta (Trend Smoothing Parameter)**', min_value=0.01, max_value=1.0, value=0.2,key = 'ses_2')
                                    gamma = st.slider('**Gamma (Seasonality Smoothing Parameter)**', min_value=0.01, max_value=1.0, value=0.2,key = 'ses_3') 
                                st.divider()  
                                  
                                #order_arima = st.text_input('**ARIMA Order (p,d,q)**:', '1,1,1')
                                #order_arima = tuple(map(int, order_arima.split(',')))     
                                #order_sarima = st.text_input('**SARIMA Order (p,d,q,m)**:', '1,1,1,12')
                                #order_sarima = tuple(map(int, order_sarima.split(',')))    
                                
                                with st.expander("**Tune ARIMA**"):
                                        arima_p = st.slider("**ARIMA p (max)**", 0, 5, 2)
                                        arima_d = st.slider("**ARIMA d (max)**", 0, 3, 2)
                                        arima_q = st.slider("**ARIMA q (max)**", 0, 5, 2)

                                with st.expander("**Tune Prophet**"):
                                        changepoint_prior = st.slider("**Prophet: Changepoint Prior**", 0.001, 1.0, 0.05, 0.01)
                                        seasonality_prior = st.slider("**Seasonality Prior**", 0.01, 10.0, 1.0, 0.1)
                                        seasonality_mode = st.selectbox("**Seasonality Mode**", ["additive", "multiplicative"])

                                with st.expander("**Tune Random Forest & XGBoost**"):
                                        n_estimators = st.slider("**n_estimators**", 50, 200, 100)
                                        max_depth = st.slider("**max_depth**", 3, 10, 5)

                            with subcol5:
                                selected_metric = st.selectbox("🔍 Select Error Metric to Find Best Model", ["MAE", "MSE", "RMSE", "MAPE"])
                                st.divider() 
                                model = st.selectbox("**Select Model**", ["l2", "l1", "rbf"], index=0)
                                algo_name = st.selectbox("**Select Algorithm**", ["Binary Segmentation", "Pelt", "Window", "Bottom-Up"], index=0)
                                num_change_points = st.slider("**Number of Change Points**", min_value=1, max_value=20, value=3)                    
    
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    tabs = st.tabs(["**📊 Overview**","**📈 Visualizations**","**🔧 Preprocessing**","**✅ Checks**","**⚖️ Comparison**","**📈 Graph**","**🎲 Forecast**", "**⚠︎ Drift**"])
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    with tabs[0]:
        
        st.dataframe(df.head(3)) 
        
        with st.container(border=True):
                                    
            df_describe_table = df.describe(include='all').T.reset_index().rename(columns={'index': 'Feature'})
            st.markdown("##### 📊 Descriptive Statistics")
            st.dataframe(df_describe_table) 
            
        col1, col2 = st.columns((0.85,0.15))
        with col1: 
            with st.container(border=True):

                if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                df.dropna(subset=[time_col], inplace=True)  
                df.sort_values(by=time_col, inplace=True)
                fig, ax = plt.subplots(figsize=(16,4))
                ax.plot(df[time_col], df[target_col], label='Sales', color='blue')
                ax.set_title(f"{target_col} over {time_col}", fontsize=10)
                ax.set_xlabel(time_col)
                ax.set_ylabel(target_col)
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)                         

        with col2:                                
            with st.container(border=True):
                                    
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
                        
    with tabs[1]:
        
            with st.container(border=True):
                                
                st.markdown("##### 🎢 Rolling Statistics")
                df.sort_index(inplace=True)
                rolling_mean = df[target_col].rolling(window=12).mean()
                rolling_std = df[target_col].rolling(window=12).std()
                                
                fig3, ax2 = plt.subplots(figsize=(20,3))
                ax2.plot(df[target_col], label='Original', color='blue')
                ax2.plot(rolling_mean, label='Rolling Mean', color='orange')
                ax2.plot(rolling_std, label='Rolling Std Dev', color='green')
                ax2.set_title(f"Rolling Statistics of {target_col} over Time", fontsize=10)
                ax2.set_xlabel(time_col)
                ax2.set_ylabel(target_col)
                ax2.legend(fontsize=10)
                ax2.grid(True)
                st.pyplot(fig3, use_container_width=True)
                                
            with st.container(border=True):

                                st.markdown("##### 🧪 Decomposition ")
                                decomposition = seasonal_decompose(df[target_col], model='additive', period=12)
                                fig, axs = plt.subplots(4, 1, figsize=(20,10), sharex=True)
                                axs[0].plot(decomposition.observed, label="Observed")
                                axs[1].plot(decomposition.trend, label="Trend", color='orange')
                                axs[2].plot(decomposition.seasonal, label="Seasonality", color='green')
                                axs[3].plot(decomposition.resid, label="Residuals", color='red')
                                for ax in axs:
                                    ax.legend(loc='upper left')
                                    ax.grid(True)
                                st.pyplot(fig, use_container_width=True)
                                
    with tabs[2]:
        
            with st.container(border=True):
                
                st.markdown("##### Transformations ")
                if st.session_state['transformations']:
                    for t in st.session_state['transformations']:
                        st.write(f"✅ {t}")
                else:
                    st.write("No transformations applied.")

            with st.container(border=True):
                
                st.markdown("##### Missing Values ")
                missing = df[feature_cols].isnull().sum()
                st.write(missing[missing > 0] if missing.sum() > 0 else "No missing values.")
                
    with tabs[3]:
        
            with st.container(border=True):
                
                st.markdown("##### 🧪 Stationarity Tests ")
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
                    ]
                })

                st.dataframe(test_results,hide_index=True,use_container_width=True)
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    
                    st.write("**ADF Test (Null: Non-Stationary)**")
                    if adf_p < 0.05:
                        st.success("Reject H₀ → ✅ Series is already **stationary**. No differencing needed.")
                    else:
                        st.warning("Fail to reject H₀ → ❗Series is **non-stationary** — differencing required.")
                
                with col2:
                    st.write("**KPSS Test (Null: Stationary)**")
                    if kpss_p > 0.05:
                        st.success("Fail to reject H₀ → ✅ Series is already **stationary**. No differencing needed.")
                    else:
                        st.warning("Reject H₀ → ❗Series is **non-stationary** — differencing required.")
                        
            with st.container(border=True):
                
                d = 0
                diff_series = series.copy()
                max_d = 3
                while (adf_p >= 0.05 or (kpss_p <= 0.05 and kpss_p != np.nan)) and d < max_d:
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

                st.markdown(f"Required differencing order: `d = {d}`")

                fig, ax = plt.subplots(figsize=(20,3))
                ax.plot(series.index, series, label="Original", alpha=0.7)
                if d > 0:
                    ax.plot(diff_series.index, diff_series, label=f"Differenced (d={d})", color='red')
                ax.legend()
                ax.set_title("Original vs Differenced Series",fontsize=10)
                st.pyplot(fig,use_container_width=True)
                
            with st.container(border=True):
                
                col1, col2 = st.columns(2)
                with col1:
                    fig, axes = plt.subplots(2,2, figsize=(10,8))
                    
                    # Histogram
                    axes[0,0].hist(diff_series, bins=20, color='skyblue', edgecolor='k')
                    axes[0,0].set_title("Histogram of Differenced Series")
                    
                    # Periodogram (Power Spectral Density)
                    f, Pxx = periodogram(diff_series)
                    axes[0,1].semilogy(f, Pxx)
                    axes[0,1].set_title("Periodogram")
                    axes[0,1].set_xlabel("Frequency")
                    axes[0,1].set_ylabel("Power")
                    
                    # Q-Q Plot
                    stats.probplot(diff_series, dist="norm", plot=axes[1,0])
                    axes[1,0].set_title("Q-Q Plot (Normality)")
                    
                    # Clear unused subplot
                    axes[1,1].axis("off")
                    plt.tight_layout()
                    st.pyplot(fig,use_container_width=True)

                with col2:
                    fig2, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6))
                    plot_acf(diff_series, ax=ax1, lags=20)
                    plot_pacf(diff_series, ax=ax2, lags=20)
                    plt.tight_layout()
                    st.pyplot(fig2,use_container_width=True)
                    
    with tabs[4]:
        
            with st.container(border=True):                    
                    
                split_idx = int(len(df) * 0.8)
                train, test = df[:split_idx], df[split_idx:]
                y_train, y_test = train[target_col].values, test[target_col].values
                X_train = np.arange(len(y_train)).reshape(-1, 1)
                X_test = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)

                #st.write(f"Train size: {len(train)} | Test size: {len(test)}")
                col1, col2= st.columns(2)
                with col1:
                    st.markdown("##### ✅ **Final dataset | for modeling**")
                    st.info(f"""
                        - Train set size: **{train.shape[0]} rows**  
                        - Test set size: **{test.shape[0]} rows**
                            """)                  

                with col2:
                            
                    try:
                        acf_vals = acf(diff_series, nlags=10)
                        pacf_vals = pacf(diff_series, nlags=10)
                        p = np.where(pacf_vals[1:] < 0.1)[0][0] if np.any(pacf_vals[1:] < 0.1) else 1
                        q = np.where(acf_vals[1:] < 0.1)[0][0] if np.any(acf_vals[1:] < 0.1) else 1
                        st.markdown("##### 🤖 **Suggested ARIMA Order (p, d, q)**")
                        st.info(f"""
                                    - **p (AR)** suggested from PACF: `{p}`
                                    - **d (Differencing)** based on stationarity tests: `{d}`
                                    - **q (MA)** suggested from ACF: `{q}`
                                    """)

                    except:
                            st.write("Suggested ARIMA order: (1,1,1)")           
                            
            col1, col2= st.columns((0.7,0.3))
            with col1:
                with st.container(border=True):          
                
                        models_pred = {}
                        errors = []

                        # Moving Average
                        window = 5
                        ma_pred = np.convolve(y_train, np.ones(window)/window, mode='valid')
                        ma_pred = np.concatenate([ma_pred, [ma_pred[-1]] * len(y_test)])[-len(y_test):]
                        models_pred["Moving Average"] = ma_pred
                        errors.append(("Moving Average", *evaluate(ma_pred, y_test)))

                        # Smoothing
                        ses = SimpleExpSmoothing(y_train).fit()
                        ses_pred = ses.forecast(len(y_test))
                        models_pred["Smoothing (SES)"] = ses_pred
                        errors.append(("Smoothing (SES)", *evaluate(ses_pred, y_test)))

                        # Auto Regression
                        ar = AutoReg(y_train, lags=2).fit()
                        ar_pred = ar.forecast(steps=len(y_test))
                        models_pred["Auto Regression"] = ar_pred
                        errors.append(("Auto Regression", *evaluate(ar_pred, y_test)))

                        # ARIMA
                        arima_order = (p, d, q) if 'p' in locals() else (1, d, 1)
                        arima = ARIMA(y_train, order=arima_order).fit()
                        arima_pred = arima.forecast(steps=len(y_test))
                        models_pred["ARIMA"] = arima_pred
                        errors.append(("ARIMA", *evaluate(arima_pred, y_test)))

                        # Random Forest
                        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=10)
                        rf.fit(X_train, y_train)
                        rf_pred = rf.predict(X_test)
                        models_pred["Random Forest"] = rf_pred
                        errors.append(("Random Forest", *evaluate(rf_pred, y_test)))

                        # XGBoost
                        xgb = XGBRegressor(n_estimators=n_estimators, max_depth=10)
                        xgb.fit(X_train, y_train)
                        xgb_pred = xgb.predict(X_test)
                        models_pred["XGBoost"] = xgb_pred
                        errors.append(("XGBoost", *evaluate(xgb_pred, y_test)))

                        # Prophet
                        try:
                            prophet_df = df[[time_col, target_col]].rename(columns={time_col: 'ds', target_col: 'y'})
                            m = Prophet(changepoint_prior_scale=changepoint_prior,
                                            seasonality_prior_scale=seasonality_prior,
                                            seasonality_mode=seasonality_mode)
                            m.fit(prophet_df[:split_idx])
                            future = m.make_future_dataframe(periods=len(y_test))
                            forecast = m.predict(future)
                            prophet_pred = forecast.iloc[-len(y_test):]['yhat'].values
                            models_pred["Prophet"] = prophet_pred
                            errors.append(("Prophet", *evaluate(prophet_pred, y_test)))
                        except:
                            st.warning("Prophet failed.")

                        error_df = pd.DataFrame(errors, columns=["Model", "MAE", "MSE", "RMSE", "MAPE (%)", "R²"])
                        best_models = {}
                        metrics = ["MAE", "MSE", "RMSE", "MAPE (%)"]
                        for metric in metrics:
                            best_models[metric] = error_df.loc[error_df[metric].idxmin(), "Model"]
                        best_models["R²"] = error_df.loc[error_df["R²"].idxmax(), "Model"]  # Higher is better
                        best_row = pd.DataFrame([[
                            "✅ Best Model",best_models["MAE"],best_models["MSE"],best_models["RMSE"],best_models["MAPE (%)"],best_models["R²"]]], columns=error_df.columns)

                        error_df_styled = pd.concat([error_df, best_row], ignore_index=True)
                        def highlight_best_row(row):
                            return ['background-color: #d4edda; color: #155724; font-weight: bold' if row.name == len(error_df) else '' for _ in row]
                        error_df_styled = error_df_styled.style.apply(highlight_best_row, axis=1)

                        st.markdown("##### 📊 **Model Performance Comparison**")
                        st.dataframe(error_df_styled, hide_index=True, use_container_width=True)
                        
                        st.session_state.models_pred = models_pred  # ✅ Assign after computation
                        st.session_state.y_test = y_test
                        st.session_state.test_dates = test[time_col].values

            with col2:
                with st.container(border=True):  
                         
                        st.markdown("##### 🏆 Best Model Summary")
                        best_summary_data = []
                        metrics = ["MAE", "MSE", "RMSE", "MAPE (%)", "R²"]
                        for metric in metrics:
                            direction = "Minimize" if metric != "R²" else "Maximize"
                            best_model = best_models[metric]
                            best_summary_data.append({"Metric": metric,"Best Model": best_model,"Objective": direction})
                        best_summary_df = pd.DataFrame(best_summary_data)
                        styled_summary = best_summary_df.style.applymap(color_objective, subset=["Objective"]) \
                                                    .set_properties(**{'text-align': 'center'}) \
                                                    .set_table_styles([
                                                        {"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}
                                                    ])
                        st.dataframe(styled_summary, use_container_width=True)

    with tabs[5]:
        
            if st.session_state.models_pred is None:
                    st.warning("No models found. Run forecasting first.")
                    st.stop()

            y_test = st.session_state.y_test
            test_dates = st.session_state.test_dates

            for name, pred in st.session_state.models_pred.items():
                with st.container(border=True):

                    st.write(f"Shapes: {name} - Test: {y_test.shape}, Predicted: {pred.shape}")
                    if len(y_test) != len(pred):
                        st.error(f"Mismatched lengths: Test ({len(y_test)}) ≠ Predicted ({len(pred)})")
                        continue
                    if len(test_dates) != len(pred):
                        st.error(f"Mismatched lengths: Dates ({len(test_dates)}) ≠ Predicted ({len(pred)})")
                        continue

                    fig, ax = plt.subplots(figsize=(30,5))
                    ax.plot(test_dates, y_test, label="Actual", color='blue')
                    ax.plot(test_dates, pred, label="Predicted", color='red', linestyle='--')
                    ax.set_title(f"{name}: Actual vs Predicted")
                    ax.legend()
                    ax.set_xlabel(time_col)
                    ax.set_ylabel(target_col)
                    st.pyplot(fig,use_container_width=True)
                    
    with tabs[6]:                    
                    
            y_full = df[target_col].values
            last_date = df[time_col].iloc[-1]
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

            all_forecasts = pd.DataFrame({"Date": future_dates})
            X_full = np.arange(len(y_full)).reshape(-1, 1)
            X_future = np.arange(len(y_full), len(y_full) + forecast_steps).reshape(-1, 1)

            for name in st.session_state['models_pred'].keys():
                preds = []
                
                if name == "Moving Average":
                    last_avg = np.mean(y_full[-5:])
                    preds = [last_avg] * forecast_steps
                    
                elif name == "Smoothing (SES)":
                    model = SimpleExpSmoothing(y_full).fit()
                    preds = model.forecast(forecast_steps)
                    
                elif name == "Auto Regression":
                    model = AutoReg(y_full, lags=2).fit()
                    preds = model.forecast(steps=forecast_steps)
                    
                elif name == "ARIMA":
                    model = ARIMA(y_full, order=(1, 1, 1)).fit()
                    preds = model.forecast(steps=forecast_steps)
                    
                elif name == "Random Forest":
                    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                    rf.fit(X_full, y_full)
                    preds = rf.predict(X_future)
                    
                elif name == "XGBoost":
                    xgb = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth)
                    xgb.fit(X_full, y_full)
                    preds = xgb.predict(X_future)
                    
                elif name == "Prophet":
                    try:
                        m = Prophet(changepoint_prior_scale=changepoint_prior,
                                        seasonality_prior_scale=seasonality_prior,
                                        seasonality_mode=seasonality_mode)
                        prophet_df = df[[time_col, target_col]].rename(columns={time_col: 'ds', target_col: 'y'})
                        m.fit(prophet_df)
                        future = m.make_future_dataframe(periods=forecast_steps)
                        forecast = m.predict(future)
                        preds = forecast.iloc[-forecast_steps:]['yhat'].values
                    except:
                        preds = [np.nan] * forecast_steps

                all_forecasts[name] = preds
            st.dataframe(all_forecasts)

            for col in all_forecasts.columns[1:]:
                with st.container(border=True):
                    fig, ax = plt.subplots(figsize=(30,5))
                    ax.plot(df[time_col], y_full, label="Historical", color="black")
                    ax.plot(future_dates, all_forecasts[col], label=f"{col} Forecast", color="orange")

                    # Enhance visibility of the forecast area
                    ax.axvspan(df[time_col].iloc[-1], future_dates[-1], alpha=0.1, color='gray')  # Shade forecast period
                    ax.set_xlim(df[time_col].iloc[-len(df)//2], future_dates[-1])  # Zoom into recent history + forecast
                    ax.set_title(f"{col} Forecast")
                    ax.legend()
                    ax.set_xlabel(time_col)
                    ax.set_ylabel(target_col)
                    st.pyplot(fig, use_container_width=True)
        
                            
if st.sidebar.button("Reset All"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()

