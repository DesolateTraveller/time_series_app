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
from datetime import datetime, timedelta
#----------------------------------------
from sklearn.model_selection import train_test_split
#----------------------------------------
# Forecast
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
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
#from fbprophet import Prophet
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense, Dropout
#from prophet import Prophet
#from keras.models import Sequential
#from keras.layers import LSTM, GRU, Dense
#from keras.preprocessing.sequence import TimeseriesGenerator
#----------------------------------------
from tsf_func import load_file, adf_test, kpss_test, test_stationarity, first_spike
from tsf_func import calculate_metrics
from tsf_func import check_missing_values, check_outliers, handle_categorical_missing_values, handle_numerical_missing_values
from tsf_func import invert_transforms
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


#---------------------------------------------------------------------------------------------------------------------------------
### Main app
#---------------------------------------------------------------------------------------------------------------------------------

#Sst.divider()

col1, col2 = st.columns((0.15,0.85))
with col1:           
    with st.container(border=True):

        file = st.file_uploader("**:blue[Choose a file]**",type=["csv", "xls", "xlsx"], accept_multiple_files=False,key="file_upload")
        if file is not None:
                st.success("Data loaded successfully!")
                df = load_file(file)        #for filter
                
                st.divider()
                
                target_variable = st.selectbox("**:blue[Target Variable]**", options=["None"] + list(df.columns), key="target_variable")
                time_col = st.selectbox("**:blue[Time Frame Column]**", options=["None"] + list(df.columns), key="time_col")
                forecast_periods = st.slider('**:blue[Forecasting periods]**', min_value=30, max_value=90, value=60, key='for_ped') 
        
                st.divider()
                
                add_lags = st.checkbox("**:blue[Add Lagged Features?]**", value=False)
                if add_lags:
                    num_lags = st.number_input("**:blue[Number of Lags]**", min_value=1, max_value=30, value=3)
                    
                if time_col == "None" or target_variable == "None" :
                    st.warning("Please choose **target variable**, **time-frame column** to proceed with the analysis.")
        
                else:
                    st.warning("Tune or Change the **Hyperparameters**(tab shown in the top) whenever required.")   
                    with col2:
                                            
                        with st.popover("**:blue[:hammer_and_wrench: Hyperparameters]**",disabled=False, use_container_width=True):  

                            subcol1, subcol2, subcol3, subcol4, subcol5 = st.columns(5)
                            with subcol1:                            
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
                                order_arima = st.text_input('**ARIMA Order (p,d,q)**:', '1,1,1')
                                order_arima = tuple(map(int, order_arima.split(',')))     
                                order_sarima = st.text_input('**SARIMA Order (p,d,q,m)**:', '1,1,1,12')
                                order_sarima = tuple(map(int, order_sarima.split(',')))                                                                                             

                            with subcol5:
                                selected_metric = st.selectbox("🔍 Select Error Metric to Find Best Model", ["MAE", "MSE", "RMSE", "MAPE"])
                                st.divider() 
                                model = st.selectbox("**Select Model**", ["l2", "l1", "rbf"], index=0)
                                algo_name = st.selectbox("**Select Algorithm**", ["Binary Segmentation", "Pelt", "Window", "Bottom-Up"], index=0)
                                num_change_points = st.slider("**Number of Change Points**", min_value=1, max_value=20, value=3)                    

                        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        tabs = st.tabs(["**📊 Overview**","**📈 Visualizations**","**🔧 Preprocessing**","**✅ Checks**","**⚖️ Comparison**","**📈 Graph**","**📋 Results**","**🎲 Forecast**", "**⚠︎ Drift**"])
                        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            
                        with tabs[0]:
                            
                            unwanted_substrings = ['unnamed', '0', 'nan', 'deleted']
                            cols_to_delete = [col for col in df.columns if any(sub in col.lower() for sub in unwanted_substrings)]
                            if cols_to_delete:
                                st.warning(f"🗑️ {len(cols_to_delete)} column(s) deleted. | **Showing Top 3 rows for reference.**")
                            else:
                                st.info("✅ No unwanted columns found. Nothing deleted after importing. | **Showing Top 3 rows for reference.**")
                            df= df.drop(columns=cols_to_delete)
            
                            df[time_col] = pd.to_datetime(df[time_col], errors='coerce') 
                            if add_lags:
                                for lag in range(1, num_lags + 1):
                                    df[f'{target_variable}_lag_{lag}'] = df[target_variable].shift(lag) 
                            
                            st.dataframe(df.head(3))     
                              
                            df.dropna(subset=[time_col], inplace=True) 
                            df.sort_values(by=time_col, inplace=True)
                            df.reset_index(drop=True, inplace=True) 
                    
                            df1 = df.copy()             #for analysis
                            df2 = df.copy()             #for visualization
                            
                            df1.set_index(time_col, inplace=True)
                            df2.set_index(time_col, inplace=True)
                                                        
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
                                    ax.plot(df[time_col], df[target_variable], label='Sales', color='blue')
                                    ax.set_title(f"{target_variable} over {time_col}", fontsize=10)
                                    ax.set_xlabel(time_col)
                                    ax.set_ylabel(target_variable)
                                    ax.legend()
                                    ax.grid(True)
                                    plt.tight_layout()
                                    st.pyplot(fig, use_container_width=True)                          

                            with col2:                                
                                with st.container(border=True):
                                    
                                    decomposition = seasonal_decompose(df2[target_variable], model='additive', period=freq_guess)                                
    
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
                            
                            cat_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
                            num_vars = df.select_dtypes(include=['number']).columns.tolist()
                            
                            with st.container(border=True):
                                
                                if cat_vars:
                                    st.success(f"**📋 Categorical Variables Found: {len(cat_vars)}**")
                                else:
                                    st.warning("**⚠️ No categorical variables found.**")

                                if cat_vars:
                                    for i in range(0, len(cat_vars), 3):
                                        cols = st.columns(3)
                                        for j, col_name in enumerate(cat_vars[i:i+3]):
                                            with cols[j]:
                                                fig, ax = plt.subplots(figsize=(4, 3))
                                                df[col_name].value_counts().plot(kind='bar', ax=ax, color='skyblue')
                                                ax.set_title(f"{col_name}", fontsize=10)
                                                ax.set_ylabel("Count", fontsize=9)
                                                ax.set_xlabel("")
                                                ax.tick_params(axis='x', rotation=45, labelsize=8)
                                                ax.tick_params(axis='y', labelsize=8)
                                                st.pyplot(fig,use_container_width=True)

                            with st.container(border=True):

                                if num_vars:
                                    st.success(f"**📈 Numerical Variables Found: {len(num_vars)}**")
                                else:
                                    st.warning("**⚠️ No numerical variables found.**")

                                if num_vars:
                                    for i in range(0, len(num_vars), 2):  # Adjust columns as needed
                                        cols = st.columns(2)
                                        for j, col_name in enumerate(num_vars[i:i+2]):
                                            with cols[j]:
                                                st.markdown(f"**{col_name} over {time_col}**")
                                                skew_val = df[col_name].skew()
                                                skew_tag = (
                                                    "🟩 Symmetric" if abs(skew_val) < 0.5 else
                                                    "🟧 Moderate skew" if abs(skew_val) < 1 else
                                                    "🟥 Highly skewed"
                                                )
                                                st.info(f"Skewness: {skew_val:.2f} — {skew_tag}")

                                                fig_line, ax_line = plt.subplots(figsize=(7,2.5))
                                                ax_line.plot(df[time_col], df[col_name], color='teal', linewidth=1.5)
                                                ax_line.set_title(f"{col_name} over {time_col}", fontsize=10)
                                                ax_line.set_xlabel(time_col, fontsize=8)
                                                ax_line.set_ylabel(col_name, fontsize=8)
                                                ax_line.tick_params(labelsize=8)
                                                ax_line.grid(True)
                                                st.pyplot(fig_line, use_container_width=True)
                                                
                                        st.markdown('---')

                            with st.container(border=True):

                                st.markdown("##### 🧪 Decomposition ")
                                #decomposition = seasonal_decompose(df1[target_variable], model='additive', period=freq_guess)

                                fig, axs = plt.subplots(4, 1, figsize=(20,10), sharex=True)
                                axs[0].plot(decomposition.observed, label="Observed")
                                axs[1].plot(decomposition.trend, label="Trend", color='orange')
                                axs[2].plot(decomposition.seasonal, label="Seasonality", color='green')
                                axs[3].plot(decomposition.resid, label="Residuals", color='red')

                                for ax in axs:
                                    ax.legend(loc='upper left')
                                    ax.grid(True)

                                st.pyplot(fig, use_container_width=True)

                            with st.container(border=True):
                                
                                st.markdown("##### 🎢 Rolling Statistics")
                                df2.sort_index(inplace=True)
                                rolling_mean = df2[target_variable].rolling(window=12).mean()
                                rolling_std = df2[target_variable].rolling(window=12).std()
                                
                                fig3, ax2 = plt.subplots(figsize=(20,3))
                                ax2.plot(df2[target_variable], label='Original', color='blue')
                                ax2.plot(rolling_mean, label='Rolling Mean', color='orange')
                                ax2.plot(rolling_std, label='Rolling Std Dev', color='green')
                                ax2.set_title(f"Rolling Statistics of {target_variable} over Time", fontsize=10)
                                ax2.set_xlabel(time_col)
                                ax2.set_ylabel(target_variable)
                                ax2.legend()
                                ax2.grid(True)
                                st.pyplot(fig3, use_container_width=True)

                        with tabs[2]:  
                                                  
                            with st.container(border=True):
                                
                                col1, col2 = st.columns((0.2,0.8))
                                with col1:
                        
                                    missing_values = check_missing_values(df)
                                    if missing_values.empty:
                                        st.success("**No missing values found!**")
                                    else:
                                        st.warning("**Missing values found!**")
                                        st.write("**Number of missing values:**")
                                        st.table(missing_values)

                                        with col2:   
                                                          
                                            st.write("**Missing Values Treatment:**")                  
                                            cleaned_df = handle_numerical_missing_values(df, selected_numerical_strategy)
                                            cleaned_df = handle_categorical_missing_values(cleaned_df, selected_categorical_strategy)   
                                            st.table(cleaned_df.head(2))
                                            st.download_button("📥 Download Treated Data (.csv)", cleaned_df.to_csv(index=False), file_name="treated_data.csv")                            

                            with st.container(border=True):
                                
                                df.set_index(time_col, inplace=True)
                                time_diffs = df.index.to_series().diff().dropna()
                                most_common_diff = time_diffs.mode()[0]
                                is_equidistant = (time_diffs == most_common_diff).all()

                                st.markdown("##### ⏱️ Timestamp Check")
                                if is_equidistant:
                                    st.success(f"✅ Timestamps are in **chronological and equidistant** order. Interval: `{most_common_diff}`")
                                else:
                                    st.warning(f"⚠️ Timestamps are **not equidistant**. Detected interval mode: `{most_common_diff}`")

                                    if st.toggle("**🔧 Fix Missing Timestamps (Reindex & Interpolate)?**"):

                                        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=most_common_diff)
                                        df = df.reindex(full_index)
                                        num_cols = df.select_dtypes(include='number').columns
                                        df[num_cols] = df[num_cols].interpolate()
                                        st.success("✅ Reindexed and interpolated missing timestamps.")
                                        st.dataframe(df.head(3),use_container_width=True)

                            with st.container(border=True):   

                                    apply_resampling = st.toggle("🔁 **Enable Resampling?**", value=False)
                                    df_resampled = df.copy()

                                    if apply_resampling:
                                        resample_type = st.radio("Choose resampling type:", ["Downsampling", "Upsampling"], horizontal=True)

                                        if time_col not in df_resampled.columns:
                                            df_resampled = df_resampled.reset_index()
                                        df_resampled[time_col] = pd.to_datetime(df_resampled[time_col])
                                        df_resampled.set_index(time_col, inplace=True)

                                        if resample_type == "Downsampling":
                                            df_resampled = getattr(df_resampled.resample(resample_freq_down), agg_method)()
                                            st.info(f"🔽 Downsampled to `{resample_freq_down}` using `{agg_method}` aggregation.")

                                        elif resample_type == "Upsampling":
                                            df_resampled = df_resampled.resample(resample_freq_up).asfreq()
                                            num_cols = df_resampled.select_dtypes(include='number').columns
                                            df_resampled[num_cols] = df_resampled[num_cols].interpolate(method=interp_method, limit_direction="both")
                                            st.info(f"🔼 Upsampled to `{resample_freq_up}` using `{interp_method}` interpolation.")

                                        st.dataframe(df_resampled.head(), use_container_width=True)

                                    else:
                                        st.info("⏸️ Resampling disabled. using original data.")

                                    df = df_resampled.copy()                                                                 
                            
                            with st.container(border=True):
                                
                                col1, col2= st.columns((0.25,0.75))
                                with col1:                            
                                    apply_log = st.checkbox("**Apply log transformation before differencing (if data > 0)**", value=False)
                                
                                with col2:                            
                                    series = df[target_variable].dropna()
                                    
                                    if apply_log:
                                        if (series <= 0).any():
                                            st.warning("**⚠️ Cannot apply log transformation — series contains non-positive values.**")
                                            apply_log = False
                                        else:
                                            series = np.log(series)
                                            st.info("**🧮 Log transformation applied before differencing.**")
                                                                                        
                        with tabs[3]: 
                                                   
                            with st.container(border=True):
                                
                                col1, col2= st.columns((0.25,0.75))
                                with col1:
                                
                                    adf_result = adfuller(series)
                                    adf_stat, adf_pval = adf_result[0], adf_result[1]

                                    kpss_result = kpss(series, regression='c', nlags="auto")
                                    kpss_stat, kpss_pval = kpss_result[0], kpss_result[1]

                                    st.markdown("##### 🧪 Stationarity Tests ")
                                    stationarity_df = pd.DataFrame({
                                        "Test": ["ADF", "KPSS"],
                                        "Test Statistic": [adf_stat, kpss_stat],
                                        "p-value": [adf_pval, kpss_pval],
                                        "Interpretation": [
                                            "Stationary (p ≤ 0.05)" if adf_pval <= 0.05 else "Non-stationary (p > 0.05)",
                                            "Non-stationary (p ≤ 0.05)" if kpss_pval <= 0.05 else "Stationary (p > 0.05)"]})
                                    stationarity_df[["Test Statistic", "p-value"]] = stationarity_df[["Test Statistic", "p-value"]].round(3)
                                    st.table(stationarity_df.T)

                                    st.divider()
                                    #----------------------------------------------------------
                                    if adf_pval > 0.05 or kpss_pval <= 0.05:
                                        st.warning("❗Series is **non-stationary** — differencing required.")
                                    else:
                                        st.success("✅ Series is **stationary** — no differencing required.")
                                    #----------------------------------------------------------                                    
                                    needs_diff = adf_pval > 0.05 or kpss_pval <= 0.05
                                    
                                    diff_series = series.copy()
                                    diff_count = 0
                                    seasonal_diff_applied = False
                                    max_diff = 3 
                                    
                                    decomposition = seasonal_decompose(series, model='additive', period=12)
                                    seasonality_std = decomposition.seasonal.dropna().std()
                                    has_seasonality = seasonality_std > 0.2

                                    if has_seasonality:
                                        diff_series = diff_series.diff(12)
                                        seasonal_diff_applied = True
                                        st.warning("🔁 Detected strong seasonality — **seasonal differencing** applied.")

                                    adf_pval, kpss_pval = test_stationarity(diff_series)
                                    while needs_diff and diff_count < max_diff:
                                        diff_series = diff_series.diff()
                                        diff_count += 1
                                        adf_pval, kpss_pval = test_stationarity(diff_series)

                                    if diff_count == 0 and not seasonal_diff_applied:
                                        st.success("✅ Series is already **stationary**. No differencing needed.")
                                    elif seasonal_diff_applied and diff_count == 0:
                                        st.info("🔁 **Seasonal differencing** applied. Series became stationary.")
                                    else:
                                        st.warning(f"🔁 Applied **{diff_count}x differencing**{' with seasonal differencing' if seasonal_diff_applied else ''} to achieve stationarity.")

                                    df['diff'] = diff_series
                                    df.dropna(subset=['diff'], inplace=True)
                                    #----------------------------------------------------------
                                
                                with col2:
                                        
                                        st.markdown("##### 📈 Differenced Series | Visualization")
                                        fig, ax = plt.subplots(figsize=(20, 4))
                                        ax.plot(df.index, df['diff'], label='Differenced Series', color='purple')
                                        #ax.set_title('Differenced Series', fontsize=10)
                                        ax.set_xlabel('Date')
                                        ax.set_ylabel('Value')
                                        ax.legend()
                                        ax.grid(True)
                                        st.pyplot(fig, use_container_width=True)

                            with st.container(border=True):
                                                                    
                                st.markdown("##### 🔁 Differenced Series(**Showing Top 2 rows for reference.**)")
                                df_show = df.copy()
                                if time_col not in df_show.columns:
                                    df_show = df_show.reset_index()
                                df_show = df_show[[time_col, target_variable, 'diff']].rename(columns={target_variable: "Original", 'diff': "Differenced"})
                                st.dataframe(df_show.head(2), hide_index=True)

                            with st.container(border=True):

                                st.markdown("##### 📈 ACF & PCF Plot")
                                if len(df['diff']) < lags_val:
                                    st.warning(f"⚠️ Not enough data points to plot with {lags_val} lags. Minimum required: {lags_val}")
                                else:
                                    col1, col2 = st.columns(2)
                                    with col1:                                 

                                        fig_acf, ax_acf = plt.subplots(figsize=(12, 4))
                                        plot_acf(df['diff'], ax=ax_acf, lags=40, alpha=0.05)
                                        ax_acf.set_title("Autocorrelation Function (ACF)")
                                        st.pyplot(fig_acf, use_container_width=True)

                                    with col2:    
                                        
                                        fig_pacf, ax_pacf = plt.subplots(figsize=(12, 4))
                                        plot_pacf(df['diff'], ax=ax_pacf, lags=40, alpha=0.05, method="ywm")
                                        ax_pacf.set_title("Partial Autocorrelation Function (PACF)")
                                        st.pyplot(fig_pacf, use_container_width=True)

                                with st.expander("ℹ️ Interpretation Tips",expanded=True):
                                    st.markdown("""
                                    - **ACF** helps determine the **MA(q)** term: cut-off or gradual decline.
                                    - **PACF** helps determine the **AR(p)** term: sharp drop or tapering.
                                    - Use spikes outside the shaded area to identify lags.
                                    """)
  
                            with st.container(border=True):
                                
                                acf_vals = acf(df['diff'], nlags=40)
                                pacf_vals = pacf(df['diff'], nlags=40, method=pacf_method)

                                q_suggested = first_spike(acf_vals)
                                p_suggested = first_spike(pacf_vals)
                                d = diff_count            
                    
                        with tabs[4]: 

                            X = df.drop(columns=["diff"])
                            y = df["diff"]
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
                            #y_log = np.log(df["diff"])  # or original column
                            #y_log_train = y_log.iloc[y_train.index]

                            col1, col2= st.columns(2)
                            with col1:  
                                with st.container(border=True): 
                                          
                                    st.markdown("✅ **Final dataset prepared for modeling**")
                                    st.info(f"""
                                    - Train set size: **{X_train.shape[0]} rows**  
                                    - Test set size: **{X_test.shape[0]} rows**
                                    """)   

                            with col2: 
                                with st.container(border=True):
                                    
                                    st.markdown("🤖 **Suggested ARIMA Order (p, d, q)**")
                                    st.info(f"""
                                    - **p (AR)** suggested from PACF: `{p_suggested}`
                                    - **d (Differencing)** based on stationarity tests: `{d}`
                                    - **q (MA)** suggested from ACF: `{q_suggested}`
                                    """)  

                            st.divider() 
                            
                            col1, col2= st.columns(2)
                            with col1:                                                                          
                                with st.container(border=True):
                                                          
                                    results = []
                                    predictions = {}

                                    model_ets = ExponentialSmoothing(y_train, trend='add', seasonal=None, initialization_method='estimated').fit()
                                    ets_pred = model_ets.forecast(steps=len(y_test))
                                    metrics = calculate_metrics(y_test, ets_pred)
                                    results.append(["Exponential Smoothing", *metrics])
                                    predictions["Exponential Smoothing"] = ets_pred

                                    ma_pred = y_train.rolling(window=2).mean().shift(1).fillna(method='bfill')
                                    metrics = calculate_metrics(y_train, ma_pred)
                                    results.append(["Moving Average", *metrics])
                                    predictions["Moving Average"] = ma_pred.iloc[-len(y_test):]

                                    model_ar = AutoReg(y_train, lags=1).fit()
                                    ar_pred = model_ar.predict(start=len(y_train), end=len(y_train)+len(y_test)-1)
                                    metrics = calculate_metrics(y_test, ar_pred)
                                    results.append(["Auto Regression", *metrics])
                                    predictions["Auto Regression"] = ar_pred

                                    model_arima = ARIMA(y_train, order=(2, 1, 2)).fit()
                                    arima_pred = model_arima.forecast(steps=len(y_test))
                                    metrics = calculate_metrics(y_test, arima_pred)
                                    results.append(["ARIMA", *metrics])
                                    predictions["ARIMA"] = arima_pred

                                    #model_sarimax = SARIMAX(y_train, exog=X_train, order=(1, 1, 1)).fit(disp=False)
                                    #sarimax_pred = model_sarimax.predict(start=X_test.index[0], end=X_test.index[-1], exog=X_test)
                                    #metrics = calculate_metrics(y_test, sarimax_pred)
                                    #results.append(["SARIMAX", *metrics])
                                    #predictions["SARIMAX"] = sarimax_pred

                                    model_xgb = XGBRegressor()
                                    model_xgb.fit(X_train, y_train)
                                    xgb_pred = model_xgb.predict(X_test)
                                    metrics = calculate_metrics(y_test, xgb_pred)
                                    results.append(["XGBoost", *metrics])
                                    predictions["XGBoost"] = xgb_pred

                                    st.markdown("##### ⚖️ Comparison |  Error Metrics")
                                    metrics_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R2", "MAPE"]).round(3)
                                    st.dataframe(metrics_df,hide_index=True) 

                            with col2:                                                                          
                                with st.container(border=True):
                                                                 
                                    metric_directions = {"MAE": True,  "MSE": True,"RMSE": True,"MAPE": True,"R2": True}
                                    best_models_summary = []
                                    for metric, ascending in metric_directions.items():
                                        best_model = metrics_df.sort_values(by=metric, ascending=ascending).iloc[0]
                                        best_models_summary.append({"Metric": metric,"Best Model": best_model["Model"],"Score": best_model[metric]})
                                    best_models_df = pd.DataFrame(best_models_summary)
                                    st.markdown("##### 🏆 Best Models by Error Metric")
                                    st.dataframe(best_models_df, hide_index=True, use_container_width=True)

                            with st.container(border=True):     
                                
                                    ranked_df = metrics_df.copy()
                                    for metric in ["MAE", "MSE", "RMSE", "MAPE"]:
                                        ranked_df[f"{metric}_Rank"] = ranked_df[metric].rank(method='min', ascending=True)
                                    ranked_df["Rank"] = ranked_df["RMSE"].rank(method='min', ascending=False)
                                    rank_cols = [col for col in ranked_df.columns if col.endswith("_Rank")]
                                    ranked_df["Average_Rank"] = ranked_df[rank_cols].mean(axis=1)
                                    best_overall_model_row = ranked_df.sort_values(by="Average_Rank").iloc[0]
                                    best_model_name = best_overall_model_row["Model"]

                                    st.success(f"**🏅 Best Model: {best_model_name}**")
                                
                        with tabs[5]:

                            with st.container(border=True):

                                best_forecast = predictions[best_model_name]
                                plt.figure(figsize=(12, 4))
                                plt.plot(y_test.index, y_test.values, label='Actual', color='black')
                                plt.plot(y_test.index, best_forecast, label='Forecast', color='blue')
                                plt.title(f'{best_model_name} — Actual vs Forecast')
                                plt.xlabel("Time")
                                plt.ylabel("Value")
                                plt.legend()
                                st.pyplot(plt)

                                residuals = y_test - best_forecast
                                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                                sns.histplot(residuals, kde=True, ax=ax[0])
                                ax[0].set_title("Residual Distribution")
                                sns.scatterplot(x=y_test, y=residuals, ax=ax[1])
                                ax[1].set_title("Residuals vs Actual")
                                ax[1].axhline(0, linestyle='--', color='red')
                                st.pyplot(fig)
                                
                                                                                           
                        with tabs[8]:    
        
                                with st.container(border=True):
                                    
                                    st.divider()
                                
         
        
        
        
        
        
        
        
        else:
            st.warning("Please upload a file for analysis.") 
                 
