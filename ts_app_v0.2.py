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
#----------------------------------------
import itertools
#----------------------------------------
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as pyoff
import altair as alt
#----------------------------------------
import base64
import json
import random
import warnings
import itertools
from datetime import datetime
#----------------------------------------
import statsmodels.api as sm
from  statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller, kpss
#----------------------------------------
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric, add_changepoints_to_plot, plot_plotly, plot_components_plotly
from prophet.serialize import model_to_json, model_from_json
#----------------------------------------
from skimpy import skim
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
#----------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
#image = Image.open('Image_Clariant.png')
st.set_page_config(page_title="Forecasting App",
                   #page_icon=
                   layout="wide",
                   initial_sidebar_state="auto",)
#st.sidebar.image(image, use_column_width='auto') 
#----------------------------------------
st.title(f""":rainbow[Forecasting App | v0.1]""")
st.markdown('Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a>', 
            unsafe_allow_html=True)
st.info('**Disclaimer : :blue[Thank you for visiting the app] | Unauthorized uses or copying of the app is strictly prohibited | Click the :blue[sidebar] to follow the instructions to start the applications.**', icon="ℹ️")
#----------------------------------------
# Set the background image
#----------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

class SessionState:
  def __init__(self):
    self.clear_cache = False
state = SessionState()

@st.cache_data(ttl="2h")
def load_file(file):
    file_extension = file.name.split('.')[-1]
    if file_extension == 'csv':
        df = pd.read_csv(file, sep=None, engine='python', encoding='utf-8', parse_dates=True, infer_datetime_format=True)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format")
        df = pd.DataFrame()
    return df

@st.cache_data(ttl="2h")
def auto_detect_columns(df):
    date_col = None
    metric_col = None
    
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_timedelta64_dtype(df[col]):
            date_col = col
            break
        elif col.lower().find('date') != -1:
            date_col = col
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            break

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            metric_col = col
            break
    
    return date_col, metric_col

@st.cache_data(ttl="2h")
def prep_data(df, date_col, metric_col):
    df = df.rename({date_col: "ds", metric_col: "y"}, errors='raise', axis=1)
    st.success("The selected date column is now labeled as **ds** and the Target column as **y**")
    df = df[['ds', 'y']].sort_values(by='ds', ascending=True)
    return df

@st.cache_data(ttl="2h")
def plot_rolling_statistics(df, window=12):
    df.set_index('ds', inplace=True)
    rolling_mean = df['y'].rolling(window=window).mean()
    rolling_std = df['y'].rolling(window=window).std()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['y'], color='blue', label='Original')
    ax.plot(rolling_mean, color='red', label='Rolling Mean')
    ax.plot(rolling_std, color='black', label='Rolling Std')
    ax.legend(loc='best')
    ax.set_title('Rolling Mean & Standard Deviation')
    plt.tight_layout()
    st.pyplot(fig)
    df.reset_index(inplace=True)

@st.cache_data(ttl="2h")
def decompose_series(df, model='additive', period=30):
    df.set_index('ds', inplace=True)
    decomposition = seasonal_decompose(df['y'], model=model, period=period)
    df.reset_index(inplace=True)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    for ax in axes:
        ax.legend(loc='best')
    plt.xlabel('Date')
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data(ttl="2h")
def test_stationarity(df):
    df.set_index('ds', inplace=True)
    adf_result = adfuller(df['y'].dropna())
    kpss_result = kpss(df['y'].dropna(), regression='c')
    df.reset_index(inplace=True)

    adf_output = pd.DataFrame({
        'Test Statistic': [adf_result[0]],
        'p-value': [adf_result[1]],
        'Critical Values': [', '.join([f'{k}: {v:.4f}' for k, v in adf_result[4].items()])]
    })

    st.write('**Results of Augmented Dickey-Fuller Test:**')
    st.table(adf_output)

    kpss_output = pd.DataFrame({
        'Test Statistic': [kpss_result[0]],
        'p-value': [kpss_result[1]],
        'Critical Values': [', '.join([f'{k}: {v:.4f}' for k, v in kpss_result[3].items()])]
    })

    st.write('**Results of KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Test:**')
    st.table(kpss_output)

    sns.set(style="darkgrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(df['ds'], df['y'], label='Original')
    axes[0].set_title('Time Series')
    plot_acf(df['y'], ax=axes[1])
    plot_pacf(df['y'], ax=axes[2])
    plt.xlabel('Lags')
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data(ttl="2h")
def make_stationary(df, diff_order):
    df['y_diff'] = df['y'].diff(periods=diff_order).dropna()
    return df[['ds', 'y_diff']].dropna()

@st.cache_data(ttl="2h")
def plot_data(df):
    line_chart = alt.Chart(df).mark_line().encode(x='ds:T', y='y:Q', tooltip=['ds:T', 'y']).interactive()
    st.altair_chart(line_chart, use_container_width=True)

@st.cache_data(ttl="2h")
def handle_numerical_missing_values(data, numerical_strategy):
    imputer = SimpleImputer(strategy=numerical_strategy)
    numerical_features = data.select_dtypes(include=['number']).columns
    data[numerical_features] = imputer.fit_transform(data[numerical_features])
    return data

@st.cache_data(ttl="2h")
def handle_categorical_missing_values(data, categorical_strategy):
    imputer = SimpleImputer(strategy=categorical_strategy, fill_value='no_info')
    categorical_features = data.select_dtypes(exclude=['number']).columns
    data[categorical_features] = imputer.fit_transform(data[categorical_features])
    return data

@st.cache_data(ttl="2h")
def create_lstm_model(train_data, n_features, epochs=100, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(train_data.shape[1], n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------

#st.sidebar.header("Input", divider='blue')
#st.sidebar.info('Please choose from the following options to start the application.', icon="ℹ️")
#data_source = st.sidebar.radio("**:blue[Select the main source]**", ["File Upload", "AWS S3", "Sharepoint"],)

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

stats_expander = st.expander("**:blue[:pushpin: Knowledge Database]**", expanded=False)
with stats_expander: 

                    st.markdown("""
                    <style>
                    .info-container {
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-left: 6px solid #3498db;
                    border-radius: 5px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    margin-bottom: 20px;
                    }
                    .info-container h3 {
                    color: #3498db;
                    font-weight: bold;
                    margin-bottom: 10px;
                    }
                    .info-container p {
                    color: #333;
                    margin: 5px 0;
                    }
                    .info-container ul {
                    list-style-type: none;
                    padding: 0;
                    }
                    .info-container li {
                    margin: 10px 0;
                    display: flex;
                    align-items: center;
                    }
                    .info-container li:before {
                    content: "⭐";
                    margin-right: 10px;
                    color: #3498db;
                    font-size: 1.2em;
                    }
                    </style>

                    <div class="info-container">
                        <h3>🛠️ Definitions</h3>
                        <p>Below are the important words and their definitions used in the app.</p>
                        <ul>
                            <li><strong>📉 Rolling Mean & Standard Deviation</strong> - Analyze the time series' moving average and volatility over a specified window to understand trends and variability.</li>
                            <li><strong>📊 Decomposition</strong> - Break down the time series into trend, seasonality, and residual components to gain insights into its structure.</li>
                            <li><strong>🔍 ADF Test (Augmented Dickey-Fuller Test)</strong> - A statistical test used to check if a time series is stationary. The null hypothesis is that the series is non-stationary.</li>
                            <li><strong>🔬 KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin Test)</strong> - Another test for stationarity. The null hypothesis is that the series is stationary, complementing the ADF test.</li>
                            <li><strong>🔄 Differencing</strong> - A technique to transform a non-stationary series into a stationary one by subtracting the previous observation from the current observation.</li>    
                            <li><strong>📆 Seasonality Mode</strong> - Determines whether the seasonality component in the model is additive or multiplicative.</li>
                            <li><strong>📈 Changepoint Prior Scale</strong> - A hyperparameter that controls the flexibility of the trend in the Prophet model, affecting how much the trend can change at each changepoint.</li>
                            <li><strong>🌊 Seasonality Prior Scale</strong> - A hyperparameter that controls the flexibility of the seasonality in the Prophet model, impacting the seasonal component's amplitude.</li>    
                            <li><strong>🔄 Cross-validation</strong> - A method to evaluate the model’s performance by dividing the dataset into multiple training and testing parts, ensuring the model's robustness.</li>
                            <li><strong>📉 MAPE (Mean Absolute Percentage Error)</strong> - A metric used to measure the accuracy of forecasted values as a percentage, providing an intuitive sense of forecast accuracy.</li>
                            <li><strong>🎉 Holidays</strong> - Incorporating holidays can improve the forecast by accounting for special events that impact the time series data.</li>
                            <li><strong>📈 Growth Models</strong> - Defines whether the time series follows a linear or logistic growth pattern. Logistic growth is used for series expected to plateau at a certain level.</li> 
                            <li><strong>y(t) = g(t) + h(t) + s(t) + e(t)</strong> - The overall model, where:
                                <ul>
                                    <li><strong>y(t)</strong> = regressive model (the overall forecasted time series)</li>
                                    <li><strong>g(t)</strong> = trend component (long-term progression of the series)</li>
                                    <li><strong>h(t)</strong> = holiday components (effects of holidays)</li>
                                    <li><strong>s(t)</strong> = seasonality components (regular patterns repeated over time)</li>
                                    <li><strong>e(t)</strong> = error (random variability not captured by the model)</li>
                                </ul>
                            </li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

st.divider()

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

#if data_source == "File Upload":
  file = st.sidebar.file_uploader("**:blue[Choose a file]**",
                                    type=["csv", "xls", "xlsx"], 
                                    accept_multiple_files=False, 
                                    key="file_upload")
  if file:
        df = load_file(file)
        st.sidebar.divider()

#----------------------------------------

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:

            with st.expander("**Horizon**"):         

                periods_input = st.number_input('future periods (days) to forecast.', min_value=1, max_value=366, value=90)

                initial = st.number_input(value=365,label="initial",min_value=30,max_value=1096)
                initial = str(initial) + " days"

                period = st.number_input(value=90,label="period",min_value=1,max_value=365)
                period = str(period) + " days"

                horizon = st.number_input(value=90,label="horizon",min_value=30,max_value=366)
                horizon = str(horizon) + " days"
        #----------------------------------------
        with col2:
            with st.expander("**Trend**"):     
                daily = st.checkbox("Daily")
                weekly = st.checkbox("Weekly")
                monthly = st.checkbox("Monthly")
                yearly = st.checkbox("Yearly")
        #----------------------------------------
        with col3:
            with st.expander("**Seasonality**"):    
                seasonality = st.radio(label='Seasonality', options=['additive', 'multiplicative'])
        #----------------------------------------
        with col4:
            with st.expander("**Growth**"):
                growth = st.radio(label='Growth model', options=['linear', "logistic"]) 
                if growth == 'logistic':
                    st.info('Configure saturation')
                    cap = st.slider('Cap', min_value=0.0, max_value=1.0, step=0.05, value=0.5)
                    floor = st.slider('Floor', min_value=0.0, max_value=1.0, step=0.05, value=0.0)
                    if floor >= cap:
                        st.error('Invalid settings. Cap must be higher than floor.')
                        growth_settings = {}
                    else:
                        growth_settings = {'cap': cap, 'floor': floor}
                        df['cap'] = cap
                        df['floor'] = floor
                else:
                    growth_settings = {'cap': 1, 'floor': 0}
                    df['cap'] = 1
                    df['floor'] = 0
        #----------------------------------------
        with col5:
            with st.expander('**Holidays**'):    
                countries = ['United States', 'India', 'United Kingdom', 'France', 'Germany']
                selected_country = st.selectbox(label="Select country", options=countries, index=0)
                if selected_country:
                    country_code = selected_country[:2]
                    holidays = st.checkbox(f'Add {selected_country} holidays to the model')
        #----------------------------------------
        with col6:
            with st.expander('**Hyperparameters**'):       
                changepoint_scale = st.select_slider(label='Changepoint prior scale', options=[0.001, 0.01, 0.1, 0.5], value=0.1)
                seasonality_scale = st.select_slider(label='Seasonality prior scale', options=[0.01, 0.1, 1.0, 10.0], value=1.0)
#--------------------------------------------------------------------------------------------------------------------------------
        stats_expander = st.expander("**Preview of Data**", expanded=False)
        with stats_expander:  
            st.table(df.head(2))
        st.divider()
#--------------------------------------------------------------------------------------------------------------------------------

        if not df.empty:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["**Information**", 
                                                    "**Visualization**",
                                                    "**Forecast**",
                                                    "**Validation**",
                                                    "**Tuning**", 
                                                    "**Result**",
                                                    ])
            
            #---------------------------------------------------------------------------------------------------------------------
            with tab1:

                date_col, metric_col = auto_detect_columns(df)

                columns = list(df.columns)
                col1, col2 = st.columns(2)
                with col1:
                    date_col = st.selectbox("**Select date column**", options=columns, key="date", 
                                            index=df.columns.get_loc(date_col) if date_col else 0,help='Column to be parsed as a date')
                with col2:
                    metric_col = st.selectbox("**Select target column**", options=columns, key="values", 
                                              index=df.columns.get_loc(metric_col) if metric_col else 1,help='Quantity to be forecasted')
                
                df = prep_data(df, date_col, metric_col)
                st.divider()
                #----------------------------------------
                try:
                    line_chart = alt.Chart(df).mark_line().encode(x='ds:T', y="y:Q", tooltip=['ds:T', 'y']).properties(title="Input preview").interactive()
                    st.altair_chart(line_chart, use_container_width=True)
                except:
                    st.line_chart(df['y'], use_container_width=True, height=300)
                st.divider()
                #----------------------------------------
                st.write('**Descriptive Statistics**')
                st.write(df.describe().T, use_container_width=True)
                st.divider()
                #----------------------------------------
                @st.cache_data(ttl="2h")    
                def check_missing_values(data):
                    missing_values = df.isnull().sum()
                    missing_values = missing_values[missing_values > 0]
                    return missing_values 
                missing_values = check_missing_values(df)
                
                st.write('**Missing Value Treatment**')
                if missing_values.empty:
                    st.success("**No missing values found!**")
                    df1 = df.copy()
                
                else:
                    st.warning("Missing values found!")
                    st.caption("**Number of missing values for each column:**")
                    st.write(missing_values)

                    st.divider()
                    @st.cache_data(ttl="2h")
                    def handle_numerical_missing_values(data, numerical_strategy):
                        imputer = SimpleImputer(strategy=numerical_strategy)
                        numerical_features = data.select_dtypes(include=['number']).columns
                        data[numerical_features] = imputer.fit_transform(data[numerical_features])
                        return data 
                    col1, col2 = st.columns((0.3,0.7))
                    with col1:  
                        strategies = ['most_frequent', 'constant', 'median', 'mean']
                        selected_numerical_strategy = st.selectbox("**Select a strategy for treatment**", strategies, index=3) 
                    with col2:                           
                        cleaned_df = handle_numerical_missing_values(df, selected_numerical_strategy)
                        st.caption("**Treated input**")
                        st.table(cleaned_df.head(2))
                        df1 = cleaned_df.copy()                    

            #---------------------------------------------------------------------------------------------------------------------
            with tab2:                
                
                options_viz = st.radio('Options', ['Visualization (after treatment)','Rolling Mean & Standard Deviation','Decomposition', 'Stationarity',], 
                                       horizontal=True, label_visibility='collapsed', key='options_viz')
                st.divider()

                if options_viz == 'Visualization (after treatment)':
                    plot_option = st.selectbox("**Choose Plot**", ["Line Chart", "Histogram", "Scatter Plot"])
                
                #----------------------------------------
                    if plot_option == "Line Chart":
                        plot_data(df1)
                    #----------------------------------------
                    elif plot_option == "Histogram":
                        fig = px.histogram(df1, x='y', nbins=50)
                        st.plotly_chart(fig)
                    #----------------------------------------
                    elif plot_option == "Scatter Plot":
                        col1, col2 = st.columns((0.2,0.8))
                        with col1:
                            scatter_x = st.selectbox("X-axis", df1.columns, index=0)
                            scatter_y = st.selectbox("Y-axis", df1.columns, index=1)
                        with col2:                        
                            fig = px.scatter(df1, x=scatter_x, y=scatter_y, trendline="ols")
                            st.plotly_chart(fig)
                #----------------------------------------
                if options_viz == 'Rolling Mean & Standard Deviation':
                    window_size = st.number_input("**Window size for rolling statistics**", min_value=2, max_value=30, value=12)
                    plot_rolling_statistics(df1.copy(), window=window_size)
                    st.divider()
                #----------------------------------------
                if options_viz == 'Decomposition':
                    period = st.number_input("**Period for decomposition (days)**", min_value=2, max_value=365, value=30)
                    decompose_series(df1.copy(), model='additive', period=period)
                    st.divider()
                #----------------------------------------
                if options_viz == 'Stationarity':
                    st.write('**If result of Augmented Dickey-Fuller(ADF) Test > 0.05 and KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Test < 0.05,the series is non-stationary, we need to differencing the data to make it stationary**')
                    test_stationarity(df1.copy())
                    st.divider()
            #---------------------------------------------------------------------------------------------------------------------
                              
            #--------------------------------------------------------------------------------------------------------------------- 
            with tab3:

                with st.container():

                    st.write("Choose the below mentioned steps, Fit the model on the data and Generate future prediction.")
                    
                    #options_model = st.radio('Options', ['prophet','ARIMA'], 
                                       #horizontal=True, label_visibility='collapsed', key='options_model')
                    #st.divider()

                    #if options_model == 'prophet':
                    if st.checkbox("Initialize model (Fit)", key="fit"):
                            m = Prophet(seasonality_mode=seasonality,
                                #daily_seasonality=daily,
                                #weekly_seasonality=weekly,
                                #yearly_seasonality=yearly,
                                #growth=growth,
                                changepoint_prior_scale=changepoint_scale,
                                seasonality_prior_scale=seasonality_scale)
                    
                            if holidays and selected_country != 'Country name':
                                m.add_country_holidays(country_name=country_code)

                            if daily:
                                m.add_seasonality(name='daily', period=1, fourier_order=5)
                            if weekly:
                                m.add_seasonality(name='weekly', period=7, fourier_order=5)
                            if monthly:
                                m.add_seasonality(name='monthly', period=30, fourier_order=5)
                            if yearly:
                                m.add_seasonality(name='yearly', period=365, fourier_order=10)

                            with st.spinner('Fitting the model...'):
                                
                                #adf_result = adfuller(df['y'].dropna())
                                #kpss_result = kpss(df['y'].dropna(), regression='c')

                                #if adf_result[1] > 0.05 and kpss_result[1] < 0.05:
                                    #st.warning('The time series is non-stationary. Differencing the data...')
                                    #diff_order = st.slider('Differencing Order', min_value=1, max_value=5, value=1)
                                    #df = make_stationary(df, diff_order)
                                    #st.write(f"Data differenced to make it stationary with {len(df)} rows remaining.")

                                m.fit(df1)
                                future = m.make_future_dataframe(periods=periods_input, freq='D')
                                if growth == 'logistic':
                                    future['cap'] = cap
                                    future['floor'] = floor

                                st.caption(f"The model will produce forecast up to {future['ds'].max()}")
                                st.success('Model fitted successfully')
                            st.divider()

                    if st.checkbox("Generate forecast (Predict)", key="predict"):
                            try:
                                with st.spinner("Forecasting..."):
                                    forecast = m.predict(future)
                                    st.success('Prediction generated successfully')
                                    df_fut = st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], use_container_width=True)
                                    #st.table(df_fut)

                                    col1, col2 = st.columns(2)

                                    with col1:
                                        fig1 = m.plot(forecast)
                                        st.write('**Forecast plot**')
                                        st.pyplot(fig1)

                                    with col2:
                                        if growth == 'linear':
                                            fig2 = m.plot(forecast)
                                            add_changepoints_to_plot(fig2.gca(), m, forecast)
                                            st.write('**Growth plot**')
                                            st.pyplot(fig2,use_container_width=True)

                            except Exception as e:
                                st.warning(f"Error generating forecast: {e}")
                            st.divider()

                    if st.checkbox('Show components'):
                            try:
                                with st.spinner("Loading..."):
                                    fig3 = m.plot_components(forecast)
                                    st.pyplot(fig3,use_container_width=True)
                                    #plot_components_plotly(m, forecast)

                            except Exception as e:
                                st.warning(f"Error showing components: {e}")

                    #if options_model == 'ARIMA':

                        #arima_model = auto_arima(df1['y'], seasonal=True, m=12)
                        #arima_forecast = arima_model.predict(n_periods=365)
            #--------------------------------------------------------------------------------------------------------------------- 
            with tab4:

                with st.container():

                    st.write("In this section, you can calculate the metrics of the model")

                    #st.subheader("Metrices", divider='blue')
                    if st.checkbox('Calculate metrics'):
                        try:
                            with st.spinner("Cross validating..."):
                                            
                                col1, col2 = st.columns((0.3,0.7))
                                                    
                                with col1:
                                    df_cv = cross_validation(m, initial=initial,period=period,horizon=horizon,parallel="processes")                                         
                                    df_p = performance_metrics(df_cv,rolling_window=1)
                                    st.dataframe(df_p,height=100)

                                    st.info('''
                                    #### Model Evaluation Metrics
                                    | Metric  | Definition                                          |
                                    |---------|-----------------------------------------------------|
                                    | **MSE** | Mean Squared Error                                  |
                                    | **RMSE**| Root Mean Squared Error                             |
                                    | **MAE** | Mean Absolute Error                                 |
                                    | **MAPE**| Mean Absolute Percentage Error                      |
                                    | **MdAPE**| Median Absolute Percentage Error                    |
                                    | **sMAPE**| Symmetric Mean Absolute Percentage Error           |
                                    ''')

                                with col2:
                                    metrics = ['mae', 'mape', 'mse', 'rmse']
                                    selected_metric = st.selectbox("Select metric to plot", options=metrics)
                                    if selected_metric:
                                        fig4 = plot_cross_validation_metric(df_cv, metric=selected_metric)
                                        st.pyplot(fig4,use_container_width=True)

                        except Exception as e:
                            st.error(f"Error during cross-validation: {e}")

            #--------------------------------------------------------------------------------------------------------------------- 
            with tab5:

                with st.container():

                    st.write("In this section, you can perform cross-validation of the model")
                    #st.subheader("Hyperparameter Tuning", divider='blue')
                    st.write("Optimize hyperparameters to find the best combination. Due to iterative process it will take time.")

                    param_grid = {'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],}
                    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
                    mapes = []

                    if st.button("Optimize hyperparameters"):
                        with st.spinner("Finding best combination. Please wait.."):
                            try:
                                for params in all_params:
                                    m = Prophet(**params).fit(df)
                                    df_cv = cross_validation(m,initial=initial,period=period,horizon=horizon,parallel="processes")
                                    df_p = performance_metrics(df_cv, rolling_window=1)
                                    mapes.append(df_p['mape'].values[0])

                                tuning_results = pd.DataFrame(all_params)
                                tuning_results['mape'] = mapes
                                st.dataframe(tuning_results)

                                best_params = all_params[np.argmin(mapes)]
                                st.write('The best parameter combination is:')
                                st.write(best_params)

                            except Exception as e:
                                st.error(f"Error during hyperparameter optimization: {e}")

                    st.divider()
                    st.subheader("Out-sample validation", divider='blue')
                    col1, col2, col3 = st.columns((1,1,1))                      
                    with col1:
                        test_size = st.number_input('Test size (number of periods)', min_value=1, max_value=len(df1), value=30)

                    with col2:
                        train_df1 = df1[:-test_size]
                        test_df1 = df1[-test_size:]

                        model = Prophet(seasonality_mode=seasonality,
                                changepoint_prior_scale=changepoint_scale,
                                seasonality_prior_scale=seasonality_scale,
                                daily_seasonality=daily,
                                weekly_seasonality=weekly,
                                yearly_seasonality=yearly)
                        if holidays:
                            model.add_country_holidays(country_name=country_code)

                        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                        with st.spinner('Fitting the model on training data...'):
                            model.fit(train_df1)
                        with st.spinner('Predicting on test data...'):
                            future = model.make_future_dataframe(periods=test_size)
                            future['cap'] = growth_settings['cap']
                            future['floor'] = growth_settings['floor']
                            forecast = model.predict(future)

                        test_df1['yhat'] = forecast['yhat'].values[-test_size:]
                        test_df1['yhat_lower'] = forecast['yhat_lower'].values[-test_size:]
                        test_df1['yhat_upper'] = forecast['yhat_upper'].values[-test_size:]
                        st.write('**Actual vs Predicted Values**')
                        st.write(test_df1[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']].tail(),use_container_width=True)

                    with col3:

                        mae = mean_absolute_error(test_df1['y'], test_df1['yhat'])
                        mse = mean_squared_error(test_df1['y'], test_df1['yhat'])
                        rmse = np.sqrt(mse)
                        mape = np.mean(np.abs((test_df1['y'] - test_df1['yhat']) / test_df1['y'])) * 100

                        st.write("**Out-sample validation metrics**")
                        metrics_df = pd.DataFrame({
                                                "Metric": ["MAE", "MSE", "RMSE", "MAPE"],
                                                "Value": [mae, mse, rmse, f"{mape}%"]
                                                })
                        st.table(metrics_df)

                    st.write('**Forecast Plot**')
                    fig3 = plot_plotly(model, forecast)
                    st.plotly_chart(fig3, use_container_width=True)  
            #--------------------------------------------------------------------------------------------------------------------- 
            with tab6:

                st.write("Export your forecast results.")

                if 'forecast' in locals():
                        forecast_df = pd.DataFrame(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
                        forecast_data = forecast_df.to_csv(index=False)

                        st.dataframe(forecast_df, use_container_width=True)
                        st.divider()

                        st.caption('**Download Section**')
                        st.download_button(label="📥 Download Forecast CSV",data=forecast_data,file_name='Forecast_results.csv',mime='text/csv')

                        if st.button("Export model metrics (.csv)"):
                            try:
                                forecast_df = forecast_df.to_csv(decimal=',')
                                b64 = base64.b64encode(df_p.encode()).decode()
                                href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (click derecho > guardar como **metrics.csv**)'
                                st.markdown(href, unsafe_allow_html=True)
                            except:
                                st.write("No metrics to export")

                        if st.button('Save model configuration (.json) in memory'):
                            with open('serialized_model.json', 'w') as fout:
                                json.dump(model_to_json(m), fout)

                        if st.button('Clear cache memory'):
                            state.clear_cache = True
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
