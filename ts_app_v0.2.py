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
import random
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
#----------------------------------------
from skimpy import skim

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
st.markdown('Developed by : **:blue[E&PT - Digital Solutions]** | prepared by : <a href="mailto:avijit.chakraborty@clariant.com">Avijit Chakraborty</a>', 
            unsafe_allow_html=True)
st.info('**Disclaimer : :blue[Thank you for visiting the app] | This app is created for internal use, unauthorized uses or copying is strictly prohibited | Click the :blue[sidebar] to follow the instructions to start the applications.**', icon="ℹ️")
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

#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------

st.sidebar.header("Input", divider='blue')
st.sidebar.info('Please choose from the following options to start the application.', icon="ℹ️")
data_source = st.sidebar.radio("**:blue[Select the main source]**", ["File Upload", "AWS S3", "Sharepoint"],)

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
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

st.divider()

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

if data_source == "File Upload":
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

                periods_input = st.number_input('Select how many future periods (days) to forecast.', min_value=1, max_value=366, value=90)

                initial = st.number_input(value=365,label="initial",min_value=30,max_value=1096)
                initial = str(initial) + " days"

                period = st.number_input(value=90,label="period",min_value=1,max_value=365)
                period = str(period) + " days"

                horizon = st.number_input(value=90,label="horizon",min_value=30,max_value=366)
                horizon = str(horizon) + " days"

        with col2:
            with st.expander("**Trend**"):     
                daily = st.checkbox("Daily")
                weekly = st.checkbox("Weekly")
                monthly = st.checkbox("Monthly")
                yearly = st.checkbox("Yearly")

        with col3:
            with st.expander("**Seasonality**"):    
                seasonality = st.radio(label='Seasonality', options=['additive', 'multiplicative'])

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

        with col5:
            with st.expander('**Holidays**'):    
                countries = ['United States', 'India', 'United Kingdom', 'France', 'Germany']
                selected_country = st.selectbox(label="Select country", options=countries, index=0)
                if selected_country:
                    country_code = selected_country[:2]
                    holidays = st.checkbox(f'Add {selected_country} holidays to the model')

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
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Information**", 
                                                    "**Visualization**",
                                                    "**Forecast**",
                                                    "**Validation**", 
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

                try:
                    line_chart = alt.Chart(df).mark_line().encode(x='ds:T', y="y:Q", tooltip=['ds:T', 'y']).properties(title="Input preview").interactive()
                    st.altair_chart(line_chart, use_container_width=True)
                except:
                    st.line_chart(df['y'], use_container_width=True, height=300)

                st.write(df.describe().T, use_container_width=True)

            #---------------------------------------------------------------------------------------------------------------------
            with tab2:                
                
                options_viz = st.radio('Options', ['Visualization','Rolling Mean & Standard Deviation', 'Decomposition', 'Stationarity',], 
                                       horizontal=True, label_visibility='collapsed', key='options_viz')
                st.divider()

                if options_viz == 'Visualization':
                    plot_option = st.selectbox("**Choose Plot**", ["Line Chart", "Histogram", "Scatter Plot"])

                    if plot_option == "Line Chart":
                        plot_data(df)
                    #----------------------------------------
                    elif plot_option == "Histogram":
                        fig = px.histogram(df, x='y', nbins=50)
                        st.plotly_chart(fig)
                    #----------------------------------------
                    elif plot_option == "Scatter Plot":
                        col1, col2 = st.columns((0.2,0.8))
                        with col1:
                            scatter_x = st.selectbox("X-axis", df.columns, index=0)
                            scatter_y = st.selectbox("Y-axis", df.columns, index=1)
                        with col2:                        
                            fig = px.scatter(df, x=scatter_x, y=scatter_y, trendline="ols")
                            st.plotly_chart(fig)
                #----------------------------------------
                if options_viz == 'Rolling Mean & Standard Deviation':
                    window_size = st.number_input("Window size for rolling statistics", min_value=2, max_value=30, value=12)
                    plot_rolling_statistics(df.copy(), window=window_size)
                    st.divider()
                #----------------------------------------
                if options_viz == 'Decomposition':
                    period = st.number_input("Period for decomposition (days)", min_value=2, max_value=365, value=30)
                    decompose_series(df.copy(), model='additive', period=period)
                    st.divider()
                #----------------------------------------
                if options_viz == 'Stationarity':
                    st.write('**If adf_result[1] > 0.05 and kpss_result[1] < 0.05,the series is non-stationary, we need to differencing the data to make it stationary**')
                    test_stationarity(df.copy())
                    st.divider()

            #--------------------------------------------------------------------------------------------------------------------- 
            with tab3:

                with st.container():

                    st.write("Choose the below mentioned model-algorithm, Fit the model on the data and Generate future prediction.")

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
                                m.add_seasonality(name='yearly', period=365, fourier_order=5)

                            with st.spinner('Fitting the model...'):
                                
                                adf_result = adfuller(df['y'].dropna())
                                kpss_result = kpss(df['y'].dropna(), regression='c')

                                if adf_result[1] > 0.05 and kpss_result[1] < 0.05:
                                    st.warning('The time series is non-stationary. Differencing the data...')
                                    diff_order = st.slider('Differencing Order', min_value=1, max_value=5, value=1)
                                    df = make_stationary(df, diff_order)
                                    st.write(f"Data differenced to make it stationary with {len(df)} rows remaining.")

                                m.fit(df)
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
                                            st.pyplot(fig2)

                            except Exception as e:
                                st.warning(f"Error generating forecast: {e}")
                            st.divider()

                    if st.checkbox('Show components'):
                            try:
                                with st.spinner("Loading..."):
                                    fig3 = m.plot_components(forecast)
                                    st.pyplot(fig3)
                                    #plot_components_plotly(m, forecast)

                            except Exception as e:
                                st.warning(f"Error showing components: {e}")

            #--------------------------------------------------------------------------------------------------------------------- 
            with tab4:

                with st.container():

                    st.write("In this section, you can perform cross-validation of the model")

                    st.subheader("Metrices", divider='blue')
                    if st.checkbox('Calculate metrics'):
                        try:
                            with st.spinner("Cross validating..."):
                                            
                                col1, col2 = st.columns((0.3,0.7))
                                                    
                                with col1:
                                    df_cv = cross_validation(m, initial=initial,period=period,horizon=horizon,parallel="processes")                                         
                                    df_p = performance_metrics(df_cv,rolling_window=1)
                                    st.dataframe(df_p)

                                with col2:
                                    metrics = ['mae', 'mape', 'mse', 'rmse']
                                    selected_metric = st.selectbox("Select metric to plot", options=metrics)
                                    if selected_metric:
                                        fig4 = plot_cross_validation_metric(df_cv, metric=selected_metric)
                                        st.pyplot(fig4)

                        except Exception as e:
                            st.error(f"Error during cross-validation: {e}")

                    st.subheader("Hyperparameter Tuning", divider='blue')
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

            #--------------------------------------------------------------------------------------------------------------------- 
            with tab5:

                st.write("Export your forecast results.")

                if 'forecast' in locals():
                        forecast_df = pd.DataFrame(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
                        forecast_data = forecast_df.to_csv(index=False)

                        st.dataframe(forecast_df, use_container_width=True)
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

                        if st.button('Clear cache memory please'):
                            state.clear_cache = True
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

if data_source == "AWS S3":

    st.title(f""":rainbow[Configuring - Something new will be coming up.. ]""")

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

if data_source == "Sharepoint":

    st.title(f""":rainbow[Configuring - Something new will be coming up.. ]""")

