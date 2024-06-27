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

#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
#image = Image.open('Image_Clariant.png')
st.set_page_config(page_title="Forecasting App",
                   page_icon='https://www.clariant.com/images/clariant-logo-small.svg',
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
st.divider()

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------
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
    
    # Detecting date columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_timedelta64_dtype(df[col]):
            date_col = col
            break
        elif col.lower().find('date') != -1:
            date_col = col
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            break

    # Detecting metric columns
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

    plt.figure(figsize=(10, 6))
    plt.plot(df['y'], color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    st.pyplot(plt)
    df.reset_index(inplace=True)

@st.cache_data(ttl="2h")
def decompose_series(df, model='additive', period=30):
    df.set_index('ds', inplace=True)
    decomposition = seasonal_decompose(df['y'], model=model, period=period)
    df.reset_index(inplace=True)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    ax1.plot(decomposition.observed, label='Observed')
    ax1.legend(loc='best')
    ax2.plot(decomposition.trend, label='Trend')
    ax2.legend(loc='best')
    ax3.plot(decomposition.seasonal, label='Seasonal')
    ax3.legend(loc='best')
    ax4.plot(decomposition.resid, label='Residual')
    ax4.legend(loc='best')
    plt.xlabel('Date')
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data(ttl="2h")
def test_stationarity(df):
    df.set_index('ds', inplace=True)
    result = adfuller(df['y'].dropna())

    st.write('Results of Dickey-Fuller Test:')
    st.write(f'ADF Statistic: {result[0]:.4f}')
    st.write(f'p-value: {result[1]:.4f}')
    st.write('Critical Values:')
    for key, value in result[4].items():
        st.write(f'   {key}: {value:.4f}')
    
    df.reset_index(inplace=True)

    sns.set(style="darkgrid")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(df['ds'], df['y'], label='Original')
    axes[0].set_title('Time Series')
    plot_acf(df['y'], ax=axes[1])
    plot_pacf(df['y'], ax=axes[2])
    plt.xlabel('Lags')
    plt.tight_layout()
    st.pyplot(fig)
#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------

st.sidebar.header("Input", divider='blue')
st.sidebar.info('Please choose from the following options to start the application.', icon="ℹ️")
data_source = st.sidebar.radio("**:blue[Select the main source]**", ["File Upload", "AWS S3", "Sharepoint"],)

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

        st.divider()

#----------------------------------------

        if not df.empty:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Information**", "**Visualization**", "**Forecast**", "**Validation & Tuning**", "**Result**"])
            
            #----------------------------------------
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

                Options = st.radio('Options', ['Plot data', 'Show data', 'Show Statistics'], horizontal=True, label_visibility='collapsed', key='options')
                st.divider()

                if Options == 'Plot data':
                        try:
                            line_chart = alt.Chart(df).mark_line().encode(x='ds:T', y="y:Q", tooltip=['ds:T', 'y']).properties(title="Time series preview").interactive()
                            st.altair_chart(line_chart, use_container_width=True)
                        except:
                            st.line_chart(df['y'], use_container_width=True, height=300)

                if Options == 'Show data':
                        st.dataframe(df.head(), use_container_width=True)

                if Options == 'Show Statistics':
                        st.write(df.describe().T, use_container_width=True)

            #----------------------------------------
            with tab2:                
                
                Options_viz = st.radio('Options', ['Rolling Mean & Standard Deviation', 'Decomposition', 'Stationarity'], horizontal=True, label_visibility='collapsed', key='options_viz')

                if Options_viz == 'Rolling Mean & Standard Deviation':
                    window_size = st.number_input("Window size for rolling statistics", min_value=2, max_value=30, value=12)
                    plot_rolling_statistics(df.copy(), window=window_size)
                    st.divider()

                if Options_viz == 'Decomposition':
                    period = st.number_input("Period for decomposition (days)", min_value=2, max_value=365, value=30)
                    decompose_series(df.copy(), model='additive', period=period)
                    st.divider()

                if Options_viz == 'Stationarity':
                    test_stationarity(df.copy())
                    st.divider()

            #---------------------------------------- 
            with tab3:

                with st.container():

                    st.write("Fit the model on the data and generate future prediction.")
                    if st.checkbox("Initialize model (Fit)", key="fit"):
                        m = Prophet(seasonality_mode=seasonality,
                                daily_seasonality=daily,
                                weekly_seasonality=weekly,
                                yearly_seasonality=yearly,
                                growth=growth,
                                changepoint_prior_scale=changepoint_scale,
                                seasonality_prior_scale=seasonality_scale)
                    
                        if holidays and selected_country != 'Country name':
                            m.add_country_holidays(country_name=country_code)

                        if monthly:
                            m.add_seasonality(name='monthly', period=30, fourier_order=5)

                        with st.spinner('Fitting the model...'):
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

            #---------------------------------------- 
            with tab4:

                with st.container():

                    st.write("In this section, you can perform cross-validation of the model")

                    with st.expander("**Forecast Horizon**"):
                        horizon = st.number_input('Enter forecast horizon in days:', min_value=1, max_value=365, value=30)

                    st.subheader("Metrices", divider='blue')
                    if st.checkbox('Calculate metrics'):
                        try:
                            with st.spinner("Cross validating..."):
                                            
                                col1, col2 = st.columns(2)
                                                    
                                with col1:
                                    df_cv = cross_validation(m, horizon=f"{horizon} days", parallel=None)
                                    df_p = performance_metrics(df_cv)
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
                    st.write("Optimize hyperparameters to find the best combination.")

                    param_grid = {'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],}
                    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
                    mapes = []

                    if st.button("Optimize hyperparameters"):
                        try:
                            with st.spinner("Finding best combination..."):
                                for params in all_params:
                                    m = Prophet(**params).fit(df)
                                    df_cv = cross_validation(m, horizon=f"{horizon} days", parallel=None)
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

            #---------------------------------------- 
            with tab5:

                st.write("Export your forecast results.")
                if 'forecast' in locals():
                    forecast_df = pd.DataFrame(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
                    forecast_data = forecast_df.to_csv(index=False)

                    st.dataframe(forecast_df, use_container_width=True)
                    st.download_button(label="Download Forecast CSV",
                                       data=forecast_data,
                                       file_name='Forecast_results.csv',
                                       mime='text/csv')

