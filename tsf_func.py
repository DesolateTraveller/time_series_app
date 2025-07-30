#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st

#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#----------------------------------------
import altair as alt
import plotly.express as px
import plotly.offline as pyoff
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
#----------------------------------------
#from ml_insample import classification_analysis
from sklearn.impute import SimpleImputer, KNNImputer
#----------------------------------------
# # Forecast
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from datetime import datetime, timedelta
#from fbprophet import Prophet
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
#from keras.models import Sequential
#from keras.layers import LSTM, GRU, Dense
#from keras.preprocessing.sequence import TimeseriesGenerator
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
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    return {
            "Test Statistic": result[0],
            "p-value": result[1],
            "Lags Used": result[2],
            "Number of Observations": result[3],
            "Critical Values": result[4],
            "Stationary": result[1] <= 0.05  # p-value <= 0.05 indicates stationarity
    }

@st.cache_data(ttl="2h")
def kpss_test(series):
    result, p_value, lags, critical_values = kpss(series, regression='c')
    return {
            "Test Statistic": result,
            "p-value": p_value,
            "Critical Values": critical_values,
            "Stationary": p_value > 0.05  # p-value > 0.05 indicates stationarity
    }

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    #rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred)) if np.all(y_pred > 0) else None
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    #return mae, mse, rmse, r2, rmsle, mape_value
    return mae, mse, rmse, r2, mape

@st.cache_data(ttl="2h")
def test_stationarity(series):
    adf_stat, adf_pval, *_ = adfuller(series.dropna())
    kpss_stat, kpss_pval, *_ = kpss(series.dropna(), nlags="auto")
    return adf_pval, kpss_pval

@st.cache_data(ttl="2h")
def first_spike(values, threshold=0.2):
    for i in range(1, len(values)):
        if abs(values[i]) > threshold:
            return i
    return 0

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
def check_missing_values(data):
    missing_values = data.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    return missing_values 

@st.cache_data(ttl="2h")
def check_outliers(data):
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    outliers = pd.DataFrame(columns=['Column', 'Number of Outliers'])
    for column in numerical_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        threshold = 1.5
        outliers_indices = ((data[column] < Q1 - threshold * IQR) | (data[column] > Q3 + threshold * IQR))
        num_outliers = outliers_indices.sum()
        outliers = outliers._append({'Column': column, 'Number of Outliers': num_outliers}, ignore_index=True)
        return outliers
    
def invert_transforms(original_series, forecast_diff, log_transformed=True):
    forecast_log = pd.Series(forecast_diff).cumsum() + original_series.iloc[-1]
    if log_transformed:
        forecast = np.exp(forecast_log)
    else:
        forecast = forecast_log
    return forecast