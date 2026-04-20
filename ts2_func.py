import streamlit as st
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Data Loading & Caching
# -----------------------------------------------------------------------------
@st.cache_data(ttl="2h")
def load_file(file):
    """Loads CSV or Excel files with robust error handling."""
    try:
        file_extension = file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(file, engine='python', encoding='utf-8')
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel.")
            return pd.DataFrame()
        
        # Basic cleanup
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 2. Data Preprocessing
# -----------------------------------------------------------------------------
@st.cache_data(ttl="2h")
def check_missing_values(data):
    missing_values = data.isnull().sum()
    return missing_values[missing_values > 0]

@st.cache_data(ttl="2h")
def handle_numerical_missing_values(data, strategy):
    data = data.copy()
    numerical_features = data.select_dtypes(include=['number']).columns
    
    if len(numerical_features) == 0:
        return data

    if strategy == 'forward fill (ffill)':
        data[numerical_features] = data[numerical_features].ffill()
    elif strategy == 'backward fill (bfill)':
        data[numerical_features] = data[numerical_features].bfill()
    elif strategy == 'interpolate':
        if isinstance(data.index, pd.DatetimeIndex):
            data[numerical_features] = data[numerical_features].interpolate(method='time')
        else:
            data[numerical_features] = data[numerical_features].interpolate(method='linear')
    else:
        # mean, median, most_frequent
        imputer = SimpleImputer(strategy=strategy)
        data[numerical_features] = imputer.fit_transform(data[numerical_features])
    
    return data

@st.cache_data(ttl="2h")
def handle_categorical_missing_values(data, strategy):
    data = data.copy()
    categorical_features = data.select_dtypes(exclude=['number']).columns
    if len(categorical_features) == 0:
        return data
        
    fill_value = 'Unknown' if strategy == 'constant' else data[categorical_features].mode().iloc[0]
    data[categorical_features] = data[categorical_features].fillna(fill_value)
    return data

# -----------------------------------------------------------------------------
# 3. Statistical Tests
# -----------------------------------------------------------------------------
@st.cache_data(ttl="2h")
def run_stationarity_tests(series):
    """Runs ADF and KPSS tests."""
    series = series.dropna()
    results = {}
    
    # ADF
    try:
        adf_res = adfuller(series, autolag='AIC')
        results['ADF'] = {
            'stat': adf_res[0],
            'pval': adf_res[1],
            'lags': adf_res[2],
            'nobs': adf_res[3],
            'is_stationary': adf_res[1] <= 0.05
        }
    except:
        results['ADF'] = None

    # KPSS
    try:
        kpss_res = kpss(series, regression='c', nlags="auto")
        results['KPSS'] = {
            'stat': kpss_res[0],
            'pval': kpss_res[1],
            'lags': kpss_res[2],
            'is_stationary': kpss_res[1] > 0.05
        }
    except:
        results['KPSS'] = None
        
    return results

@st.cache_data(ttl="2h")
def get_decomposition_insights(series, period=12):
    """Analyzes trend and seasonality strength."""
    try:
        decomp = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
        remarks = []
        
        trend_std = decomp.trend.dropna().std()
        seasonal_std = decomp.seasonal.dropna().std()
        resid_std = decomp.resid.dropna().std()
        
        if trend_std > 0.5 * series.std():
            remarks.append("🔹 Strong **trend** detected.")
        else:
            remarks.append("🔹 Weak or no significant trend.")
            
        if seasonal_std > 0.5 * series.std():
            remarks.append("🔹 Clear **seasonality** present.")
        else:
            remarks.append("🔹 Little to no seasonality detected.")
            
        if resid_std < 0.2 * series.std():
            remarks.append("🔹 Residuals show **low variance** — good fit potential.")
        else:
            remarks.append("🔹 Residuals are **noisy** — may need complex models.")
            
        return remarks, decomp
    except Exception as e:
        return ["⚠️ Decomposition failed (check data length/period)."], None

# -----------------------------------------------------------------------------
# 4. Evaluation Metrics
# -----------------------------------------------------------------------------
def evaluate(pred, true):
    """Calculates MAE, MSE, RMSE, MAPE, R²."""
    # Ensure arrays are aligned and non-zero for MAPE
    mask = true != 0
    if len(true[mask]) == 0:
        mape = 0
    else:
        mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
        
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, pred)
    
    return mae, mse, rmse, mape, r2

def color_objective(val):
    """Helper for styling dataframe cells."""
    if val == "Maximize":
        return "background-color: #d1ecf1; color: #004c6d;"
    else:
        return "background-color: #f8d7da; color: #721c24;"