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
#---------------------------------------
from typing import Any, Dict, List, Tuple

from typing import Any, Dict, List

import streamlit as st
from ts_app_prep import clean_df
from ts_app_prep import (
    add_cap_and_floor_cols,
    check_dataset_size,
    filter_and_aggregate_df,
    format_date_and_target,
    format_datetime,
    print_empty_cols,
    print_removed_cols,
    remove_empty_cols,
    resample_df,
)
from ts_app_prep import get_train_set, get_train_val_sets
from ts_app_prep import display_links, display_save_experiment_button
from ts_app_prep import (
    plot_components,
    plot_future,
    plot_overview,
    plot_performance,
)
from ts_app_prep import input_cleaning, input_dimensions, input_resampling
from ts_app_prep import (
    input_columns,
    input_dataset,
    input_future_regressors,
)
from ts_app_prep import (
    input_cv,
    input_forecast_dates,
    input_train_dates,
    input_val_dates,
)
from ts_app_prep import input_metrics, input_scope_eval
from ts_app_prep import (
    input_holidays_params,
    input_other_params,
    input_prior_scale_params,
    input_regressors,
    input_seasonality_params,
)
from ts_app_prep import forecast_workflow
from ts_app_prep import load_config, load_image

import io
import os
import re
import uuid
import toml
import base64
import requests
import toml
from PIL import Image
from pathlib import Path
from base64 import b64encode
from zipfile import ZipFile
#---------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#---------------------------------------
import datetime
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
#---------------------------------------
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#---------------------------------------
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation
#---------------------------------------
from vacances_scolaires_france import SchoolHolidayDates
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
#import custom_style()
#image = Image.open('Image_Clariant.png')
st.set_page_config(page_title="Time Series App | v0.1",
                   page_icon='https://www.clariant.com/images/clariant-logo-small.svg',
                   layout="wide",
                   initial_sidebar_state="auto",)
#st.sidebar.image(image, use_column_width='auto') 
#----------------------------------------
st.title(f""":rainbow[Time Series App | v0.1]""")
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


# Load config
config, instructions, readme = load_config("config_streamlit.toml", "config_instructions.toml", "config_readme.toml")

# Initialization
dates: Dict[Any, Any] = dict()
report: List[Dict[str, Any]] = []

# Info
with st.expander(
    "Streamlit app to build a time series forecasting model in a few clicks", expanded=False
):
    st.write(readme["app"]["app_intro"])
    st.write("")
st.write("")
#st.sidebar.image(load_image("logo.png"), use_column_width=True)
display_links(readme["links"]["repo"], readme["links"]["article"])


st.sidebar.title("1. Data")

# Load data
with st.sidebar.expander("Dataset", expanded=True):
    df, load_options, config, datasets = input_dataset(config, readme, instructions)
    df, empty_cols = remove_empty_cols(df)
    print_empty_cols(empty_cols)

# Column names
with st.sidebar.expander("Columns", expanded=True):
    date_col, target_col = input_columns(config, readme, df, load_options)
    df = format_date_and_target(df, date_col, target_col, config, load_options)

# Filtering
with st.sidebar.expander("Filtering", expanded=False):
    dimensions = input_dimensions(df, readme, config)
    df, cols_to_drop = filter_and_aggregate_df(df, dimensions, config, date_col, target_col)
    print_removed_cols(cols_to_drop)

# Resampling
with st.sidebar.expander("Resampling", expanded=False):
    resampling = input_resampling(df, readme)
    df = format_datetime(df, resampling)
    df = resample_df(df, resampling)
    check_dataset_size(df, config)

# Cleaning
with st.sidebar.expander("Cleaning", expanded=False):
    cleaning = input_cleaning(resampling, readme, config)
    df = clean_df(df, cleaning)
    check_dataset_size(df, config)

st.sidebar.title("2. Modelling")

# Prior scale
with st.sidebar.expander("Prior scale", expanded=False):
    params = input_prior_scale_params(config, readme)

# Seasonalities
with st.sidebar.expander("Seasonalities", expanded=False):
    params = input_seasonality_params(config, params, resampling, readme)

# Holidays
with st.sidebar.expander("Holidays"):
    params = input_holidays_params(params, readme, config)

# External regressors
with st.sidebar.expander("Regressors"):
    params = input_regressors(df, config, params, readme)

# Other parameters
with st.sidebar.expander("Other parameters", expanded=False):
    params = input_other_params(config, params, readme)
    df = add_cap_and_floor_cols(df, params)

st.sidebar.title("3. Evaluation")

# Choose whether or not to do evaluation
evaluate = st.sidebar.checkbox(
    "Evaluate my model", value=True, help=readme["tooltips"]["choice_eval"]
)

if evaluate:

    # Split
    with st.sidebar.expander("Split", expanded=True):
        use_cv = st.checkbox(
            "Perform cross-validation", value=False, help=readme["tooltips"]["choice_cv"]
        )
        dates = input_train_dates(df, use_cv, config, resampling, dates)
        if use_cv:
            dates = input_cv(dates, resampling, config, readme)
            datasets = get_train_set(df, dates, datasets)
        else:
            dates = input_val_dates(df, dates, config)
            datasets = get_train_val_sets(df, dates, config, datasets)

    # Performance metrics
    with st.sidebar.expander("Metrics", expanded=False):
        eval = input_metrics(readme, config)

    # Scope of evaluation
    with st.sidebar.expander("Scope", expanded=False):
        eval = input_scope_eval(eval, use_cv, readme)

else:
    use_cv = False

st.sidebar.title("4. Forecast")

# Choose whether or not to do future forecasts
make_future_forecast = st.sidebar.checkbox(
    "Make forecast on future dates", value=False, help=readme["tooltips"]["choice_forecast"]
)
if make_future_forecast:
    with st.sidebar.expander("Horizon", expanded=False):
        dates = input_forecast_dates(df, dates, resampling, config, readme)
    with st.sidebar.expander("Regressors", expanded=False):
        datasets = input_future_regressors(
            datasets, dates, params, dimensions, load_options, date_col
        )

# Launch training & forecast
if st.checkbox(
    "Launch forecast",
    value=False,
    help=readme["tooltips"]["launch_forecast"],
):

    if not (evaluate | make_future_forecast):
        st.error("Please check at least 'Evaluation' or 'Forecast' in the sidebar.")

    track_experiments = st.checkbox(
        "Track experiments", value=False, help=readme["tooltips"]["track_experiments"]
    )

    datasets, models, forecasts = forecast_workflow(
        config,
        use_cv,
        make_future_forecast,
        evaluate,
        cleaning,
        resampling,
        params,
        dates,
        datasets,
        df,
        date_col,
        target_col,
        dimensions,
        load_options,
    )

    # Visualizations

    if evaluate | make_future_forecast:
        st.write("# 1. Overview")
        report = plot_overview(
            make_future_forecast, use_cv, models, forecasts, target_col, cleaning, readme, report
        )

    if evaluate:
        st.write(
            f'# 2. Evaluation on {"CV" if use_cv else ""} {eval["set"].lower()} set{"s" if use_cv else ""}'
        )
        report = plot_performance(
            use_cv, target_col, datasets, forecasts, dates, eval, resampling, config, readme, report
        )

    if evaluate | make_future_forecast:
        st.write(
            "# 3. Impact of components and regressors"
            if evaluate
            else "# 2. Impact of components and regressors"
        )
        report = plot_components(
            use_cv,
            make_future_forecast,
            target_col,
            models,
            forecasts,
            cleaning,
            resampling,
            config,
            readme,
            df,
            report,
        )

    if make_future_forecast:
        st.write("# 4. Future forecast" if evaluate else "# 3. Future forecast")
        report = plot_future(models, forecasts, dates, target_col, cleaning, readme, report)

    # Save experiment
    if track_experiments:
        display_save_experiment_button(
            report,
            config,
            use_cv,
            make_future_forecast,
            evaluate,
            cleaning,
            resampling,
            params,
            dates,
            date_col,
            target_col,
            dimensions,
        )
