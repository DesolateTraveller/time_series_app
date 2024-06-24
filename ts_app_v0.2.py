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
import holidays
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
#---------------------------------------
import altair as alt
import plotly as plt
import plotly.offline as pyoff
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
#---------------------------------------
import os
import itertools
from PIL import Image
from datetime import datetime
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
#import custom_style()
#image = Image.open('Image_Clariant.png')
st.set_page_config(page_title="Time Series Forecasting | App",
                   page_icon='https://www.clariant.com/images/clariant-logo-small.svg',
                   layout="wide",
                   initial_sidebar_state="auto",)
#st.sidebar.image(image, use_column_width='auto') 
#----------------------------------------
st.title(f""":rainbow[Time Series Forecasting App | v0.1]""")
st.markdown('Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a>', 
            unsafe_allow_html=True)
st.info('**Disclaimer : :blue[Thank you for visiting the app] | Unauthorized uses or copying of the app is strictly prohibited | Click the :blue[sidebar] to follow the instructions to start the applications.**', icon="ℹ️")
#----------------------------------------
# Set the background image
st.divider()

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

def load_csv():  
    df_input = pd.DataFrame()    
    df_input=pd.read_csv(input,sep=None, engine='python', encoding='utf-8',parse_dates=True,infer_datetime_format=True)
    return df_input

def prep_data(df, date_col, metric_col):
    df_input = df.rename({date_col:"ds",metric_col:"y"},errors='raise',axis=1)
    st.success("The selected date column is now labeled as **ds** and the Target column as **y**")
    df_input = df_input[['ds','y']]
    df_input =  df_input.sort_values(by='ds',ascending=True)
    return df_input


#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------

st.sidebar.header("Input", divider='blue')
st.sidebar.info('Please choose from the following options and follow the instructions to start the application.', icon="ℹ️")
data_source = st.sidebar.radio("**:blue[Select the main source]**", ["File Upload", "AWS S3","Sharepoint"],)
