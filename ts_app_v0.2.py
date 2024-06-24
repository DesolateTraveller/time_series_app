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

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

if data_source == "File Upload" :   

    file = st.sidebar.file_uploader("**:blue[Choose a file]**",
                                type=["xlsx","csv"],
                                accept_multiple_files=True,
                                key=0)
    if file is not None:
        df = pd.DataFrame()
        with st.spinner('Loading data..'):
            for file in file:
                df = pd.read_csv(file)
    st.sidebar.divider()

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

    st.subheader("Parameters configuration", divider='blue')
    st.write('In this section you can modify the algorithm settings.')

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:        
        
        with st.expander("**Horizon**"):
            periods_input = st.number_input('Select how many future periods (days) to forecast.',
            min_value = 1, max_value = 366,value=90)

    with col2: 

        with st.expander("**Trend**"):
            st.write("Add or remove components:")
            daily = st.checkbox("Daily")
            weekly= st.checkbox("Weekly")
            monthly = st.checkbox("Monthly")
            yearly = st.checkbox("Yearly")

    with col3: 
        
        with st.expander("**Seasonality**"):
            st.markdown("""The default seasonality used is additive, but the best choice depends on the specific case, therefore specific domain knowledge is required.""")
            seasonality = st.radio(label='Seasonality',options=['additive','multiplicative'])

    with col4: 
        
        with st.expander("**Growth**"):
            st.write('Prophet uses by default a linear growth model.')         

            growth = st.radio(label='Growth model',options=['linear',"logistic"]) 
            if growth == 'linear':
                growth_settings= {'cap':1,'floor':0}
                cap=1
                floor=0
                df['cap']=1
                df['floor']=0

            if growth == 'logistic':
                st.info('Configure saturation')
                cap = st.slider('Cap',min_value=0.0,max_value=1.0,step=0.05)
                floor = st.slider('Floor',min_value=0.0,max_value=1.0,step=0.05)
                if floor > cap:
                    st.error('Invalid settings. Cap must be higher than floor.')
                    growth_settings={}

                if floor == cap:
                    st.warning('Cap must be higher than floor')
                else:
                    growth_settings = {'cap':cap,'floor':floor}
                    df['cap']=cap
                    df['floor']=floor

    with col5: 

        with st.expander('**Holidays**'):    
            countries = ['Country name','United States','India', 'United Kingdom', 'France','Germany']
            with st.container():
                years=datetime.now().year
                selected_country = st.selectbox(label="Select country",options=countries)
                #holidays = st.checkbox('Add country holidays to the model')

                if selected_country == 'India':
                    for date, name in sorted(holidays.IN(years=years).items()):
                        st.write(date,name)
                        country_code = 'IN' 
                            
                if selected_country == 'United Kingdom':
                    for date, name in sorted(holidays.GB(years=years).items()):
                        st.write(date,name)
                        country_code = 'GB'                      

                if selected_country == 'United States':                   
                    for date, name in sorted(holidays.US(years=years).items()):
                        st.write(date,name)
                        country_code = 'US'

                if selected_country == 'France':                    
                    for date, name in sorted(holidays.FR(years=years).items()):
                        st.write(date,name)
                        country_code = 'FR'
                            
                if selected_country == 'Germany':                    
                    for date, name in sorted(holidays.DE(years=years).items()):
                        st.write(date,name)
                        country_code = 'DE'

                else:
                    holidays = False
                            
                holidays = st.checkbox('Add country holidays to the model')

    with col6: 

        with st.expander('**Hyperparameters**'):
            st.write('In this section it is possible to tune the scaling coefficients.')
            
            seasonality_scale_values= [0.01, 0.1, 1.0, 10.0]    
            changepoint_scale_values= [0.001, 0.01, 0.1, 0.5]

            st.write("The changepoint prior scale determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints.")
            changepoint_scale= st.select_slider(label= 'Changepoint prior scale',options=changepoint_scale_values)
            
            st.write("The seasonality change point controls the flexibility of the seasonality.")
            seasonality_scale= st.select_slider(label= 'Seasonality prior scale',options=seasonality_scale_values)
