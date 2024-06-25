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

#-----------------------------------------------------------------------------------------------------

    #st.subheader("Parameters", divider='blue')
    st.caption('**In this section you can modify the algorithm settings.**')

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

    st.divider()

#-----------------------------------------------------------------------------------------------------

    if not df.empty:

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Information**","**Forecast**","**Validation**","**Tuning**","**Result**",])

#----------------------------------------------------

        with tab1:

            #st.subheader("Input Information", divider='blue')
        
            columns = list(df.columns)
            col1,col2 = st.columns(2)
        
            with col1:
                date_col = st.selectbox("Select date column",index= 0,options=columns,key="date", help='Column to be parsed as a date')
            with col2:
                metric_col = st.selectbox("Select Target column",index=1,options=columns,key="values", help='Quantity to be forecasted')

            df = prep_data(df, date_col, metric_col)
            output = 0

            Options = st.radio('Options', ['Plot Time-Seies data', 'Show Dataframe', 'Show Descriptive Statistics'], horizontal=True, label_visibility='collapsed', key = 'options')
        
            if Options == 'Plot Time-Seies data':
                try:
                    line_chart = alt.Chart(df).mark_line().encode(
                    x = 'ds:T',
                    y = "y:Q",tooltip=['ds:T', 'y']).properties(title="Time series preview").interactive()
                    st.altair_chart(line_chart,use_container_width=True)
                
                except:
                    st.line_chart(df['y'],use_container_width =True,height = 300)

            if Options =='Show Dataframe':
                st.dataframe(df.head(), use_container_width=True)

            if Options == 'Show Descriptive Statistics':
                st.write(df.describe().T, use_container_width=True)

#----------------------------------------------------

        with tab2:

            st.write("Fit the model on the data and generate future prediction.")

            if input:            
                if st.checkbox("Initialize model (Fit)",key="fit"):
                    if len(growth_settings)==2:
                        m = Prophet(seasonality_mode=seasonality,
                                    daily_seasonality=daily,
                                    weekly_seasonality=weekly,
                                    yearly_seasonality=yearly,
                                    growth=growth)
                            
                        if holidays:
                            m.add_country_holidays(country_name=country_code)
                        
                        if monthly:
                            m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

                        with st.spinner('Fitting the model..'):
                            m = m.fit(df)
                            future = m.make_future_dataframe(periods=periods_input,freq='D')
                            future['cap']=cap
                            future['floor']=floor
                            st.write("The model will produce forecast up to ", future['ds'].max())
                            st.success('Model fitted successfully')

                    else:
                        st.warning('Invalid configuration')
                    st.divider()
                
                if st.checkbox("Generate forecast (Predict)",key="predict"):
                    try:
                        with st.spinner("Forecasting.."):
                            forecast = m.predict(future)
                            st.success('Prediction generated successfully')
                            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], use_container_width=True)

                            col1, col2 = st.columns(2)

                            with col1: 
                                fig1 = m.plot(forecast)
                                st.write('Forecast plot')
                                st.write(fig1)                   
                                output = 1

                            with col2: 
                                if growth == 'linear':
                                    fig2 = m.plot(forecast)
                                    a = add_changepoints_to_plot(fig2.gca(), m, forecast)
                                    st.write('Growth plot')
                                    st.write(fig2)
                                    output = 1
                    except:
                        st.warning("You need to train the model first.. ")
                    st.divider()

                if st.checkbox('Show components'):
                    try:
                        with st.spinner("Loading.."):
                            fig3 = m.plot_components(forecast)
                            st.write(fig3)
                    except: 
                        st.warning("Requires forecast generation..")

#----------------------------------------------------

        with tab3:

            st.write("In this section it is possible to do cross-validation of the model.")

            col1, col2 = st.columns(2)

            with col1:
                with st.expander("**Explanation**"):
                    st.markdown("""The Prophet library makes it possible to divide our historical data into training data and testing data for cross validation.""")
                    st.write("Horizon: The data set aside for validation.")

            with col2:    
                with st.expander("**Cross-validation**"):
                    horizon = st.number_input(value= 90,label="Horizon",min_value=30,max_value=365)

            st.subheader("Metrics", divider='blue')           
            if input:
                if output == 1:
                    metrics = 0
                if st.checkbox('Calculate metrics'):
                    with st.spinner("Cross validating.."):
                        try:
                            df_cv = cross_validation(m, horizon = f"{horizon} days",parallel="processes")                                                                  
                            df_p= performance_metrics(df_cv)
                            metrics = ['mae','mape', 'mse', 'rmse']
                            selected_metric = st.selectbox("Select metric to plot",options=metrics)
                            if selected_metric:
                                fig4 = plot_cross_validation_metric(df_cv, metric=selected_metric)
                                st.write(fig4)
                                    # # PERFORMANCE METRICS TABLE
                                    # st.dataframe(df_p, use_container_width=True)
                                    # metrics = 1

                        except Exception as e:
                            st.error(f"Error during Cross-validation: {e}")
                            metrics=0

                    # if metrics == 1:
                    #     metrics = ['mae','mape', 'mse', 'rmse']
                    #     selected_metric = st.selectbox("Select metric to plot",options=metrics)
                    #     if selected_metric:
                    #         fig4 = plot_cross_validation_metric(df_cv, metric=selected_metric)
                    #         st.write(fig4)

            else:
                st.write("Create a forecast to see metrics")

#----------------------------------------------------

        with tab4:
        
            st.write("In this section it is possible to find the best combination of hyperparamenters.")

            param_grid = {'changepoint_prior_scale': changepoint_scale_values,
                        'seasonality_prior_scale': seasonality_scale_values,}

            # Generate all combinations of parameters
            all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
            mapes = []  # Store the MAPE for each params here

            if input:
                if output == 1:

                    if st.button("Optimize hyperparameters"):            
                        with st.spinner("Finding best combination. Please wait.."):                        
                            try:
                                # Use cross validation to evaluate all parameters
                                for params in all_params:
                                    m = Prophet(**params).fit(df)  # Fit model with given params
                                    df_cv = cross_validation(m, horizon=f"{horizon} days", parallel="processes")
                                    df_p = performance_metrics(df_cv, rolling_window=1)
                                    mapes.append(df_p['mape'].values[0])
                            except Exception as e:
                                st.error(f"Error during Hyperparameter optimization: {e}")
                        

                        # Find the best parameters
                        tuning_results = pd.DataFrame(all_params)
                        tuning_results['mape'] = mapes
                        st.dataframe(tuning_results, use_container_width=True)
                            
                        best_params = all_params[np.argmin(mapes)]
                    
                        st.write('The best parameter combination is:')
                        st.write(best_params)
                
                else:
                    st.write("Create a model to optimize") 
#----------------------------------------------------

        with tab5:   
        
            st.write("Finally you can export your result forecast.")
        
            if input:
                if output == 1:               
                    forecast_df = pd.DataFrame(forecast[['ds','yhat','yhat_lower','yhat_upper']])
                    forecast_data = forecast_df.to_csv(index=False)
                    st.download_button(label="Download Forecast CSV",
                                data=forecast_data,
                                file_name=f'Forecast_results.csv',
                                mime='text/csv',
                                key='download forecast')
                    st.dataframe(forecast_df, use_container_width=True)
             
         
                
   
