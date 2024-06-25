import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.plot import add_changepoints_to_plot
import itertools

st.set_page_config(page_title="Time Series Forecasting | App",
                   page_icon='https://www.clariant.com/images/clariant-logo-small.svg',
                   layout="wide")

st.title(f"Time Series Forecasting App | v0.1")
st.markdown('Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a>', 
            unsafe_allow_html=True)
st.info('**Disclaimer: Unauthorized use or copying of the app is strictly prohibited.**', icon="ℹ️")

# Function to load the CSV file
def load_csv(file):
    df = pd.read_csv(file, sep=None, engine='python', encoding='utf-8', parse_dates=True, infer_datetime_format=True)
    return df

# Function to prepare the data for Prophet
def prep_data(df, date_col, metric_col):
    df = df.rename({date_col: "ds", metric_col: "y"}, errors='raise', axis=1)
    st.success("The selected date column is now labeled as **ds** and the Target column as **y**")
    df = df[['ds', 'y']].sort_values(by='ds', ascending=True)
    return df

# Sidebar inputs for file upload
st.sidebar.header("Input", divider='blue')
st.sidebar.info('Please choose from the following options to start the application.', icon="ℹ️")
data_source = st.sidebar.radio("Select the main source", ["File Upload", "AWS S3", "Sharepoint"],)

if data_source == "File Upload":
    file = st.sidebar.file_uploader("Choose a file", type=["csv"], accept_multiple_files=False, key="file_upload")
    if file:
        df = load_csv(file)
        st.sidebar.divider()
        
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

        if not df.empty:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Information**", "**Forecast**", "**Validation**", "**Tuning**", "**Result**"])

            with tab1:
                columns = list(df.columns)
                col1, col2 = st.columns(2)
                with col1:
                    date_col = st.selectbox("Select date column", options=columns, key="date", help='Column to be parsed as a date')
                with col2:
                    metric_col = st.selectbox("Select Target column", options=columns, key="values", help='Quantity to be forecasted')
                
                df = prep_data(df, date_col, metric_col)
                Options = st.radio('Options', ['Plot Time-Series data', 'Show Dataframe', 'Show Descriptive Statistics'], horizontal=True, label_visibility='collapsed', key='options')
                
                if Options == 'Plot Time-Series data':
                    try:
                        line_chart = alt.Chart(df).mark_line().encode(x='ds:T', y="y:Q", tooltip=['ds:T', 'y']).properties(title="Time series preview").interactive()
                        st.altair_chart(line_chart, use_container_width=True)
                    except:
                        st.line_chart(df['y'], use_container_width=True, height=300)

                if Options == 'Show Dataframe':
                    st.dataframe(df.head(), use_container_width=True)

                if Options == 'Show Descriptive Statistics':
                    st.write(df.describe().T, use_container_width=True)

            with tab2:
                st.write("Fit the model on the data and generate future prediction.")
                if st.checkbox("Initialize model (Fit)", key="fit"):
                    m = Prophet(
                        seasonality_mode=seasonality,
                        daily_seasonality=daily,
                        weekly_seasonality=weekly,
                        yearly_seasonality=yearly,
                        growth=growth,
                        changepoint_prior_scale=changepoint_scale,
                        seasonality_prior_scale=seasonality_scale
                    )
                    if holidays and selected_country != 'Country name':
                        m.add_country_holidays(country_name=country_code)

                    if monthly:
                        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

                    with st.spinner('Fitting the model...'):
                        m.fit(df)
                        future = m.make_future_dataframe(periods=periods_input, freq='D')
                        if growth == 'logistic':
                            future['cap'] = cap
                            future['floor'] = floor

                        st.write(f"The model will produce forecast up to {future['ds'].max()}")
                        st.success('Model fitted successfully')
                    st.divider()

                if st.checkbox("Generate forecast (Predict)", key="predict"):
                    try:
                        with st.spinner("Forecasting..."):
                            forecast = m.predict(future)
                            st.success('Prediction generated successfully')
                            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], use_container_width=True)

                            col1, col2 = st.columns(2)

                            with col1:
                                fig1 = m.plot(forecast)
                                st.write('Forecast plot')
                                st.pyplot(fig1)

                            with col2:
                                if growth == 'linear':
                                    fig2 = m.plot(forecast)
                                    add_changepoints_to_plot(fig2.gca(), m, forecast)
                                    st.write('Growth plot')
                                    st.pyplot(fig2)
                    except Exception as e:
                        st.warning(f"Error generating forecast: {e}")
                    st.divider()

                if st.checkbox('Show components'):
                    try:
                        with st.spinner("Loading..."):
                            fig3 = m.plot_components(forecast)
                            st.pyplot(fig3)
                    except Exception as e:
                        st.warning(f"Error showing components: {e}")

            with tab3:
                st.write("In this section, you can perform cross-validation of the model.")
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("**Explanation**"):
                        st.markdown("The Prophet library allows you to split your data into training and validation sets for cross-validation.")
                with col2:
                    with st.expander("**Cross-validation**"):
                        horizon = st.number_input('Enter forecast horizon in days:', min_value=1, max_value=365, value=30)

                st.subheader("Metrics", divider='blue')
                if st.checkbox('Calculate metrics'):
                    try:
                        with st.spinner("Cross validating..."):
                            df_cv = cross_validation(m, horizon=f"{horizon} days", parallel=None)
                            df_p = performance_metrics(df_cv)
                            st.dataframe(df_p)

                            metrics = ['mae', 'mape', 'mse', 'rmse']
                            selected_metric = st.selectbox("Select metric to plot", options=metrics)
                            if selected_metric:
                                fig4 = plot_cross_validation_metric(df_cv, metric=selected_metric)
                                st.pyplot(fig4)
                    except Exception as e:
                        st.error(f"Error during cross-validation: {e}")

            with tab4:
                st.write("Optimize hyperparameters to find the best combination.")

                param_grid = {
                    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
                    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                }

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

            with tab5:
                st.write("Export your forecast results.")
                if 'forecast' in locals():
                    forecast_df = pd.DataFrame(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
                    forecast_data = forecast_df.to_csv(index=False)
                    st.download_button(label="Download Forecast CSV",
                                       data=forecast_data,
                                       file_name='Forecast_results.csv',
                                       mime='text/csv')
                    st.dataframe(forecast_df, use_container_width=True)

