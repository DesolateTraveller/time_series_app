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
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Forecasting | v0.1",
                   #page_icon='https://www.clariant.com/images/clariant-logo-small.svg',
                   page_icon= '📈',
                   layout="wide",
                   initial_sidebar_state="auto",)
#---------------------------------------
#st.title(f""":rainbow[Time Series Forecasting]""")
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    <div class="title">Time Series Forecasting</div>
    """,
    unsafe_allow_html=True
)
#----------------------------------------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F0F2F6;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
        z-index: 100;
    }
    .footer p {
        margin: 0;
    }
    .footer .highlight {
        font-weight: bold;
        color: blue;
    }
    </style>

    <div class="footer">
        <p>© 2024 | Created by : <span class="highlight">Avijit Chakraborty</span> | Prepared by: <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a></p> <span class="highlight">Thank you for visiting the app | Unauthorized uses or copying is strictly prohibited | For best view of the app, please zoom out the browser to 75%.</span>
    </div>
    """,
    unsafe_allow_html=True)
#---------------------------------------------------------------------------------------------------------------------------------
### knowledge 
#---------------------------------------------------------------------------------------------------------------------------------
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


#---------------------------------------------------------------------------------------------------------------------------------
### Main app
#---------------------------------------------------------------------------------------------------------------------------------

page = st.sidebar.selectbox("**:blue[Contents]**", ["Introduction", "Analysis"])

#---------------------------------------------------------------------------------------------------------------------------------
if page == "Introduction" :

    #st.divider()
    stats_expander = st.expander("**:red[Description]**", expanded=True)
    with stats_expander: 

        st.info("""
        xx
        """)

    stats_expander = st.expander("**:red[Steps to Use the App]**", expanded=False)
    with stats_expander: 

        st.info("""
                
            1. ***Upload Your Data:***
            - xx.
            - xx.
    
            2. ***Select Custom Range:***
            - xx.
            - xx.
            - xx.
            - xx.                
            - xx.

            3. ***Run the Analysis & Download Results:***
            - xx.               
            - xx.
         
            """)
        
    #st.divider()
    #---------------------------------------------------------------
    st.markdown("""
    <div style="background-color: #F0F8FF; padding: 10px; border-radius: 10px;">
    <h4 style="color: #6495ed;">How to Navigate:</h4>
    <p style="color: #4B4B4B;">
    Use the dropdown menu in the sidebar to access different sections:
    </p>
    <ul style="color: #4B4B4B;">
        <li><strong>Introduction:</strong> Understand the project overview and get started with the app.</li>
        <li><strong>Analysis:</strong> Upload your data and explore step-by-step analytical tools.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    #---------------------------------------------------------------

    #---------------------------------------------------------------
    st.sidebar.empty()
    st.sidebar.markdown("""
    <div style="background-color: #F9F9FB; padding: 10px; border-radius: 8px; margin-top: 20px; box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);">    
        <h5 style="color: #0056b3; font-weight: bold;">What's New</h5>
        <ul style="color: #333333; padding-left: 15px; margin: 10px 0;">
            <li><b>Version:</b> 0.1</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    #---------------------------------------------------------------
    st.markdown("---")
    stats_expander = st.expander("**:red[Note]**", expanded=True)
    with stats_expander:

        st.markdown("""
            **Need Help?** If you encounter any issues or have feedback, please contact the **Owner of the App** mentioned in the footer.
            """)
        
#---------------------------------------------------------------------------------------------------------------------------------
elif page == "Analysis":
    #st.sidebar.header(":blue[Application]", divider='blue')
    st.sidebar.divider()
    st.divider()

    file = st.sidebar.file_uploader("**:blue[Choose a file]**",
                                    type=["csv", "xls", "xlsx"], 
                                    accept_multiple_files=False, 
                                    key="file_upload")
    if file is not None:
        df = load_file(file)        #for filter
        df1 = df.copy()             #for analysis
        df2 = df.copy()             #for visualization

        st.sidebar.divider()

        target_variable = st.sidebar.selectbox("**:blue[Target Variable]**", options=["None"] + list(df.columns), key="target_variable")
        time_col = st.sidebar.selectbox("**:blue[Time Frame Column]**", options=["None"] + list(df.columns), key="time_col")
        if time_col == "None" or target_variable == "None" :
            st.warning("Please choose **target variable**, **time-frame column** to proceed with the analysis.")
        
        else:
            stats_expander = st.expander("**Preview of Data**", expanded=True)
            with stats_expander:  
                st.table(df.head(2))

            #with st.sidebar.popover("**:blue[:hammer_and_wrench: Criteria & Hyperparameters]**", help="Check the criteria & Tune the hyperparameters whenever required"):  
            stats_expander = st.sidebar.expander("**Criteria & Hyperparameters**", expanded=False)
            with stats_expander: 
                train_size_per = st.slider("**Train Size (as %)**", 10, 90, 70, 5)
                test_size_per = st.slider("**Test Size (as %)**", 10, 50, 30, 5)  
                st.divider()
                alpha = st.slider('**Alpha (Smoothing Parameter)**', min_value=0.01, max_value=1.0, value=0.2,key = 'ses_1')
                beta = st.slider('**Beta (Trend Smoothing Parameter)**', min_value=0.01, max_value=1.0, value=0.2,key = 'ses_2')
                gamma = st.slider('**Gamma (Seasonality Smoothing Parameter)**', min_value=0.01, max_value=1.0, value=0.2,key = 'ses_3') 
                st.divider()
                order_arima = st.text_input('**ARIMA Order (p,d,q)**:', '1,1,1')
                order_arima = tuple(map(int, order_arima.split(',')))     
                order_sarima = st.text_input('**SARIMA Order (p,d,q,m)**:', '1,1,1,12')
                order_sarima = tuple(map(int, order_sarima.split(',')))                                                                                             
                st.divider()
                forecast_periods = st.slider('**Forecasting periods**', min_value=30, max_value=90, value=60, key='for_ped') 
                        
            st.sidebar.divider()

            #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            tab1, tab2, tab3 = st.tabs([ "**Visualization**","**Performance**", "**Forecast**"])
            #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                
            with tab1:
                
                plt.figure(figsize=(25,5))
                plt.plot(df2, label='Target', color='blue')
                plt.xlabel('Time')
                plt.ylabel('Target')
                plt.title('Time vs Target')
                plt.legend()
                plt.grid(True)
                st.pyplot(plt,use_container_width=True)
