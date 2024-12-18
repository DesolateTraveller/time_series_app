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
st.set_page_config(page_title="Time Series Forecasting | v0.1",
                   #page_icon='https://www.clariant.com/images/clariant-logo-small.svg',
                   page_icon= '📈',
                   layout="wide",
                   initial_sidebar_state="auto",)
#---------------------------------------
st.title(f""":rainbow[Time Series Forecasting]""")
#----------------------------------------
st.sidebar.markdown(
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
        #<p>© 2024 | Developed by: <span class="highlight">E&PT - Digital Solutions</span> | Prepared by: <a href="mailto:avijit.chakraborty@clariant.com">Avijit Chakraborty</a></p> <span class="highlight">Thank you for visiting the app | This app is created for internal use, unauthorized uses or copying is strictly prohibited | For best view of the app, please zoom out the browser to 75%.</span>    
    </div>
    """,
    unsafe_allow_html=True)
#---------------------------------------------------------------------------------------------------------------------------------
### knowledge 
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------
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
