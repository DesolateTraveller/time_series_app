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
from typing import Any, Dict, Tuple
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

## check 'ts_app_prep.py' file :