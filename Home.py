import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as seaborn
import streamlit as st
import io
import os

# Load the data
df = pd.read_csv("Mall_Customers.csv")

st.title(":red[Customer Segmentation Analysis]")

st.header(":blue[Details of the Dataset]")

data_info = st.radio('Click to view the Details of the Dataset:',
                      ('Head', 'Tail','Sample', 'Columns', 'Shape', 'Info', 'Descriptive Statistics'),
                      horizontal=True)

if data_info == 'Shape':
    st.write(f"Number of Rows:  {df.shape[0]}")
    st.write(f"Number of Columns:  {df.shape[1]}")
elif data_info == 'Head':
    st.write(df.head())
elif data_info == 'Tail':
    st.write(df.tail())
elif data_info == 'Sample':
    st.write(df.sample(10))  
elif data_info == 'Columns':
    for column in list(df.columns):
        st.write(column)
elif data_info == 'Info':
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
else:
    st.write(df.describe())


@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(df)



st.markdown("<br>",unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center';><b>Click to Downlaod the Dataset</b></h5>",unsafe_allow_html=True)
col1, col2, col3 , col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3 :
    center_button = st.download_button(
     label="Download",
     data=csv,
     file_name='heart_disease_dataset.csv',
     mime='text/csv',
 )