import streamlit as st
from PIL import Image
from backend import DCT
import cv2
import numpy as np
import pandas as pd 
import os

st.set_page_config(
    page_title= "COMPRESS/DECOMPRESS USING DCT & SVD",
    page_icon = "😍",
    layout="wide",
    menu_items={
        'Get Help': 'https://www.facebook.com/beetibao',
        'Report a bug': "https://www.facebook.com/beetibao",
        'About': "Install and evaluate two algorithms DCT & SVD on picture"
    }
)
dir_path = os.path.dirname(os.path.realpath(__file__))
col1, col2 = st.columns((2,4))
with col1:
    st.image(Image.open(dir_path +'/Logo_UIT.png'), width=250)
with col2:
    st.title(":blue[COMPRESS/DECOMPRESS WITH DCT & SVD]")
    st.header('CS232 - Multimedia Computing')
    st.subheader('👱 Member:')
    col3, col4 = st.columns((2,2))
    with col3:
        st.subheader('Phạm Thiện Bảo')
        st.image(Image.open(dir_path + '/Member1.jpg'), width=100)
    with col4:
        st.subheader('Lê Nguyễn Tiến Đạt')
        st.image(Image.open(dir_path + '/Member2.jpg'), width=100)

with st.form("first_form"):
    st.subheader(":blue[1. Choose a picture]")
    # uploaded_files = st.file_uploader("Upload a picture here", accept_multiple_files = False)
    upload_img = st.file_uploader("**Upload an image here**", type=["png", "jpg", "jpeg","tif"])
    # img = Image.open(upload_img)
    #img_array = np.array(image)

    st.subheader(":blue[2. Choose algorithm]")
    col5, col6 = st.columns((1,1))
    with col5:
        dct = st.checkbox("**Discrete Cosine Transform**", key="dct")
    with col6:
        svd = st.checkbox("**Singular Value Decomposition**", key="svd")
        
    st.subheader(":blue[3. Choose level]")
    level = st.number_input('**Choose a number of level compression**', min_value = 10, 
                            max_value = 90, value = 50, step = 10, key = 'level', format = '%d')
    
    submitted = st.form_submit_button("Run")

if upload_img and submitted:
    st.success('**We received your request!**', icon="✅")
    img = Image.open(upload_img)
    if dct:
        DCT(img,level,dir_path)
        