import streamlit as st
from PIL import Image
from algos import compress_img_DCT
from algos import decompress_img_DCT
from algos import evaluate_DCT
import algos
import cv2
import os
import numpy as np
import pandas as pd 

def DCT(img,level,dir_path):
    st.header('📍 :blue[Discrete Cosine Transform]')
    st.subheader("Image Before:")
    #img_before = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_before = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2YCR_CB)
    st.image(img)
    st.write(f"**Image shape :** {img_before.shape}")
        
    metric_dct = dict(  level = level, 
                        time_com_sec = [], 
                        time_de_sec = [], 
                        total_time_sec = [], 
                        RMS = [], 
                        SNR = [] )
       
    C_B,C_G,C_R,Q,T,T_prime,image_DCT,time_comp = compress_img_DCT(img_before,level,dir_path)

    img_after, time_de = decompress_img_DCT(C_B,C_G,C_R,Q,T,T_prime,image_DCT)
    st.subheader("Image After:")
    
    cv2.imwrite(dir_path + '/output_DCT.jpg', img_after)
    st.image(Image.open(dir_path + '/output_DCT.jpg'))
       
    rms, snr = evaluate_DCT(img_before,img_after)
    metric_dct.update({"time_com_sec": time_comp, 
                        "time_de_sec": time_de, 
                        "total_time_sec": np.round(time_de + time_comp,3),
                        "RMS": np.round(rms,4),
                        "SNR": np.round(snr,4)})
    
    result_DCT = pd.DataFrame([metric_dct],index=None)
    st.subheader("Result:")
    st.table(data=result_DCT)