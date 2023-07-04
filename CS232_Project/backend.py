import streamlit as st
from PIL import Image
from algos import compress_img_DCT
from algos import decompress_img_DCT
from algos import evaluate_DCT
import algos
import cv2
import numpy as np
import pandas as pd 

def DCT(img,level):
    st.header('Discrete Cosine Transform')
    st.subheader("Image Before:")
    img_before = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    st.image(img)
    st.text(f"**Image shape :** {img_before.shape}")
        
    metric_dct = dict(  level = level, 
                        time_com_sec = [], 
                        time_de_sec = [], 
                        total_time_sec = [], 
                        RMS = [], 
                        SNR = [] )
       
    C_B,C_G,C_R,Q,T,T_prime,image_DCT,time_comp = compress_img_DCT(img_before,level)

    img_after, time_de = decompress_img_DCT(C_B,C_G,C_R,Q,T,T_prime,image_DCT)
    st.subheader("Image After:")
    
    cv2.imwrite('output_DCT.jpg',img_after)
    st.image(Image.open('output_DCT.jpg'))
       
    rms, snr = evaluate_DCT(img_before,img_after)
    metric_dct.update({"time_com_sec": time_comp, 
                        "time_de_sec": time_de, 
                        "total_time_sec": np.round(time_de + time_comp,3),
                        "RMS": np.round(rms,4),
                        "SNR": np.round(snr,4)})
    
    result_DCT = pd.DataFrame([metric_dct],index=None)
    st.subheader("Result")
    st.table(data=result_DCT)