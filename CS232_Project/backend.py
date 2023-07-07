import streamlit as st
from PIL import Image
from algos_DCT import compress_img_DCT
from algos_DCT import decompress_img_DCT
from algos_DCT import evaluate_DCT
import cv2
import os
import numpy as np
import pandas as pd 
#from algos_SVD import compress_and_display_image
#from algos_SVD import calculate_metrics
#from algos_SVD import compress_and_plot

def DCT(img,level,dir_path):
    st.header('📍 :blue[Discrete Cosine Transform]')
    st.subheader("Image Before:")
    img_before = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #img_before = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2YCR_CB)
    #img_before = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2YCR_CB)

    st.image(img)
    st.write(f"**Image shape :** {img_before.shape}")
        
    metric_dct = dict(  level = level, 
                        time_com_sec = [], 
                        time_de_sec = [], 
                        total_time_sec = [], 
                        compression_ratio = [],
                        RMS = [], 
                        SNR = [] )
       
    C_B,C_G,C_R,Q,T,T_prime,time_comp = compress_img_DCT(img_before,level,dir_path)

    img_after, time_de = decompress_img_DCT(C_B,C_G,C_R,Q,T,T_prime,dir_path)
   
    st.write("Size before:")
    st.write(os.getsize(img))
    # st.write("Size after:")
    # st.write(os.path.getsize(dir_path + '/output_DCT.jpg'))
    #comp_ratio = round(((img_size-img_rle_size)/img_size))*100
    rms, snr = evaluate_DCT(img_before,img_after)
    metric_dct.update({"time_com_sec": time_comp, 
                        "time_de_sec": time_de, 
                        "total_time_sec": np.round(time_de + time_comp,3),
                        "RMS": np.round(rms,4),
                        "SNR": np.round(snr,4)})
    
    result_DCT = pd.DataFrame([metric_dct],index=None)
    st.subheader("Result:")
    st.table(data=result_DCT)

# def SVD():
#     st.header('📍 :red[Singular Value Decomposition])
#     st.subheader("Image before: ")
#     img_before = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

#     st.image(img)
#     st.write(f"**Image shape :** {img_before.shape}")

#     order = st.slider("Compression Order", 1, 640, 1)
#     compressed_img ,compressed_img_shape, compression_time = compress_and_display_image(img, order)
#     st.subheader("Image after: ")
    
#     cv2.imwrite(dir_path + '/output_SVD.jpg' + compressed_img)
#     st.image(Image.open(dir_path + '/output_SVD.jpg'))
    
             
#     rmse, snr, compression_ratio = calculate_metrics(img, compressed_img_shape, order)
#     st.write("Compression time: ", compression_time, 'seconds')
#     st.write("Ratio compressed size / original size: ", compression_ratio)
#     st.write("Compressed image size is " + str(round(compression_ratio * 100, 2)) + " % of the original image ")
#     st.write("RMSE at order = {} is {:.2f}".format(order, rmse))
#     st.write("SNR at order = {} is {:.2f}".format(order, snr))
    
#     st.subheader("Compression at different orders: ")
#     compress_and_plot(img)
    

    
    
    
    
