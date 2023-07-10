import streamlit as st
from PIL import Image
from algos_DCT import compress_img_DCT
from algos_DCT import decompress_img_DCT
from algos_DCT import evaluate_DCT
from compressor_SVD import img2double, compress_svd, svd_evaluation, decompress_svd
import cv2
import os
import numpy as np
import pandas as pd 

def DCT(img,level,dir_path):
    st.header('📍 :blue[Discrete Cosine Transform]')
    st.subheader("Image Before:")
    
    img_before = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(dir_path + '/input_.jpg', img_before)
    st.image(Image.open(dir_path + '/input_.jpg'))
    img_size = int(os.path.getsize(dir_path + '/input_.jpg'))
    #img_before = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2YCR_CB)
    #img_before = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2YCR_CB)
    #st.image(img)
    st.write(f"**Image shape :** {img_before.shape}")
        
    metric_dct = {}
       
    C_B,C_G,C_R,Q,T,T_prime,time_comp = compress_img_DCT(img_before,level,dir_path)

    img_after, time_de = decompress_img_DCT(C_B,C_G,C_R,Q,T,T_prime,dir_path)

    # st.write("Size after:")
    img_size_after = int(os.path.getsize(dir_path + '/output_DCT.jpg'))
    comp_ratio = round(abs((img_size-img_size_after)/img_size))*100
    rms, snr = evaluate_DCT(img_before,img_after)
    metric_dct.update({ "time(s)": np.round(time_de + time_comp,3),
                        "compression_ratio(%)": comp_ratio,
                        "RMSE": np.round(rms,2),
                        "PSNR": np.round(snr,2)})
    
    result_DCT = pd.DataFrame([metric_dct],index=None)
    st.subheader("Result_DCT:")
    st.table(data=result_DCT)

def SVD(img, k):
    st.header('📍 :blue[Singular Value Decomposition]')
    
    #st.markdown("Please upload your image and set the compression parameters.")
    # Set the compression parameter and the SVD algorithm
    metric_svd = {}   
    compressed_image_1, compressed_image_2, compression_time, size_reduction, compressed_image_1.shape, img.shape = compress_svd(img, k)
    decompression_time = decompress_svd(img, k)

    st.subheader("Image Before:")
    st.image(img)
    st.write(f"**Image shape :** {img.shape}")

    # Display compressed image
    st.subheader("Image After:")
    st.image(compressed_image_1)
    st.write(f"**Compressed image shape :** {compressed_image_1.shape}")

    rmse, snr = svd_evaluation(img, compressed_image_2)
    metric_svd.update({"time(s)": np.round(compression_time + decompression_time, 3),
                        "compression_ratio(%)": round(size_reduction),
                        "RMSE": np.round(rmse,2),
                        "PSNR": np.round(snr,2)})
    result_SVD = pd.DataFrame([metric_svd],index=None)
    st.subheader("Result:")
    st.table(data=result_SVD)
            
            
            

    
    
    
    
