import streamlit as st
from PIL import Image
from algos_DCT import compress_img_DCT
from algos_DCT import decompress_img_DCT
from algos_DCT import evaluate_DCT
from compressor_SVD import img2double, compress_svd, svd_evaluation
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
    comp_ratio = round(((img_size-img_size_after)/img_size)*100)
    rms, snr = evaluate_DCT(img_before,img_after)
    metric_dct.update({"time_compress(s)": time_comp, 
                        "time_decompress(s)": time_de, 
                        "total_time(s)": np.round(time_de + time_comp,3),
                        "compression_ratio(%)": comp_ratio,
                        "RMS": np.round(rms,4),
                        "SNR": np.round(snr,4)})
    
    result_DCT = pd.DataFrame([metric_dct],index=None)
    st.subheader("Result:")
    st.table(data=result_DCT)

def SVD(img, level):
    st.header('📍 :blue[Singular Value Decomposition]')
    #st.markdown("Please upload your image and set the compression parameters.")
    # Set the compression parameter and the SVD algorithm
    k = level
        
    compressed_image, compression_time, size_reduction = compress_svd(img, k)
    # Display compressed image
    st.image(compressed_image, caption="Image after", use_column_width=True)
    st.write(f"Compression Time: {round(compression_time, 3)} seconds")
    st.write(f"Size Reduction: {size_reduction}%")
            
    rmse, snr = svd_evaluation(img, compressed_image)
    st.write(f"RMSE at k = {k} is {round(rmse,3)}")
    st.write(f"SNR at k = {k} is {round(snr,3)}")
            

    
    
    
    
