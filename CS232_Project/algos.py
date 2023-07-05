import time
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from numpy.linalg import inv
from stqdm import stqdm
import streamlit as st
from PIL import Image

def dct_coeff():
    T = np.zeros([8,8])
    for i in range(8):
        for j in range(8):
            if i==0:
                T[i,j] = 1/np.sqrt(8)
            elif i>0:
                T[i,j] = np.sqrt(2/8)*np.cos((2*j+1)*i*np.pi/16)
    return T


def quantization_level(n):
    Q50 = np.zeros([8,8])

    Q50 = np.array([[16, 11, 10, 16, 24, 40, 52, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])

    Q = np.zeros([8,8])
    for i in range(8):
        for j in range(8):
            if n>50:
                Q[i,j]= min(np.round((100-n)/50*Q50[i,j]),255)
            else:
                Q[i,j]= min(np.round(50/n *Q50[i,j]),255)
    return Q

def dct(M,T,T_prime):
    tmp = np.zeros(M.shape)
    mask = np.zeros([8,8])
    for i in stqdm(range(M.shape[0]//8)):
        for j in range(M.shape[1]//8):
            mask = M[8*i:8*i+8,8*j:8*j+8]
            tmp[8*i:8*i+8,8*j:8*j+8] = T @ mask @ T_prime

    return (tmp)

def quantiz_div(a,b):
    tmp = np.zeros(a.shape)
    for i in range(8):
        for j in range(8):
            tmp[i,j] = np.round(a[i,j]/b[i,j])
    return tmp


def quantiz(D,Q):
    tmp = np.zeros(D.shape)
    mask = np.zeros([8,8])
    for i in stqdm(range(D.shape[0]//8)):
        for j in range(D.shape[1]//8):
            mask = quantiz_div(D[8*i:8*i+8,8*j:8*j+8],Q)
            tmp[8*i:8*i+8,8*j:8*j+8] = mask
    return (tmp)


def decompress_mul(a,b):
    tmp = np.zeros(a.shape)
    for i in range(8):
        for j in range(8):
            tmp[i,j] = a[i,j]*b[i,j]
    return tmp

def decompress(C,Q,T,T_prime):
    R = np.zeros(C.shape)
    mask = np.zeros([8,8])
    for i in stqdm(range(C.shape[0]//8)):
        for j in range(C.shape[1]//8):
            mask = decompress_mul(C[8*i:8*i+8,8*j:8*j+8],Q)
            R[8*i:8*i+8,8*j:8*j+8] = mask

    N = np.zeros(C.shape)

    for i in stqdm(range(R.shape[0]//8)):
        for j in range(R.shape[1]//8):
            mask = T_prime @ R[8*i:8*i+8,8*j:8*j+8] @ T
            N[8*i:8*i+8,8*j:8*j+8] = np.round(mask) + 128*np.ones([8,8])

    return N


def compress_img_DCT(img_before,level,dir_path):
    start_com = time.time()
    I = img_before
    B, G, R = cv2.split(I)

    H = I.shape[0]
    W = I.shape[1]

    B = B - 128*np.ones([H,W])
    G = G - 128*np.ones([H,W])
    R = R - 128*np.ones([H,W])

    T = dct_coeff()
    T_prime = inv(T)
    Q = quantization_level(level)

    st.text("DCT Process.........")
    D_R = dct(R,T,T_prime)
    D_G = dct(G,T,T_prime)
    D_B = dct(B,T,T_prime)

    tmp = cv2.merge((D_B, D_G, D_R))

    st.text("Quantiz Process.........")
    C_R = quantiz(D_R,Q)
    C_R[C_R==0] = 0
    C_G = quantiz(D_G,Q)
    C_G[C_G==0] = 0
    C_B = quantiz(D_B,Q)
    C_B[C_B==0] = 0

    image_DCT = cv2.merge((C_B,C_G,C_R))
    end_com = time.time()
    cv2.imwrite(dir_path +'/After_Quantiz'+str(level)+'.jpg',tmp)
    st.image(Image.open(dir_path + '/After_Quantiz'+str(level)+'.jpg'))
    time_comp = end_com - start_com

    return C_B,C_G,C_R,Q,T,T_prime,image_DCT,time_comp

def decompress_img_DCT(C_B,C_G,C_R,Q,T,T_prime,fileout):
    start_de = time.time()
    st.text("Decompress Process.........")
    N_R = decompress(C_R,Q,T,T_prime)
    N_G = decompress(C_G,Q,T,T_prime)
    N_B = decompress(C_B,Q,T,T_prime)
    
    #N_R = np.round(N_R).astype(np.uint8)
    #N_G = np.round(N_G).astype(np.uint8)
    #N_B = np.round(N_B).astype(np.uint8)

    image_de = cv2.merge((N_B, N_G, N_R))
    end_de = time.time()
    time_de = end_de - start_de
    #cv2.imwrite(fileout,N_I)
    st.success('Done!', icon="✅")
    return image_de, time_de
    
def evaluate_DCT(I_before,I_after):
    #print("----Before")
    #print(I_before)
    a,b,c = I_before.shape
    m,n,k = I_after.shape
    #print("----After")
    #print(I_after)
    rms = np.sqrt(np.sum(np.square(I_after-I_before)))/(m*n)

    # snr = np.sum(np.square(I_after))/np.sum(np.square(I_after-I_before))
    I_after_round = np.round(I_after).astype(np.uint8)
    snr = np.sum(np.square(I_after_round))/np.sum(np.square(I_after_round-I_before))

    return rms, snr

