import numpy as np
import os
import scipy
import time
from PIL import Image
from matplotlib import pyplot as plt
import math
from tqdm import tqdm

def img2double(image):

    image = np.array(image)          
    image = image.astype(float)/ 255.0   ## Thêm dòng code này:  image = image.astype(float)/ 255.0 

    return image     ## Chuyển dòng code này từ image.astype(float)/ 255.0 thành image.

def svd(matrix, full_matrices=True, compute_uv=True):
    # Compute the eigenvalues and eigenvectors of A^T * A
    eigenvalues, eigenvectors = np.linalg.eig(matrix.T @ matrix)

    # Sort the eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute the singular values and singular vectors
    singular_values = np.sqrt(np.abs(eigenvalues))

    # Compute the matrices U and V
    if full_matrices:
        U = matrix @ eigenvectors / singular_values
        V = eigenvectors
    else:
        num_singular_values = min(matrix.shape)
        U = matrix @ eigenvectors[:, :num_singular_values] / singular_values[:num_singular_values]
        V = eigenvectors[:, :num_singular_values]

    if compute_uv:
        return U, singular_values, V.T
    else:
        return singular_values

    return result

def svd_compressor(image, order):
    """Returns the compressed image channel at the specified order"""

    # Create an array filled with zeros having the shape of the image
    compressed = np.zeros(image.shape)
    
    # Get the U, S and V terms (S = SIGMA)
    U, S, V = svd(image)

    # Loop over U columns (Ui), S diagonal terms (Si) and V rows (Vi) until the chosen order
    for i in range(order):
        Ui = U[:, i].reshape(-1, 1)
        Vi = V[i, :].reshape(1, -1)
        Si = S[i]
        compressed += (Ui * Si * Vi)

    return compressed

def compress_svd(image, order):
    # Convert image to float
    image = img2double(image)   ##Chuyển dòng code này lên đầu 
    # Use nbytes to get the size of the numpy array in bytes
    original_size = image.shape[0]*image.shape[1]*image.shape[2]  ## Đổi dòng original_size = 640*640*3 thành original_size = image.shape[0]*image.shape[1]*image.shape[2]

    # Initialize start time
    start_time = time.time()
    
    #Separation of the image channels
    red_image = np.array(image)[:, :, 0]
    green_image = np.array(image)[:, :, 1]
    blue_image = np.array(image)[:, :, 2]

    # Compression of each channel
    red_comp = svd_compressor(red_image, order)
    green_comp = svd_compressor(green_image, order)
    blue_comp = svd_compressor(blue_image, order)

    # Recombinasion of the colored image
    compressed_image = np.zeros((np.array(image).shape[0], np.array(image).shape[1], 3))
    compressed_image[:, :, 0] = red_comp
    compressed_image[:, :, 1] = green_comp
    compressed_image[:, :, 2] = blue_comp
    compressed_image = np.clip(compressed_image, 0, 1)


    # Calculate compression time
    end_time = time.time()
    compression_time = end_time - start_time
    compressed_image  ## Thêm dòng compressed_image 
    # Compute the size reduction of compressed image
    compressed_size = order * (1 + image.shape[0] + image.shape[1]) * image.shape[2]  ##Chuyển dòng compressed_size = order * (1 + 640 + 640) * 3 thành  compressed_size = order * (1 + image.shape[0] + image.shape[1]) * image.shape[2]
    size_reduction = ((compressed_size * 1.0 / original_size)/2)*100 ##Chuyển dòng size_reduction = compressed_size * 1.0 / original_size thành size_reduction = (compressed_size * 1.0 / original_size)*100
    
    return compressed_image, compression_time, size_reduction, compressed_image.shape, image.shape ## Thêm compressed_image.shape, image.shape

### THÊM HÀM decompress_svd(_) sau hàm compress_svd
def decompress_svd(compressed_image, order):
    # Convert compressed image to float
    compressed_image = img2double(compressed_image)

    # Use nbytes to get the size of the numpy array in bytes
    original_size = compressed_image.shape[0] * compressed_image.shape[1] * compressed_image.shape[2]

    # Initialize start time
    start_time = time.time()

    # Separation of the compressed image channels
    red_comp = np.array(compressed_image)[:, :, 0]
    green_comp = np.array(compressed_image)[:, :, 1]
    blue_comp = np.array(compressed_image)[:, :, 2]

    # Decompress each channel
    red_image = svd_compressor(red_comp, order)
    green_image = svd_compressor(green_comp, order)
    blue_image = svd_compressor(blue_comp, order)

    # Recombine the channels
    decompressed_image = np.zeros((np.array(compressed_image).shape[0], np.array(compressed_image).shape[1], 3))
    decompressed_image[:, :, 0] = red_image
    decompressed_image[:, :, 1] = green_image
    decompressed_image[:, :, 2] = blue_image
    decompressed_image = np.clip(decompressed_image, 0, 1)

    # Calculate decompression time
    end_time = time.time()
    decompression_time = end_time - start_time

    return decompression_time

def svd_evaluation(image, compressed_image):
    mse = np.mean((image - compressed_image)**2)
    signal_power = np.max(image) ** 2

    rmse = np.sqrt(mse)
    snr = 10 * math.log10(signal_power / mse)

    return rmse, snr 
