import numpy as np
import os
import scipy
import time
from PIL import Image
from matplotlib import pyplot as plt
import math
from tqdm import tqdm

def img2double(image):
    """
    Converts image pixel intensities to double precision floating point numbers.

    Args:
    img (numpy.ndarray): Input image as an array.

    Returns:
    numpy.ndarray: Image with pixel intensities as double precision floating point numbers.
    """
    image = np.array(image)
    image = image.astype(float)/255.0
    return image


    

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
    image = img2double(image)
    # Use nbytes to get the size of the numpy array in bytes
    original_size = image.shape[0]*image.shape[1]*image.shape[2]
    # Initialize start time
    start_time = time.time()

    # If image is RGB
    """if len(img.shape) == 3:
        red_channel, green_channel, blue_channel =  img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # XR_r = img_func.truncate_svd(red_channel, 66, r, r, svd_version=algo)[0]
        # XG_r = img_func.truncate_svd(green_channel, 66, r, r, svd_version=algo)[0]
        # XB_r = img_func.truncate_svd(blue_channel, 66, r, r, svd_version=algo)[0]
        # X_r = np.dstack((XR_r, XG_r, XB_r))
        U_B, S_B, VT_B = img_func.svd(blue_channel)
        U_G, S_G, VT_G = img_func.svd(green_channel)
        U_R, S_R, VT_R = img_func.svd(red_channel)

        # Output compressed image
        XR_r = U_R[:, :r] @ S_R[:r, :r] @ VT_R[:r, :]
        XG_r = U_G[:, :r] @ S_G[:r, :r] @ VT_G[:r, :]
        XB_r = U_B[:, :r] @ S_B[:r, :r] @ VT_B[:r, :]
        compressed_image = np.dstack((XR_r, XG_r, XB_r))

        # Truncate U, S, and VT using the rank-k approximation
        U_k = U[:, :r]
        S_k = S[:r, :r]
        VT_k = VT[:r, :]

        # Reconstruct the compressed image
        compressed_image = U_k @ S_k @ VT_k"""
    # Separation of the image channels
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
    compressed_image_1 = np.clip(compressed_image, 0, 1)
    compressed_image_2 = np.around(compressed_image*255.0).astype(int)
    #compressed_image_2 = compressed_image_2.astype(int)


    # Calculate compression time
    end_time = time.time()
    compression_time = end_time - start_time

    # Compute the size reduction of compressed image
    compressed_size = order * (1 + image.shape[0] + image.shape[1]) * image.shape[2]
    size_reduction = (compressed_size * 1.0 / original_size)*100
    #compressed_image_shape = compressed_image.shape
    return compressed_image_1,  compressed_image_2, compression_time, size_reduction, compressed_image_1.shape, image.shape

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
    decompressed_image_1 = np.clip(decompressed_image, 0, 1)
    decompressed_image_2 = np.around(decompressed_image).astype(int)

    # Calculate decompression time
    end_time = time.time()
    decompression_time = end_time - start_time

    return decompression_time

def svd_evaluation(image, compressed_image_2):
    
    mse = np.mean((compressed_image_2 - image)**2)
    signal_power = np.max(image) ** 2
    
    rmse = np.sqrt(mse)
    psnr = 10 * math.log10(signal_power / mse)
  
    return rmse, psnr

