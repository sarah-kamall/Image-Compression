import cv2
import numpy as np

def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 10 * np.log10((max_pixel**2) / np.sqrt(mse))
    return psnr_value

# Read original and restored images
original_image = cv2.imread('original_image.jpg', cv2.IMREAD_GRAYSCALE)
restored_image = cv2.imread('restored_image.jpg', cv2.IMREAD_GRAYSCALE)

# Ensure images have the same dimensions
original_image = cv2.resize(original_image, (restored_image.shape[1], restored_image.shape[0]))

# Calculate PSNR
psnr_value = psnr(original_image, restored_image)
print("PSNR:", psnr_value)
