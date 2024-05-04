import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
def psnr(original, compressed_channels):
    # Calculate squared differences
    squared_diff = np.square(original - compressed_channels)
    mse_value = np.mean(squared_diff)
    max_pixel = 255.0
    psnr_value = 10 * np.log10((max_pixel**2) / mse_value)
    return psnr_value


def drawpsnrvsm(psnr):
    m = [1, 2, 3, 4]
    plt.plot(m, psnr, marker='o')
    plt.xlabel('m')
    plt.ylabel('PSNR')
    plt.title('PSNR vs m')
    plt.grid(True)
    plt.show()

image = cv2.imread('mountain.png')

image_size_bytes = os.path.getsize('mountain.png')

# Convert bytes to megabytes
image_size_mbs = image_size_bytes / (1024 * 1024)

print("Original Image Size:", image_size_mbs, "MB")

# Split the image into its RGB channels
b, g, r = cv2.split(image)
cv2.imshow('Blue Channel', b)
cv2.imshow('Green Channel', g)
cv2.imshow('Red Channel', r)
psnrarr = []
for quantization_parameter in range(1,5):

    # Read the color image
    # # DCT Compression for each channel
    block_size = 8
    compressed_channels = []
    for channel in [b, g, r]:
        compressed_channel = np.zeros_like(channel, dtype=np.float32)
        for y in range(0, channel.shape[0], block_size):
            for x in range(0, channel.shape[1], block_size):
                block = channel[y:y+block_size, x:x+block_size]
                dct = cv2.dct(block.astype(np.float32))
                # Apply quantization and compression
                compressed_block = dct  # Example compression: rounding to nearest 10       #-- WHY THIS
                mblock = np.zeros_like(block, dtype=np.float32)
                mblock[:quantization_parameter, :quantization_parameter] = compressed_block[:quantization_parameter, :quantization_parameter]
                compressed_channel[y:y+block_size, x:x+block_size] = mblock
               
        compressed_channels.append(compressed_channel)
    # Inverse DCT to get the compressed image for each channel
    restored_channels = np.zeros_like(image, dtype=np.uint8)
    for i, compressed_channel in enumerate(compressed_channels):
        for y in range(0, compressed_channel.shape[0], block_size):
            for x in range(0, compressed_channel.shape[1], block_size):
                block = compressed_channel[y:y+block_size, x:x+block_size]
                idct = cv2.idct(block.astype(np.float32))
                restored_channels[y:y+block_size, x:x+block_size, i] = np.uint8(idct)

    # Save the compressed image to a temporary file
    cv2.imwrite('compressed_image.png', restored_channels)

    # Get the size of the compressed image file
    compressed_image_size_bytes = os.path.getsize('compressed_image.png')
    compressed_image_size_mbs = compressed_image_size_bytes / (1024 * 1024)
    print("Compressed Image Size:", compressed_image_size_mbs, "MB")
    cv2.imshow('Original Image', image)
    cv2.imshow('Restored Image', restored_channels)

    psnrarr.append(psnr(image,restored_channels))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

drawpsnrvsm(psnrarr)