import cv2
import os
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

# Create folders to store images if they don't exist
if not os.path.exists('restored_images'):
    os.makedirs('restored_images')
if not os.path.exists('compressed_images'):
    os.makedirs('compressed_images')

image = cv2.imread('mountain.png')
image_size_bytes = os.path.getsize('mountain.png')

# Convert bytes to megabytes
image_size_mbs = image_size_bytes / (1024 * 1024)
print("Original Image Size:", image_size_mbs, "MB")

# Split the image into its RGB channels
b, g, r = cv2.split(image)

psnrarr = []
for quantization_parameter in range(1, 5):
    # DCT Compression for each channel
    block_size = 8
    s = image.shape[0] * quantization_parameter // 8
    sc = image.shape[1] * quantization_parameter // 8
    print(s)
    compressed_channels = []
    compressed_image_slides = []

    for channel in [b, g, r]:
        compressed_channel = np.zeros_like(channel, dtype=np.float32)
        compressed_image_slide = np.zeros((s, sc), dtype=np.float32)
        i = 0
        for y in range(0, channel.shape[0], block_size):
            j = 0
            for x in range(0, channel.shape[1], block_size):
                block = channel[y:y+block_size, x:x+block_size]
                dct = cv2.dct(block.astype(np.float32))
                # Apply quantization and compression
                compressed_block = dct
                mblock = np.zeros_like(block, dtype=np.float32)
                mblock[:quantization_parameter, :quantization_parameter] = compressed_block[:quantization_parameter, :quantization_parameter]
                compressed_channel[y:y+block_size, x:x+block_size] = mblock
                compressed_image_slide[i:i+quantization_parameter, j:j+quantization_parameter] = (mblock[:quantization_parameter, :quantization_parameter])
                j += quantization_parameter
            i += quantization_parameter
        compressed_image_slides.append(compressed_image_slide)
        compressed_channels.append(compressed_channel)

    # Inverse DCT to get the compressed image for each channel
    restored_channels = np.zeros_like(image, dtype=np.float32)
    compressed_img = cv2.merge(compressed_image_slides)
    for i, compressed_channel in enumerate(compressed_channels):
        for y in range(0, compressed_channel.shape[0], block_size):
            for x in range(0, compressed_channel.shape[1], block_size):
                block = compressed_channel[y:y+block_size, x:x+block_size]
                idct = cv2.idct(block.astype(np.float32))
                restored_channels[y:y+block_size, x:x+block_size, i] = np.float32(idct)

    # Save the compressed image and restored image to folders
    cv2.imwrite(os.path.join('compressed_images', f'compressed_image_{quantization_parameter}.png'), compressed_img)
    cv2.imwrite(os.path.join('restored_images', f'restored_img_{quantization_parameter}.png'), restored_channels)

    # Get the size of the compressed image file
    compressed_image_size_bytes = os.path.getsize(os.path.join('compressed_images', f'compressed_image_{quantization_parameter}.png'))
    compressed_image_size_mbs = compressed_image_size_bytes / (1024 * 1024)
    print("Compressed Image Size:", compressed_image_size_mbs, "MB")
    
    restored_image_size_bytes = os.path.getsize(os.path.join('restored_images', f'restored_img_{quantization_parameter}.png'))
    restored_image_size_mbs = restored_image_size_bytes / (1024 * 1024)
    print("Restored Image Size:", restored_image_size_mbs, "MB")

    psnrarr.append(psnr(image, restored_channels))

drawpsnrvsm(psnrarr)
