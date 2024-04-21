import cv2
import numpy as np

quantization_parameter = int(input("Enter quantization parameter (m value): "))

# Read the color image
image = cv2.imread('peppers.png')

# Split the image into its RGB channels
b, g, r = cv2.split(image)

# DCT Compression for each channel
block_size = 8
compressed_channels = []
for channel in [b, g, r]:
    compressed_channel = np.zeros_like(channel, dtype=np.float32)
    for y in range(0, channel.shape[0], block_size):
        for x in range(0, channel.shape[1], block_size):
            block = channel[y:y+block_size, x:x+block_size]
            dct = cv2.dct(np.float32(block))

            # Apply quantization and compression
            compressed_block = dct  # Example compression: rounding to nearest 10
            mblock = np.zeros_like(block, dtype=np.float32)
            mblock[:quantization_parameter, :quantization_parameter] = compressed_block[:quantization_parameter, :quantization_parameter]
            compressed_channel[y:y+block_size, x:x+block_size] = mblock

    compressed_channels.append(compressed_channel)

# Inverse DCT to get the compressed image for each channel
restored_channels = []
for compressed_channel in compressed_channels:
    restored_channel = np.zeros_like(compressed_channel, dtype=np.uint8)
    for y in range(0, compressed_channel.shape[0], block_size):
        for x in range(0, compressed_channel.shape[1], block_size):
            block = compressed_channel[y:y+block_size, x:x+block_size]
            idct = cv2.idct(np.float32(block))
            restored_channel[y:y+block_size, x:x+block_size] = np.uint8(idct)
    restored_channels.append(restored_channel)

# Merge the restored channels into a single color image
restored_image = cv2.merge(restored_channels)
compressed_image = cv2.merge(compressed_channels)
# Display and save the images
cv2.imshow('Original Image', image)
cv2.imshow('Compressed Image', compressed_image)
cv2.imshow('Restored Image', restored_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
