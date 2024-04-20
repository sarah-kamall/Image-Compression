import cv2
import numpy as np

quantization_parameter = int(input("Enter quantization parameter (m value): "))


image = cv2.imread('bird.png', cv2.IMREAD_GRAYSCALE)  # Read as grayscale for simplicity

# DCT Compression
block_size = 8  # Size of the blocks for DCT compression
compressed_image = np.zeros_like(image, dtype=np.float32)

for y in range(0, image.shape[0], block_size):
    for x in range(0, image.shape[1], block_size):
        block = image[y:y+block_size, x:x+block_size]
        dct = cv2.dct(np.float32(block))
        compressed_block = np.round(dct / 1) * 1  # Example compression: rounding to nearest 10
        compressed_image[y:y+block_size, x:x+block_size] = compressed_block

# Inverse DCT to get the compressed image
compressed_image = np.uint8(compressed_image)
restored_image = np.zeros_like(compressed_image, dtype=np.uint8)

for y in range(0, compressed_image.shape[0], block_size):
    for x in range(0, compressed_image.shape[1], block_size):
        block = compressed_image[y:y+block_size, x:x+block_size]
        idct = cv2.idct(np.float32(block))
        restored_image[y:y+block_size, x:x+block_size] = np.uint8(idct)

# Display and save the images
cv2.imshow('Original Image', image)
cv2.imshow('Compressed Image', compressed_image)
cv2.imshow('Restored Image', restored_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
