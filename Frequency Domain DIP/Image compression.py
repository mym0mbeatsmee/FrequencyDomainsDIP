import cv2
import numpy as np
from matplotlib import pyplot as plt

def image_compression(image_path, compression_ratio):
    # Read the image
    image = cv2.imread(image_path, 0)

    # Perform Discrete Cosine Transform (DCT)
    dct = cv2.dct(np.float32(image))

    # Determine the number of coefficients to keep
    num_coefficients = int(dct.shape[0] * dct.shape[1] * compression_ratio)

    # Sort the DCT coefficients by magnitude
    sorted_coefficients = np.abs(dct).ravel()
    sorted_coefficients[::-1].sort()

    # Threshold the DCT coefficients
    threshold = sorted_coefficients[num_coefficients]
    dct[np.abs(dct) < threshold] = 0

    # Perform Inverse Discrete Cosine Transform (IDCT)
    compressed_image = cv2.idct(dct)

    # Convert to uint8 and display the result
    compressed_image = compressed_image.astype(np.uint8)

    # Display the original and compressed images
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(compressed_image, cmap='gray')
    plt.title('Compressed Image'), plt.xticks([]), plt.yticks([])
    plt.show()

# Usage
image_path = r'C:\Users\SAHYADRI\Desktop\images.png'
compression_ratio = 0.1  # Compression ratio: 10%
image_compression(image_path, compression_ratio)
