import cv2
import numpy as np
from matplotlib import pyplot as plt

def homomorphic_filter(image_path, cutoff_freq, gamma_low, gamma_high):
    # Read the image and convert to float32
    image = cv2.imread(image_path, 0).astype(np.float32)

    # Perform log transform
    image_log = np.log1p(image)

    # Perform Fourier Transform
    f = np.fft.fft2(image_log)
    fshift = np.fft.fftshift(f)

    # Create a high-pass filter mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    mask[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq] = 1

    # Apply the high-pass filter in the frequency domain
    fshift_filtered = fshift * mask

    # Perform inverse Fourier Transform
    f_filtered_shifted = np.fft.ifftshift(fshift_filtered)
    img_filtered_log = np.fft.ifft2(f_filtered_shifted)
    img_filtered_log = np.abs(img_filtered_log)

    # Perform exponential transform
    img_filtered = np.expm1(img_filtered_log)

    # Normalize the filtered image
    img_filtered = (img_filtered - np.min(img_filtered)) / (np.max(img_filtered) - np.min(img_filtered))

    # Perform gamma correction
    img_filtered_gamma = np.power(img_filtered, gamma_high - gamma_low)
    img_filtered_gamma = img_filtered_gamma * (255 ** gamma_low)

    # Convert to uint8 and display the result
    img_filtered_gamma = img_filtered_gamma.astype(np.uint8)

    # Display the original and filtered images
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(img_filtered_gamma, cmap='gray')
    plt.title('Homomorphic Filtered Image'), plt.xticks([]), plt.yticks([])
    plt.show()

# Usage
image_path = r'C:\Users\SAHYADRI\Desktop\images.png'
cutoff_freq = 30
gamma_low = 0.2
gamma_high = 2.5
homomorphic_filter(image_path, cutoff_freq, gamma_low, gamma_high)
