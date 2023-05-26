import cv2
import numpy as np
from matplotlib import pyplot as plt

def high_pass_filter(image_path, cutoff_frequency):
    # Load the image in grayscale
    image = cv2.imread(image_path, 0)

    # Perform Fourier Transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Create a mask for high-pass filtering
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 0

    # Apply the mask in the frequency domain
    fshift_filtered = fshift * mask

    # Perform inverse Fourier Transform
    img_back = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(img_back)
    img_back = np.abs(img_back)

    # Display the filtered image
    plt.imshow(img_back, cmap='gray')
    plt.title('High-pass Filtered Image')
    plt.xticks([]), plt.yticks([])
    plt.show()

# Usage
image_path = r'C:\Users\SAHYADRI\Desktop\images.png'
cutoff_frequency = 30
high_pass_filter(image_path, cutoff_frequency)
