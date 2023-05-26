import cv2
import numpy as np
from matplotlib import pyplot as plt

def inverse_fourier_transform(image_path):
    # Read the image
    image = cv2.imread(image_path, 0)

    # Perform Fourier Transform
    f = np.fft.fft2(image)

    # Shift the zero frequency component to the center of the spectrum
    fshift = np.fft.fftshift(f)

    # Perform Inverse Fourier Transform
    img_back = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(img_back)
    img_back = np.abs(img_back)

    # Display the original and inverse transformed images
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(img_back, cmap='gray')
    plt.title('Inverse Fourier Transform'), plt.xticks([]), plt.yticks([])
    plt.show()

# Usage
image_path = r'C:\Users\SAHYADRI\Desktop\images.png'
inverse_fourier_transform(image_path)
