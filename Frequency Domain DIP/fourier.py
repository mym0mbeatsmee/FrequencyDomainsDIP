import cv2
import numpy as np
from matplotlib import pyplot as plt

def image_fourier_transform(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, 0)

    # Perform Fourier Transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Display the magnitude spectrum
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.xticks([]), plt.yticks([])
    plt.show()

# Usage
read=r'C:\Users\SAHYADRI\Desktop\images.png'
image_path = read
image_fourier_transform(image_path)

