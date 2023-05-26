import cv2
import numpy as np
from matplotlib import pyplot as plt

def frequency_domain_analysis(image_path):
    # Read the image
    image = cv2.imread(image_path, 0)

    # Perform Fourier Transform
    f = np.fft.fft2(image)

    # Shift the zero frequency component to the center of the spectrum
    fshift = np.fft.fftshift(f)

    # Compute the magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Compute the phase spectrum
    phase_spectrum = np.angle(fshift)

    # Display the original image
    plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    # Display the magnitude spectrum
    plt.subplot(2, 2, 2), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    # Display the phase spectrum
    plt.subplot(2, 2, 3), plt.imshow(phase_spectrum, cmap='gray')
    plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])

    # Display the inverse Fourier Transform (reconstructed image)
    img_back = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(img_back)
    img_back = np.abs(img_back)
    plt.subplot(2, 2, 4), plt.imshow(img_back, cmap='gray')
    plt.title('Reconstructed Image'), plt.xticks([]), plt.yticks([])

    plt.show()

# Usage
image_path = r'C:\Users\SAHYADRI\Desktop\images.png'
frequency_domain_analysis(image_path)
