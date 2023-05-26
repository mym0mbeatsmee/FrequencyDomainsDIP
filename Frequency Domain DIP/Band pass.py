import cv2
import numpy as np
from matplotlib import pyplot as plt

def band_pass_filter(image_path, lower_cutoff, upper_cutoff):
    # Read the image
    image = cv2.imread(image_path, 0)

    # Perform Fourier Transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Create a mask for the band-pass filter
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)

    # Determine the lower and upper frequency bounds
    lower_bound = int(crow - lower_cutoff)
    upper_bound = int(crow + upper_cutoff)

    # Apply the band-pass filter mask
    mask[lower_bound:upper_bound, ccol - upper_cutoff:ccol + upper_cutoff] = 1

    # Apply the mask in the frequency domain
    fshift_filtered = fshift * mask

    # Perform inverse Fourier Transform
    img_back = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(img_back)
    img_back = np.abs(img_back)

    # Display the original and filtered images
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(img_back, cmap='gray')
    plt.title('Band-pass Filtered Image'), plt.xticks([]), plt.yticks([])
    plt.show()

# Usage
image_path = r'C:\Users\SAHYADRI\Desktop\images.png'
lower_cutoff = 30
upper_cutoff = 80
band_pass_filter(image_path, lower_cutoff, upper_cutoff)
