import cv2
import numpy as np
from matplotlib import pyplot as plt

def notch_filter(image_path, notch_center, notch_radius):
    # Read the image
    image = cv2.imread(image_path, 0)

    # Perform Fourier Transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Create a mask for the notch filter
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)

    # Determine the coordinates of the notch
    notch_x, notch_y = notch_center
    mask[ccol - notch_radius:ccol + notch_radius, crow - notch_radius:notch_y + notch_radius] = 0
    mask[ccol - notch_radius:ccol + notch_radius, notch_y - notch_radius:ccol + notch_radius] = 0

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
    plt.title('Notch Filtered Image'), plt.xticks([]), plt.yticks([])
    plt.show()

# Usage
image_path = r'C:\Users\SAHYADRI\Desktop\images.png'
notch_center = (250, 250)  # Coordinates of the notch center
notch_radius = 20  # Radius of the notch
notch_filter(image_path, notch_center, notch_radius)
