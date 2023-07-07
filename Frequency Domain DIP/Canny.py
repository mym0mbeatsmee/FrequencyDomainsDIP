import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("/content/wallpaper1.jpg", 0)  # Read the image in grayscale

# Apply Canny edge detection
edges = cv2.Canny(image, 100, 200)  # Adjust the threshold values as needed

# Display the original image and the detected edges
plt.subplot(121),plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(122),plt.imshow(edges, cmap="gray")
plt.title("Canny Edges")
plt.axis("off")
plt.show()
