import matplotlib.pyplot as plt
from skimage import io, morphology

# Load the image
for i in range(1,7):
  j=str(i)
  print("Skeleton and Image Number "+j)
  image_path = "/content/"+j+".png"
  image = io.imread(image_path, as_gray=True)

# Threshold the image (convert it to binary)
  threshold = 0.5  # Adjust the threshold value as needed
  binary_image = image > threshold

# Obtain the skeleton of the binary image
  skeleton = morphology.skeletonize(binary_image)

# Display the original image and its skeleton side by side
  fig, axes = plt.subplots(1, 2, figsize=(8, 4))
  axes[0].imshow(image, cmap='gray')
  axes[0].set_title("Original Image")
  axes[0].axis('off')
  axes[1].imshow(skeleton, cmap='gray')
  axes[1].set_title("Skeleton")
  axes[1].axis('off')

# Show the plots
  plt.tight_layout()
  plt.show()
