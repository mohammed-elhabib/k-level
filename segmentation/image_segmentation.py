import time

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from sklearn.metrics import pairwise_distances_argmin_min

from Klevel import k_level

# Load an image
image = io.imread("img_1.png")

# Reshape the image to a 2D array of pixels
height, width, channels = image.shape
data = np.reshape(image, (height * width, channels))
print(data.shape[0])
# Number of clusters (desired segments)
k = 100

time_start = time.time()

# Perform k-level clustering
level_data, final_clusters = k_level(data, k, m=1)
time_end = time.time()

print("time ", time_end - time_start)
for level_idx, clusters_at_level in enumerate(level_data):
    label_data = np.zeros((len(data),), dtype=int)
    for center_index, center in enumerate(clusters_at_level.get_clusters()):
        for point_index in center.get_indexes():
            label_data[point_index] = int(center_index + 1)
    centers = [np.uint8(center.get_values()) for center in clusters_at_level.get_clusters()]
    segmented_data = np.array([centers[i - 1] for i in label_data])

    # Reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((height, width, channels))

    # Save segmented image with a unique filename
    filename = f"1-segmented_image_{level_idx}_cluster{len(centers)}.png"
    plt.imsave(filename, segmented_image)