import copy
import random

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import morphology
from scipy.ndimage import label


# Cluster and ClusterArray classes
class Cluster:
    def __init__(self, indexes, values):
        self.indexes = indexes
        self.values = np.array(values)
        self.selected = False

    def get_values(self):
        return self.values

    def get_indexes(self):
        return self.indexes

    def set_select(self, status):
        self.selected = status


class ClusterArray:
    def __init__(self, clusters):
        self.clusters = clusters
        self.random_index_cluster = None

    def get_cluster(self):
        self.random_index_cluster = random.randint(0, len(self.clusters) - 1)
        return self.clusters[self.random_index_cluster]

    def size(self):
        return len(self.clusters)

    def append(self, new_item):
        self.clusters.append(new_item)
        return self

    def append_first(self, new_item):
        self.clusters.insert(0, new_item)
        return self

    def delete_cluster(self):
        del self.clusters[self.random_index_cluster]
        return self

    def get_clusters(self):
        return self.clusters

    def get_selected_clusters(self, indexes):
        return ClusterArray([self.clusters[index] for index in indexes])

    def get_mean_cluster(self):
        cluster_values = [cluster.get_values() for cluster in self.clusters]
        cluster_indexes = sum([cluster.get_indexes() for cluster in self.clusters], [])
        return Cluster(cluster_indexes, np.mean(cluster_values, axis=0))

    def delete_clusters(self, indexes):
        for index in np.sort(indexes)[::-1]:
            del self.clusters[index]


def k_level(data, k, m):
    clusters = ClusterArray([Cluster(indexes=[index], values=values) for (index, values) in enumerate(data)])
    level_data = []
    for level in range(k):
        next_clusters = ClusterArray([])
        if clusters.size() == 1:
            return level_data, clusters

        while clusters.size() > 0:
            if clusters.size() == 1:
                next_clusters.append(clusters.get_cluster())
                clusters.delete_cluster()
                break
            cluster = clusters.get_cluster()
            clusters.delete_cluster()
            distances = [np.linalg.norm(cluster.get_values() - other_cluster.get_values()) for other_cluster in
                         clusters.get_clusters()]
            if next_clusters.size() > 0:
                distances_next = [np.linalg.norm(cluster.get_values() - other_cluster.get_values()) for other_cluster in
                                  next_clusters.get_clusters()]
                if np.min(distances) > np.min(distances_next):
                    min_index = [np.argmin(distances_next)]
                    next_clusters.append(
                        next_clusters.get_selected_clusters(min_index).append(cluster).get_mean_cluster())
                    next_clusters.delete_clusters(min_index)
                    continue
            indexes = np.argsort(distances)[:m]
            next_clusters.append(clusters.get_selected_clusters(indexes).append(cluster).get_mean_cluster())
            clusters.delete_clusters(indexes)
        clusters = next_clusters
        level_data.append(copy.deepcopy(next_clusters))
    return level_data, next_clusters


# Preprocessing (e.g., noise removal, normalization)
def preprocess_medical_image(image):
    # Implement preprocessing steps here
    preprocessed_image = image  # Placeholder, replace with actual preprocessing
    return preprocessed_image

# Feature extraction
def extract_features(image):
    # Implement feature extraction methods (e.g., intensity, texture, shape)
    features = image  # Placeholder, replace with actual feature extraction
    return features

# Hierarchical clustering (using k_level function)
def hierarchical_segmentation(data, k, m):
    # Implement hierarchical clustering using k_level function
    level_data, result = k_level(data, k, m)
    # Return segmentation result
    return result

# Post-processing
def postprocess_segmentation(segmented_image):
    # Implement post-processing steps (e.g., morphology operations, region growing)
    processed_image = segmented_image  # Placeholder, replace with actual post-processing
    return processed_image

# Visualization and Evaluation
def visualize_segmentation(original_image, segmented_image):
    # Visualize segmented regions overlaid on the original image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
#    plt.imshow(segmented_image, cmap='jet')  # Colormap can be adjusted based on the segmentation result
 #   plt.title('Segmented Image')
    plt.show()

def evaluate_segmentation(segmented_image, ground_truth):
    # Implement evaluation metrics (e.g., Dice coefficient, Jaccard index)
    evaluation_metrics = {}  # Placeholder, replace with actual evaluation
    return evaluation_metrics

# Load medical image (replace this with your actual medical image loading code)
image_path = "download (1).jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Preprocess medical image
preprocessed_image = preprocess_medical_image(image)

# Extract features
features = extract_features(preprocessed_image)

# Perform hierarchical segmentation
k = 3  # Adjust k and m values as needed
m = 1
segmented_image = hierarchical_segmentation(features, k, m)

# Post-process segmented image
processed_segmentation = postprocess_segmentation(segmented_image)

# Visualize segmentation
visualize_segmentation(preprocessed_image, processed_segmentation)

# Evaluate segmentation (if ground truth is available)
# evaluation_results = evaluate_segmentation(processed_segmentation, ground_truth)
