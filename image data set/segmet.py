import copy
import time
import tracemalloc
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import cv2
import requests
from PIL import Image
from io import BytesIO


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


def segment_image(image_path, k, m):
    # Read image
    image = cv2.imread(image_path)
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_width = 50
    new_height = 50

    # Resize the image
    resized_image = cv2.resize(image_rgb, (new_width, new_height))
    # Flatten image to use as data
    data = resized_image.reshape((-1, 3))

    # Perform hierarchical clustering
    level_data, result = k_level(data, k, m)

    # Create segmented image
    segmented_image = np.zeros_like(data)
    for cluster in result.get_clusters():
        for idx in cluster.get_indexes():
            print(idx, cluster.get_values().astype(int))
            segmented_image[idx] = cluster.get_values().astype(int)

    # Reshape segmented image to original shape
    segmented_image = segmented_image.reshape(resized_image.shape)

    return segmented_image


# Example usage
image_path = "download (2).jpg"  # Provide the path to your image
k = 6  # Desired level of clustering
m = 1  # Number of clusters to merge simultaneously

# Segment image
segmented_image = segment_image(image_path, k, m)

# Display original and segmented images
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(cv2.imread(image_path))
axs[0].set_title("Original Image")
axs[0].axis('off')
axs[1].imshow(segmented_image)
axs[1].set_title("Segmented Image")
axs[1].axis('off')
plt.show()
