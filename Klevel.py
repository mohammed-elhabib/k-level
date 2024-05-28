import copy
import time
import tracemalloc
import random

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
import profile

# importing libraries
import os
import psutil
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# instantiation of decorator function
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
        #print(self.random_index_cluster, len(self.clusters))
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


def evaluate_clustering(data, level_data):
    for level_idx, clusters_at_level in enumerate(level_data):
        label_data = np.zeros((len(data),))

        for center_index, center in enumerate(clusters_at_level.get_clusters()):
            for point_index in center.get_indexes():
                label_data[point_index] = (center_index + 1)
        # Calculate internal evaluation metrics
        if len(np.unique(label_data)) > 1:  # Ensure at least 2 unique labels for silhouette_score
            silhouette = silhouette_score(data, label_data)
            davies_bouldin = davies_bouldin_score(data, label_data)
            calinski_harabasz = calinski_harabasz_score(data, label_data)

            # Print metrics
            print(f"Level {level_idx + 1} - Silhouette: {silhouette:.4f}, "
                  f"Davies-Bouldin: {davies_bouldin:.4f}, Calinski-Harabasz: {calinski_harabasz:.4f}, number of clustring : {clusters_at_level.size()}")
        else:
            print(f"Level {level_idx + 1} - Single cluster detected, skipping evaluation.")



def plt_2d(level_data,size,fileName):
    for level_idx, clusters_at_level in enumerate(level_data):
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))
        ax = axs
        # Set axis limits
        ax.set_xlim(size[0],size[1])
        ax.set_ylim(size[0],size[1])
        # for cluster in clusters_at_level.get_clusters():
        #     points = data[cluster.get_indexes()]
        #     ax.scatter(points[:, 0], points[:, 1])
        #     ax.add_patch(plt.Circle(cluster.get_values(), radius=0.25, fill=False))
        centers = np.array([center.get_values() for center in clusters_at_level.get_clusters()])
        ax.scatter(centers[:, 0], centers[:, 1])
        #ax.set_title(f"Level {level_idx + 1} cluster number {len(clusters_at_level.get_clusters())}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.savefig(f"{fileName}/level_{level_idx}.png", dpi=300)

    plt.show()


def plt_2d_origen(level_data, data):
    for level_idx, clusters_at_level in enumerate(level_data):
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))
        ax = axs
        # Set axis limits
        # ax.set_xlim(-2.5, 2.5)
        # ax.set_ylim(-2.5, 2.5)
        for cluster in clusters_at_level.get_clusters():
            points = data[cluster.get_indexes()]
            ax.scatter(points[:, 0], points[:, 1])
            ax.add_patch(plt.Circle(cluster.get_values(), radius=0.25, fill=False))

        ax.set_title(f"Level {level_idx + 1} cluster number {len(clusters_at_level.get_clusters())}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    plt.show()
    plt.savefig()

def plt_3d(level_data, data):
    for level_idx, clusters_at_level in enumerate(level_data):
        fig = plt.figure(figsize=(10, 6))  # Create a new figure for each level

        ax = fig.add_subplot(111, projection='3d')  # Add 3D projection

        # for cluster in clusters_at_level.get_clusters():
        #     points = data[cluster.get_indexes()]
        #     ax.scatter(points[:, 0], points[:, 1], points[:, 2])  # Corrected indexing here

        ax.set_title(f"Level {level_idx + 1} cluster number {len(clusters_at_level.get_clusters())}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # Add Z label for 3D plots

        plt.tight_layout()
        plt.show()  #


def plot_custom_dendrograms(level_data):
    for level_idx, clusters_at_level in enumerate(level_data):
        plt.figure(figsize=(10, 6))
        plt.title(f"Custom Dendrogram - Level {level_idx + 1}")
        plt.xlabel("Clusters")
        plt.ylabel("Distance")

        # Create a list to store the heights of each cluster's dendrogram line
        cluster_heights = [0] * len(clusters_at_level.get_clusters())

        # Iterate through each cluster and its associated height
        for cluster_index, cluster in enumerate(clusters_at_level.get_clusters()):
            cluster_height = cluster_heights[cluster_index]

            # Draw a vertical line representing the cluster
            plt.plot([cluster_index, cluster_index], [0, cluster_height], '-k')

            # Update the heights for the child clusters if available
            if isinstance(cluster, ClusterArray):
                child_cluster_indices = cluster.get_indexes()
                child_cluster_heights = [cluster_height + 1] * len(child_cluster_indices)
                for child_index, child_height in zip(child_cluster_indices, child_cluster_heights):
                    cluster_heights[child_index] = child_height

        plt.tight_layout()
        plt.show()