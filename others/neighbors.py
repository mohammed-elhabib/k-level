import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Generate synthetic data using make_blobs

n_samples = 6000

t = np.linspace(0, 2 * np.pi, n_samples)
x = t * np.cos(t)
y = t * np.sin(t)

data = np.column_stack((x, y))


# Apply KMeans to obtain cluster centers
kmeans = KMeans(n_clusters=3)
cluster_centers = kmeans.fit_predict(data)



# Evaluate the clustering-like results using internal validation metrics
silhouette = silhouette_score(data, cluster_centers)
davies_bouldin = davies_bouldin_score(data, cluster_centers)
calinski_harabasz = calinski_harabasz_score(data, cluster_centers)

print("KNN-Based Clustering Evaluation:")
print("Silhouette Score:", silhouette)
print("Davies-Bouldin Index:", davies_bouldin)
print("Calinski-Harabasz Index:", calinski_harabasz)
print("number cluster:", len(np.unique(cluster_centers)))
