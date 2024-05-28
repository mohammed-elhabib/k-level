import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import euclidean

n_samples = 20000

t = np.linspace(0, 2 * np.pi, n_samples)
x = t * np.cos(t)
y = t * np.sin(t)

data = np.column_stack((x, y))
# Number of clusters to evaluate
num_clusters = 4000

# List of linkage methods to evaluate
linkage_methods = ['ward', 'complete', 'average', 'single']
for linkage_method in linkage_methods:
    # Create an instance of AgglomerativeClustering with the current linkage method
    clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_method)

    # Fit the clustering model to the data
    labels = clustering.fit_predict(data)

    silhouette_per_sample = silhouette_samples(data, labels)

    # Print evaluation metrics for the current linkage method
    print(f"Linkage: {linkage_method}")
    print("Silhouette Score:", silhouette_score(data, labels))
    print("Davies-Bouldin Index:", davies_bouldin_score(data, labels))
    print("Calinski-Harabasz Index:", calinski_harabasz_score(data, labels))
    print("Average Silhouette Per Sample:", np.mean(silhouette_per_sample))
    print("number cluster:", len(np.unique(labels)))
    print()

