import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score

# Generate synthetic data (moons)

n_samples = 3000

data, labels_moons = make_moons(n_samples=n_samples, noise=0.1)




# Apply DBSCAN algorithm
dbscan = DBSCAN(eps=0.3, min_samples=5)
predicted_labels = dbscan.fit_predict(data)
print(f"DBSCAN Clustering Evaluation:{np.unique(predicted_labels)}")

# Evaluate the clustering results using internal validation metrics
silhouette = silhouette_score(data, predicted_labels)
davies_bouldin = davies_bouldin_score(data, predicted_labels)
calinski_harabasz = calinski_harabasz_score(data, predicted_labels)

print("Silhouette Score:", silhouette)
print("Davies-Bouldin Index:", davies_bouldin)
print("Calinski-Harabasz Index:", calinski_harabasz)
