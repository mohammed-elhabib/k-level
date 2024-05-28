import numpy as np
from sklearn.datasets import make_blobs

# 2. Moon-Shaped Clusters
data2, labels2 = make_moons(n_samples=300, noise=0.15, random_state=0)
data1, labels1 = make_circles(n_samples=200, noise=0.05, factor=0.3, random_state=0)
data3, labels3 = make_blobs(n_samples=2000, centers=3)
# Set random seed for reproducibility
np.random.seed(42)

# Number of clusters and data points per cluster
num_clusters = 3
points_per_cluster = 100

# Generate synthetic data using K-Means clustering
data, _ = make_blobs(n_samples=1000, centers=4, n_features=2, cluster_std=0.51)
# Example usage
np.random.seed(0)
# data = np.random.rand(30000, 2)
# print(data)
k = 100  # Desired level of clustering
m = 1  # Number of clusters to merge simultaneously
time_start = time.time()
# starting the monitoring
tracemalloc.start()

# function call


level_data, result = k_level(data1, k, m)
evaluate_clustering(data1, level_data)

# displaying the memory
print(tracemalloc.get_traced_memory())

# stopping the library
tracemalloc.stop()
time_end = time.time()
plt_2d(level_data, data1)

print("time ", time_end - time_start)