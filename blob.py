import time
import tracemalloc

from sklearn.datasets import make_blobs

from Klevel import k_level, evaluate_clustering, plt_2d

n_samples = 600
n_features = 2
n_clusters = 3

data_blob, labels_blob = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters)

k = 100  # Desired level of clustering
m = 1  # Number of clusters to merge simultaneously
time_start = time.time()
# starting the monitoring
tracemalloc.start()

# function call


level_data, result = k_level(data_blob, k, m)
evaluate_clustering(data_blob, level_data)

# displaying the memory
print(tracemalloc.get_traced_memory())
size=[-12,15]

# stopping the library
tracemalloc.stop()
time_end = time.time()
plt_2d(level_data, size, "blob")
print("time ", time_end - time_start)
