import time
import tracemalloc

import numpy as np
from sklearn.datasets import make_blobs

from Klevel import plt_2d_origen, plt_2d, evaluate_clustering, k_level

n_samples = 4000
n_features = 2
n_clusters = 6

cov_matrix = np.array([[0.1, 0.8], [2.5, 0.3]])

data_anisotropic, labels_anisotropic = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, cluster_std=[1.0, 1.5, 1.5, 1.5, 1.5, 0.5], random_state=42)



k = 100  # Desired level of clustering
m = 1  # Number of clusters to merge simultaneously
time_start = time.time()
# starting the monitoring
tracemalloc.start()

# function call


level_data, result = k_level(data_anisotropic, k, m)
evaluate_clustering(data_anisotropic, level_data)

# displaying the memory
print(tracemalloc.get_traced_memory())

# stopping the library
tracemalloc.stop()
time_end = time.time()
size=[-12,15]
plt_2d(level_data,size,"anisotropic")
#plt_2d_origen(level_data,data_anisotropic)

print("time ", time_end - time_start)