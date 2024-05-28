import time
import tracemalloc

from sklearn.datasets import make_gaussian_quantiles

from Klevel import k_level, plt_2d, evaluate_clustering

n_samples = 6000
n_features = 2

data_gaussian, labels_gaussian = make_gaussian_quantiles(n_samples=n_samples, n_features=n_features, n_classes=3)
k = 100  # Desired level of clustering
m = 1  # Number of clusters to merge simultaneously
time_start = time.time()
# starting the monitoring
tracemalloc.start()

# function call


level_data, result = k_level(data_gaussian, k, m)
evaluate_clustering(data_gaussian, level_data)

# displaying the memory
print(tracemalloc.get_traced_memory())

# stopping the library
tracemalloc.stop()
time_end = time.time()
size=[-2,2]

plt_2d(level_data, size,"gaussian")

print("time ", time_end - time_start)
