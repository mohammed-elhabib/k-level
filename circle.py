import time
import tracemalloc

from sklearn.datasets import make_circles

from Klevel import evaluate_clustering, k_level, plt_2d

n_samples = 3000

data_circles, labels_circles = make_circles(n_samples=n_samples, noise=0.1)

k = 100  # Desired level of clustering
m = 1  # Number of clusters to merge simultaneously
time_start = time.time()
# starting the monitoring
tracemalloc.start()

# function call


level_data, result = k_level(data_circles, k, m)
evaluate_clustering(data_circles, level_data)

# displaying the memory
print(tracemalloc.get_traced_memory())
size=[-2,2]

# stopping the library
tracemalloc.stop()
time_end = time.time()
plt_2d(level_data, size,"circles")

print("time ", time_end - time_start)