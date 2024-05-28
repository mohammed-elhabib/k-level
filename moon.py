import time
import tracemalloc

from sklearn.datasets import make_moons

from Klevel import plt_2d, evaluate_clustering, k_level, plt_2d_origen

n_samples = 3000

data_moons, labels_moons = make_moons(n_samples=n_samples, noise=0.1)
k = 100  # Desired level of clustering
m = 1  # Number of clusters to merge simultaneously
time_start = time.time()
# starting the monitoring
tracemalloc.start()

# function call


level_data, result = k_level(data_moons, k, m)
evaluate_clustering(data_moons, level_data)

# displaying the memory
print(tracemalloc.get_traced_memory())

# stopping the library
tracemalloc.stop()
time_end = time.time()
size=[-2,2.5]
plt_2d(level_data, size,"moon")

print("time ", time_end - time_start)
