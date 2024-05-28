import time
import tracemalloc

import numpy as np

from Klevel import plt_2d, plt_2d_origen, evaluate_clustering, k_level

n_samples = 6000

t = np.linspace(0, 2 * np.pi, n_samples)
x = t * np.cos(t)
y = t * np.sin(t)

data_spiral = np.column_stack((x, y))

k = 100  # Desired level of clustering
m = 1  # Number of clusters to merge simultaneously
time_start = time.time()
# starting the monitoring
tracemalloc.start()

# function call


level_data, result = k_level(data_spiral, k, m)
evaluate_clustering(data_spiral, level_data)

# displaying the memory
print(tracemalloc.get_traced_memory())

# stopping the library
tracemalloc.stop()
time_end = time.time()
size=[-5,5]

plt_2d(level_data,size, "spiral")


print("time ", time_end - time_start)
