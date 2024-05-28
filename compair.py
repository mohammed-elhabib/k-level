# Import required libraries
import tracemalloc
import time
import numpy as np
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
from Klevel import k_level
# Define functions for clustering algorithms
def upgma(data):
    return linkage(data, method='average')

def clink(data):
    return linkage(data, method='single')

def upgmc(data):
    return linkage(data, method='complete')

def slink(data):
    return linkage(data, method='ward')

def median_link(data):
    # Implement Median-link clustering algorithm
    pass  # Implement your Median-link algorithm here

# Modify k_level function to accept clustering algorithm as an argument
def exec(data,method):
    if method == 'UPGMA':
        return upgma(data)
    elif method == 'CLINK':
        return clink(data)
    elif method == 'UPGMC':
        return upgmc(data)
    elif method == 'SLINK':
        return slink(data)
    elif method == 'Median-link':
        return median_link(data)
    else:
        raise ValueError("Invalid method provided")

# Function to measure execution time and memory size
def measure_time_memory(data, k, m, method):
    tracemalloc.start()  # Start memory tracing
    start_time = time.time()  # Record start time
    exec(data, method)  # Execute clustering
    execution_time = time.time() - start_time  # Calculate execution time
    current, peak = tracemalloc.get_traced_memory()  # Get memory usage
    memory_size = peak / 10**6  # Convert bytes to megabytes
    tracemalloc.stop()  # Stop memory tracing
    return execution_time, memory_size
# Function to measure execution time and memory size
def measure_time_memory_k_level(data, k, m):
    tracemalloc.start()  # Start memory tracing
    start_time = time.time()  # Record start time
    k_level(data,k,m) # Execute clustering
    execution_time = time.time() - start_time  # Calculate execution time
    current, peak = tracemalloc.get_traced_memory()  # Get memory usage
    memory_size = peak / 10**6  # Convert bytes to megabytes
    tracemalloc.stop()  # Stop memory tracing
    return execution_time, memory_size
# Define sample data and parameters
data = np.random.rand(20000, 4)  # Sample data
k = 15  # Number of levels
m = 10  # Number of clusters to merge
methods = ['UPGMA', 'CLINK', 'UPGMC', 'SLINK', 'Median-link']  # List of clustering algorithms

# Measure execution time and memory size for each method
results = {}
for method in methods:
    execution_time, memory_size = measure_time_memory(data, k, m, method)
    results[method] = {'time': execution_time, 'memory': memory_size}
execution_time, memory_size = measure_time_memory_k_level(data, k,m )
results["k-level"] = {'time': execution_time, 'memory': memory_size}

# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), [result['time'] for result in results.values()], color='blue', alpha=0.7, label='Execution Time')
plt.xlabel('Clustering Algorithm')
plt.ylabel('Time (s)')
plt.title('Execution Time Comparison for Clustering Algorithms')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(results.keys(), [result['memory'] for result in results.values()], color='green', alpha=0.7, label='Memory Size (MB)')
plt.xlabel('Clustering Algorithm')
plt.ylabel('Memory Size (MB)')
plt.title('Memory Size Comparison for Clustering Algorithms')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
