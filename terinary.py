# Import necessary libraries
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import dask.array as da
from dask.distributed import Client, LocalCluster

# Define LUTs for ternary operations
TERNARY_OR_3_INPUT_LUT = {
    (-1, -1, -1): -1, (-1, -1, 0): 0, (-1, -1, 1): 1,
    (-1, 0, -1): 0,  (-1, 0, 0): 0,  (-1, 0, 1): 1,
    (-1, 1, -1): 1,  (-1, 1, 0): 1,  (-1, 1, 1): 1,
    (0, -1, -1): 0,  (0, -1, 0): 0,  (0, -1, 1): 1,
    (0, 0, -1): 0,   (0, 0, 0): 0,   (0, 0, 1): 1,
    (0, 1, -1): 1,   (0, 1, 0): 1,   (0, 1, 1): 1,
    (1, -1, -1): 1,  (1, -1, 0): 1,  (1, -1, 1): 1,
    (1, 0, -1): 1,   (1, 0, 0): 1,   (1, 0, 1): 1,
    (1, 1, -1): 1,   (1, 1, 0): 1,   (1, 1, 1): 1,
}

TERNARY_ADD_3_INPUT_LUT = {
    (a, b, c): max(-1, min(1, a + b + c))  # Ternary addition clamped to [-1, 1]
    for a in [-1, 0, 1] for b in [-1, 0, 1] for c in [-1, 0, 1]
}

# LUT-based ternary operations
def ternary_or_3_input_lut(a, b, c):
    return TERNARY_OR_3_INPUT_LUT[(a, b, c)]

def ternary_add_3_input_lut(a, b, c):
    return TERNARY_ADD_3_INPUT_LUT[(a, b, c)]

# Parallel processing function
def parallel_lut_operation(operation, inputs):
    """
    Parallelizes a given LUT-based ternary operation over a large dataset.
    """
    with Pool(cpu_count()) as pool:
        results = pool.starmap(operation, inputs)
    return results

# Distributed processing function
def distributed_lut_operation(operation, inputs):
    """
    Distributes a given LUT-based ternary operation over a large dataset using Dask.
    """
    # Convert inputs to a NumPy array
    inputs_np = np.array(inputs, dtype=int)

    # Convert inputs to a Dask array
    inputs_da = da.from_array(inputs_np, chunks=(len(inputs) // cpu_count(), 3))

    # Map the operation across the dataset
    def apply_operation(block):
        # Ensure the block is 2D and return a 1D result
        return np.array([operation(*row) for row in block])

    results = inputs_da.map_blocks(apply_operation, dtype=int, drop_axis=1)

    # Compute the results
    return results.compute()

# Main function for benchmarking with distributed processing
if __name__ == "__main__":
    # Initialize Dask cluster
    cluster = LocalCluster()
    client = Client(cluster)

    print("Dask cluster initialized")

    # Test with 1,000,000 inputs
    print("Testing with 1,000,000 inputs using distributed processing...")
    test_inputs_1m = [
        (np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1]))
        for _ in range(10000000)
    ]

    start_time = time.time()
    distributed_or_results = distributed_lut_operation(ternary_or_3_input_lut, test_inputs_1m)
    distributed_or_time = time.time() - start_time

    start_time = time.time()
    distributed_add_results = distributed_lut_operation(ternary_add_3_input_lut, test_inputs_1m)
    distributed_add_time = time.time() - start_time

    print({
        "Distributed Time OR (1M inputs)": distributed_or_time,
        "Distributed Time ADD (1M inputs)": distributed_add_time
    })

    client.close()

# # Main function for benchmarking
# if __name__ == "__main__":
#     print("Testing with 10,000 inputs...")
#     # Generate a dataset for benchmarking
#     test_inputs = [
#         (np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1]))
#         for _ in range(10000)
#     ]

#     # Benchmark parallelized LUT-based OR operation
#     start_time = time.time()
#     parallel_or_results = parallel_lut_operation(ternary_or_3_input_lut, test_inputs)
#     parallel_or_time = time.time() - start_time

#     # Benchmark parallelized LUT-based ADD operation
#     start_time = time.time()
#     parallel_add_results = parallel_lut_operation(ternary_add_3_input_lut, test_inputs)
#     parallel_add_time = time.time() - start_time

#     # Output benchmark results
#     print({
#         "Parallel Time OR": parallel_or_time,
#         "Parallel Time ADD": parallel_add_time
#     })

#     # Test with 100,000 inputs
#     print("Testing with 100,000 inputs...")
#     test_inputs_100k = [
#         (np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1]))
#         for _ in range(100000)
#     ]

#     start_time = time.time()
#     parallel_or_results_100k = parallel_lut_operation(ternary_or_3_input_lut, test_inputs_100k)
#     parallel_or_time_100k = time.time() - start_time

#     start_time = time.time()
#     parallel_add_results_100k = parallel_lut_operation(ternary_add_3_input_lut, test_inputs_100k)
#     parallel_add_time_100k = time.time() - start_time

#     print({
#         "Parallel Time OR (100k inputs)": parallel_or_time_100k,
#         "Parallel Time ADD (100k inputs)": parallel_add_time_100k
#     })

#     # Test with 1,000,000 inputs
#     print("Testing with 1,000,000 inputs...")
#     test_inputs_1m = [
#         (np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1]))
#         for _ in range(1000000)
#     ]

#     start_time = time.time()
#     parallel_or_results_1m = parallel_lut_operation(ternary_or_3_input_lut, test_inputs_1m)
#     parallel_or_time_1m = time.time() - start_time

#     start_time = time.time()
#     parallel_add_results_1m = parallel_lut_operation(ternary_add_3_input_lut, test_inputs_1m)
#     parallel_add_time_1m = time.time() - start_time

#     print({
#         "Parallel Time OR (1M inputs)": parallel_or_time_1m,
#         "Parallel Time ADD (1M inputs)": parallel_add_time_1m
#     })
