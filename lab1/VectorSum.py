from numba import cuda, int32, int64
import time
import numpy as np

@cuda.jit
def GPU_sum(vector, result, threads_per_block):
    buffer = cuda.shared.array(threads_per_block, dtype=int64)

    idx = cuda.threadIdx.x + cuda.blockIdx.x * threads_per_block

    buffer[cuda.threadIdx.x] = 0

    if idx < vector.shape[0]:
        buffer[cuda.threadIdx.x] = vector[idx]

        cuda.syncthreads()
        if cuda.threadIdx.x == 0:
            sum = 0
            for i in range(threads_per_block):
                sum += buffer[i]
            cuda.atomic.add(result, 0, sum)


def CPU_sum(vector):
    return np.sum(vector)


def perform_vector_sum(vector_size):
    vector = np.random.randint(-10, 10, vector_size)
    result = np.zeros(1, dtype=np.int32)

    print("Input vector: ", vector)

    print("CPU calculation:")

    start_time_CPU = time.time()
    CPU_result = CPU_sum(vector)
    result_time_CPU = time.time() - start_time_CPU

    print("Result CPU : ", CPU_result)
    print("Time CPU: ", result_time_CPU)

    print("_________________________________________________________")

    print("GPU calculation: ")

    threads_per_block = 16
    GPU_vector = cuda.to_device(vector)
    GPU_for_res = cuda.to_device(result)

    start_time_GPU = time.time()
    GPU_sum[threads_per_block, threads_per_block](GPU_vector, GPU_for_res, threads_per_block)
    result_time_GPU = time.time() - start_time_GPU

    result_GPU = GPU_for_res.copy_to_host()

    print("Result GPU : ", result_GPU)
    print("Time GPU: ", result_time_GPU)

if __name__ == "__main__":
    vector_size = 1000
    perform_vector_sum(vector_size)