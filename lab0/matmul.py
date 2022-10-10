from numba import cuda
import time
import math
import numpy as np


def matmul_CPU(a, b, c, size):
    for i in range(size):
        for j in range(size):
            rez = 0

            for z in range(size):
                rez += a[i, z] * b[z, j]

            c[i, j] = rez


@cuda.jit
def matmul_GPU(a, b, c, size):
    for i in range(size):
        for j in range(size):
            rez = 0

            for z in range(size):
                rez += a[i, z] * b[z, j]

            c[i, j] = rez


def perform_matmul(matrix_size):
    matrix1 = np.random.randint(0, 10, (matrix_size, matrix_size))
    matrix2 = np.random.randint(0, 10, (matrix_size, matrix_size))

    matrix1_cuda = cuda.to_device(matrix1)
    matrix2_cuda = cuda.to_device(matrix2)

    threads_in_block = (32, 32)
    grid_in_block_x = int(math.ceil(matrix1.shape[0] / threads_in_block[0]))
    grid_in_block_y = int(math.ceil(matrix2.shape[1] / threads_in_block[1]))

    blocks_in_grid = (grid_in_block_x, grid_in_block_y)

    print("Grid dim: ", blocks_in_grid)
    print("Block dim: ", threads_in_block)

    # CPU calculation
    start_time = time.time()
    cpu_matmul_result = np.zeros((matrix_size, matrix_size), dtype=int)

    matmul_CPU(matrix1, matrix2, cpu_matmul_result, matrix_size)

    time_cpu = time.time() - start_time

    print("CPU calculation time: ", time_cpu)

    # GPU (CUDA) calculation
    start_time = time.time()
    gpu_matmul_result = cuda.device_array((len(matrix1), len(matrix2)))

    matmul_GPU[blocks_in_grid, threads_in_block](matrix1_cuda, matrix2_cuda, gpu_matmul_result, matrix_size)

    time_gpu = time.time() - start_time

    print("GPU calculation time: ", time_gpu)

    print("Boost: ", time_cpu / time_gpu)

    print("Matriсes from CPU and GPU are equal: ", np.allclose(cpu_matmul_result, gpu_matmul_result))


if __name__ == "__main__":
    matrix_size = 2000
    print("Size = ", matrix_size, "x", matrix_size)
    perform_matmul(matrix_size)
