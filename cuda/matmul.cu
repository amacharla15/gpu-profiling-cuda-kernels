#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            printf("CUDA error at %s:%d: %s\n",                          \
                   __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(1);                                                     \
        }                                                                \
    } while (0)

#define CUBLAS_CHECK(call)                                               \
    do {                                                                 \
        cublasStatus_t status = call;                                    \
        if (status != CUBLAS_STATUS_SUCCESS) {                           \
            printf("cuBLAS error at %s:%d: %d\n",                        \
                   __FILE__, __LINE__, (int)status);                     \
            exit(1);                                                     \
        }                                                                \
    } while (0)

#define TILE_SIZE 32

__global__ void matmul_naive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_tiled(float *A, float *B, float *C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        if (row < N && a_col < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < N && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_cpu(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

typedef void (*kernel_fn)(float*, float*, float*, int);

float benchmark_kernel(kernel_fn kernel, float *d_A, float *d_B, float *d_C,
                       int N, dim3 grid, dim3 block, int warmup, int runs) {
    for (int i = 0; i < warmup; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / runs;
}

float benchmark_cublas(float *d_A, float *d_B, float *d_C, int N,
                       int warmup, int runs) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    for (int i = 0; i < warmup; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUBLAS_CHECK(cublasDestroy(handle));
    return total_ms / runs;
}

int verify(float *A, float *B, int N, float tolerance) {
    int errors = 0;
    int total = N * N;
    for (int i = 0; i < total; i++) {
        float diff = fabs(A[i] - B[i]);
        if (diff > tolerance) {
            errors++;
            if (errors <= 3) {
                printf("  MISMATCH at %d: %.6f vs %.6f (diff=%.6f)\n",
                       i, A[i], B[i], diff);
            }
        }
    }
    return errors;
}

int main() {
    int N = 1024;
    size_t size = (size_t)N * (size_t)N * sizeof(float);

    printf("============================================================\n");
    printf("MATRIX MULTIPLICATION — Naive vs Tiled vs cuBLAS\n");
    printf("============================================================\n");
    printf("Matrix size: %d x %d (%.1f MB per matrix)\n", N, N, size / 1e6);
    printf("TILE_SIZE: %d\n", TILE_SIZE);
    printf("FLOPS per multiply: %lld (2*N^3)\n\n",
           (long long)2 * N * N * N);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_naive = (float *)malloc(size);
    float *h_C_tiled = (float *)malloc(size);
    float *h_C_cublas = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C_naive == NULL ||
        h_C_tiled == NULL || h_C_cublas == NULL || h_C_cpu == NULL) {
        printf("Host malloc failed\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A;
    float *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);

    printf("Launch config: grid(%d,%d) x block(%d,%d)\n\n",
           grid.x, grid.y, block.x, block.y);

    printf("Running CPU baseline...\n");
    clock_t cpu_start = clock();
    matmul_cpu(h_A, h_B, h_C_cpu, N);
    clock_t cpu_end = clock();
    float cpu_ms = (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0f;
    printf("  CPU: %.2f ms\n\n", cpu_ms);

    printf("Running naive GPU kernel...\n");
    float naive_ms = benchmark_kernel(matmul_naive, d_A, d_B, d_C,
                                      N, grid, block, 3, 10);
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, size, cudaMemcpyDeviceToHost));
    int naive_errors = verify(h_C_naive, h_C_cpu, N, 1e-2f);
    printf("  Naive GPU: %.3f ms (%s)\n\n",
           naive_ms, naive_errors == 0 ? "PASS" : "FAIL");

    printf("Running tiled GPU kernel...\n");
    float tiled_ms = benchmark_kernel(matmul_tiled, d_A, d_B, d_C,
                                      N, grid, block, 3, 10);
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, size, cudaMemcpyDeviceToHost));
    int tiled_errors = verify(h_C_tiled, h_C_cpu, N, 1e-2f);
    printf("  Tiled GPU: %.3f ms (%s)\n\n",
           tiled_ms, tiled_errors == 0 ? "PASS" : "FAIL");

    printf("Running cuBLAS...\n");
    float cublas_ms = benchmark_cublas(d_A, d_B, d_C, N, 3, 10);
    CUDA_CHECK(cudaMemcpy(h_C_cublas, d_C, size, cudaMemcpyDeviceToHost));
    int cublas_errors = verify(h_C_cublas, h_C_cpu, N, 1e-2f);
    printf("  cuBLAS: %.3f ms (%s)\n\n",
           cublas_ms, cublas_errors == 0 ? "PASS" : "FAIL");

    float naive_gflops = (2.0 * N * N * N) / (naive_ms / 1000.0) / 1e9;
    float tiled_gflops = (2.0 * N * N * N) / (tiled_ms / 1000.0) / 1e9;
    float cublas_gflops = (2.0 * N * N * N) / (cublas_ms / 1000.0) / 1e9;

    printf("============================================================\n");
    printf("RESULTS SUMMARY (%dx%d matrix multiply)\n", N, N);
    printf("============================================================\n");
    printf("  %-15s %10s %10s %12s %8s\n",
           "Version", "Time(ms)", "GFLOPS", "vs Naive", "Status");
    printf("  %-15s %10.2f %10s %12s %8s\n",
           "CPU", cpu_ms, "-", "-",
           "baseline");
    printf("  %-15s %10.3f %10.1f %12s %8s\n",
           "GPU Naive", naive_ms, naive_gflops, "1.0x",
           naive_errors == 0 ? "PASS" : "FAIL");
    printf("  %-15s %10.3f %10.1f %11.1fx %8s\n",
           "GPU Tiled", tiled_ms, tiled_gflops, naive_ms / tiled_ms,
           tiled_errors == 0 ? "PASS" : "FAIL");
    printf("  %-15s %10.3f %10.1f %11.1fx %8s\n",
           "cuBLAS", cublas_ms, cublas_gflops, naive_ms / cublas_ms,
           cublas_errors == 0 ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);
    free(h_C_cublas);
    free(h_C_cpu);

    return 0;
}