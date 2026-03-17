#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            printf("CUDA error at %s:%d: %s\n",                          \
                   __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(1);                                                     \
        }                                                                \
    } while (0)

__global__ void vector_add_kernel(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void vector_add_cpu(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 10000000;
    size_t size = (size_t)N * sizeof(float);

    printf("============================================================\n");
    printf("VECTOR ADDITION — First CUDA Program\n");
    printf("============================================================\n");
    printf("N = %d elements (%.1f MB per array)\n\n", N, size / 1e6);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);

    srand(42);
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    printf("Memory: %.1f MB on CPU + %.1f MB on GPU\n\n",
           4 * size / 1e6, 3 * size / 1e6);

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    printf("Data copied to GPU (%.1f MB over PCIe)\n\n", 2 * size / 1e6);

    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    printf("Launch: %d blocks x %d threads = %d total threads\n",
           num_blocks, threads_per_block, num_blocks * threads_per_block);
    printf("  (%d threads will be idle due to rounding)\n\n",
           num_blocks * threads_per_block - N);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    vector_add_kernel<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    clock_t cpu_start = clock();
    vector_add_cpu(h_A, h_B, h_C_cpu, N);
    clock_t cpu_end = clock();
    float cpu_ms = (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0f;

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - h_C_cpu[i]) > 1e-5f) {
            errors++;
            if (errors <= 3) {
                printf("MISMATCH at i=%d: GPU=%.6f, CPU=%.6f\n",
                       i, h_C[i], h_C_cpu[i]);
            }
        }
    }

    printf("RESULTS:\n");
    printf("  GPU time:  %.3f ms\n", gpu_ms);
    printf("  CPU time:  %.3f ms\n", cpu_ms);
    printf("  Speedup:   %.1fx\n", cpu_ms / gpu_ms);
    printf("  Errors:    %d / %d\n", errors, N);
    printf("  Status:    %s\n\n", errors == 0 ? "PASS" : "FAIL");

    printf("Sample (first 5 elements):\n");
    printf("  %10s %10s %10s %10s\n", "A[i]", "B[i]", "GPU", "CPU");
    for (int i = 0; i < 5; i++) {
        printf("  %10.6f %10.6f %10.6f %10.6f\n",
               h_A[i], h_B[i], h_C[i], h_C_cpu[i]);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);

    printf("\nDone! 10 million parallel adds on the GPU.\n");
    return 0;
}