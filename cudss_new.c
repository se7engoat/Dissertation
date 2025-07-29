#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudss.h>
#include "mmio.h"
#include <nvml.h>

// ======================== Macros & Definitions ========================
#define CUDA_CHECK(err) \
    do { \
        if (err != cudaSuccess) { \
            printf("[CUDA ERROR] %s (code: %d)\n", cudaGetErrorString(err), err); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUDSS_CHECK(err) \
    do { \
        if (err != CUDSS_STATUS_SUCCESS) { \
            printf("[cuDSS ERROR] Code: %d\n", err); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

typedef struct {
    size_t host_memory;   // Host memory allocated (bytes)
    size_t device_memory; // Device memory allocated (bytes)
    int power_usage_mW;   // GPU power usage (milliwatts)
} benchmark_stats_t;

// ======================== Matrix I/O ========================
void read_mtx_to_csr(const char* filename, int** row_ptr, int** col_idx, 
                     double** values, int* n_rows, int* n_cols, int* nnz, 
                     benchmark_stats_t* stats) {
    FILE* f = fopen(filename, "r");
    MM_typecode matcode;
    int ret_code;
    
    if ((ret_code = mm_read_banner(f, &matcode)) != 0) {
        printf("[ERROR] Could not read Matrix Market banner (code: %d)\n", ret_code);
        exit(EXIT_FAILURE);
    }

    if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode)) {
        printf("[ERROR] Only sparse matrices are supported\n");
        exit(EXIT_FAILURE);
    }

    // Read matrix dimensions
    int M, N, nz;
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0) {
        printf("[ERROR] Failed to read matrix size (code: %d)\n", ret_code);
        exit(EXIT_FAILURE);
    }
    printf("\n[MATRIX] %d x %d, Non-zeros: %d\n", M, N, nz);

    // Allocate COO temporary storage
    int* I = (int*)malloc(nz * sizeof(int));
    int* J = (int*)malloc(nz * sizeof(int));
    double* val = (double*)malloc(nz * sizeof(double));
    stats->host_memory += nz * (2 * sizeof(int) + sizeof(double));

    printf("[MEMORY] Allocated COO temporary storage: %.2f MB\n", 
           nz * (2 * sizeof(int) + sizeof(double)) / (1024.0 * 1024.0));

    // Read COO data
    for (int i = 0; i < nz; i++) {
        fscanf(f, "%d %d %lf\n", &I[i], &J[i], &val[i]);
        I[i]--; J[i]--; // Convert to 0-based indexing
    }
    fclose(f);

    // Convert COO to CSR
    *row_ptr = (int*)calloc(M + 1, sizeof(int));
    *col_idx = (int*)malloc(nz * sizeof(int));
    *values = (double*)malloc(nz * sizeof(double));
    stats->host_memory += (M + 1) * sizeof(int) + nz * (sizeof(int) + sizeof(double));

    printf("[MEMORY] Allocated CSR storage: %.2f MB\n", 
           ((M + 1) * sizeof(int) + nz * (sizeof(int) + sizeof(double))) / (1024.0 * 1024.0));

    // Count non-zeros per row
    for (int i = 0; i < nz; i++) (*row_ptr)[I[i] + 1]++;

    // Cumulative sum
    for (int i = 0; i < M; i++) (*row_ptr)[i + 1] += (*row_ptr)[i];

    // Fill CSR
    int* row_count = (int*)calloc(M, sizeof(int));
    stats->host_memory += M * sizeof(int);
    for (int i = 0; i < nz; i++) {
        int row = I[i];
        int dest = (*row_ptr)[row] + row_count[row];
        (*col_idx)[dest] = J[i];
        (*values)[dest] = val[i];
        row_count[row]++;
    }

    // Free temporary storage
    free(I); free(J); free(val); free(row_count);
    stats->host_memory -= nz * (2 * sizeof(int) + sizeof(double)) + M * sizeof(int);
    printf("[MEMORY] Freed COO temporary storage\n");

    *n_rows = M;
    *n_cols = N;
    *nnz = nz;
}

// ======================== GPU Utilities ========================
void get_gpu_stats(benchmark_stats_t* stats) {
    nvmlReturn_t result;
    nvmlDevice_t device;
    unsigned int power_mW;
    nvmlMemory_t memory;

    result = nvmlInit();
    if (NVML_SUCCESS != result) {
        printf("[NVML WARNING] Initialization failed: %s\n", nvmlErrorString(result));
        return;
    }

    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (NVML_SUCCESS == result) {
        result = nvmlDeviceGetPowerUsage(device, &power_mW);
        if (NVML_SUCCESS == result) stats->power_usage_mW = power_mW;

        result = nvmlDeviceGetMemoryInfo(device, &memory);
        if (NVML_SUCCESS == result) {
            stats->device_memory = memory.used;
            printf("[GPU] Memory Usage: %.2f MB | Power: %d mW\n", 
                   memory.used / (1024.0 * 1024.0), power_mW);
        }
    }
    nvmlShutdown();
}

// ======================== Benchmark Core ========================
void benchmark_cudss(const char* mtx_path, int n_runs) {
    printf("\n============= cuDSS Benchmark =============\n");
    printf("[CONFIG] Matrix: %s | Runs: %d\n", mtx_path, n_runs);
    
    benchmark_stats_t stats = {0};
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Matrix Loading ---
    printf("\n[PHASE] Loading matrix...\n");
    int *row_ptr, *col_idx, *d_row_ptr, *d_col_idx;
    double *values, *d_values, *d_b, *d_x;
    int n_rows, n_cols, nnz;

    read_mtx_to_csr(mtx_path, &row_ptr, &col_idx, &values, &n_rows, &n_cols, &nnz, &stats);

    // --- GPU Allocation ---
    printf("\n[PHASE] Allocating GPU memory...\n");
    CUDA_CHECK(cudaMalloc((void**)&d_row_ptr, (n_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_values, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, n_rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_x, n_rows * sizeof(double)));
    stats.device_memory = (n_rows + 1 + nnz) * sizeof(int) + 
                         (nnz + n_rows) * sizeof(double);

    printf("[MEMORY] GPU allocation: %.2f MB\n", stats.device_memory / (1024.0 * 1024.0));

    // --- Data Transfer ---
    printf("\n[PHASE] Transferring data to GPU...\n");
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr, (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values, nnz * sizeof(double), cudaMemcpyHostToDevice));

    // Generate RHS
    double* b = (double*)malloc(n_rows * sizeof(double));
    stats.host_memory += n_rows * sizeof(double);
    for (int i = 0; i < n_rows; i++) b[i] = (double)rand() / RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d_b, b, n_rows * sizeof(double), cudaMemcpyHostToDevice));

    // --- cuDSS Setup ---
    printf("\n[PHASE] Initializing cuDSS...\n");
    cudssHandle_t handle;
    cudssConfig_t config;
    cudssData_t data;
    CUDSS_CHECK(cudssCreate(&handle));
    CUDSS_CHECK(cudssConfigCreate(&config));
    CUDSS_CHECK(cudssDataCreate(handle, &data));

    // Enable hybrid mode (CPU+GPU)
    int hybrid_mode = 1;
    cudssConfigSet(handle, CUDSS_CONFIG_HYBRID_MODE, &hybrid_mode, sizeof(int));

    // --- Matrix Descriptor ---
    cudssMatrix_t A;
    CUDSS_CHECK(cudssMatrixCreateCsr(
        &A, n_rows, n_cols, nnz,
        d_row_ptr, NULL, d_col_idx, d_values,
        CUDA_R_32I, CUDA_R_64F,
        CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO
    ));

    // --- Vector Descriptors ---
    cudssMatrix_t b_mat, x_mat;
    CUDSS_CHECK(cudssMatrixCreateDn(&b_mat, n_rows, 1, n_rows, d_b, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&x_mat, n_rows, 1, n_rows, d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

    // --- Warmup ---
    printf("\n[PHASE] Warmup...\n");
    float analysis_ms, factorization_ms, solve_ms;
    
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, A, x_mat, b_mat));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&analysis_ms, start, stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, A, x_mat, b_mat));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&factorization_ms, start, stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A, x_mat, b_mat));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&solve_ms, start, stop));

    printf("[WARMUP] Analysis: %.2f ms | Factorization: %.2f ms | Solve: %.2f ms\n",
           analysis_ms, factorization_ms, solve_ms);

    // --- Benchmark ---
    printf("\n[PHASE] Benchmarking solve phase (%d runs)...\n", n_runs);
    float total_solve = 0, min_solve = 1e9, max_solve = 0;
    get_gpu_stats(&stats); // Initial GPU stats

    for (int i = 0; i < n_runs; i++) {
        CUDA_CHECK(cudaEventRecord(start, 0));
        CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A, x_mat, b_mat));
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&solve_ms, start, stop));

        total_solve += solve_ms;
        if (solve_ms < min_solve) min_solve = solve_ms;
        if (solve_ms > max_solve) max_solve = solve_ms;
        printf("[RUN %02d] Solve time: %.2f ms\n", i + 1, solve_ms);
    }

    // --- Results ---
    printf("\n============= Results =============\n");
    printf("[TIMING] Solve (avg/min/max): %.2f / %.2f / %.2f ms\n",
           total_solve / n_runs, min_solve, max_solve);
    printf("[TIMING] Total (analysis + factorization + solve): %.2f ms\n",
           analysis_ms + factorization_ms + total_solve / n_runs);
    get_gpu_stats(&stats);
    printf("[MEMORY] Host: %.2f MB | Device: %.2f MB\n",
           stats.host_memory / (1024.0 * 1024.0),
           stats.device_memory / (1024.0 * 1024.0));

    // --- Cleanup ---
    printf("\n[PHASE] Cleaning up...\n");
    CUDSS_CHECK(cudssMatrixDestroy(A));
    CUDSS_CHECK(cudssMatrixDestroy(b_mat));
    CUDSS_CHECK(cudssMatrixDestroy(x_mat));
    CUDSS_CHECK(cudssDataDestroy(handle, data));
    CUDSS_CHECK(cudssConfigDestroy(config));
    CUDSS_CHECK(cudssDestroy(handle));

    free(row_ptr); free(col_idx); free(values); free(b);
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));

    printf("[MEMORY] All resources freed\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <matrix.mtx> [runs=10]\n", argv[0]);
        return EXIT_FAILURE;
    }
    int runs = (argc > 2) ? atoi(argv[2]) : 10;
    benchmark_cudss(argv[1], runs);
    return EXIT_SUCCESS;
}