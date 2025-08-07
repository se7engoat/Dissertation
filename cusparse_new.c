#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusparse.h>
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

#define CUSPARSE_CHECK(err) \
    do { \
        if (err != CUSPARSE_STATUS_SUCCESS) { \
            printf("[cuSPARSE ERROR] Code: %d\n", err); \
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


void benchmark_cusparse(const char* mtx_path, int n_runs) {
    printf("\n============= cuSPARSE Benchmark =============\n");
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
                         (nnz + 2 * n_rows) * sizeof(double);

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

    // --- cuSPARSE Setup ---
    printf("\n[PHASE] Initializing cuSPARSE...\n");
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // Matrix descriptor
    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &matA, n_rows, n_cols, nnz,
        d_row_ptr, d_col_idx, d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F
    ));

    // Vector descriptors
    cusparseDnVecDescr_t vecX, vecB;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, n_rows, d_x, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecB, n_rows, d_b, CUDA_R_64F));

    // Solver configuration
    cusparseSpSVDescr_t spsvDescr;
    CUSPARSE_CHECK(cusparseSpSV_createDescr(&spsvDescr));
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    double alpha = 1.0;
    cusparseSpSVAlg_t alg = CUSPARSE_SPSV_ALG_DEFAULT;

    // Workspace
    size_t bufferSize;
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        handle, opA, &alpha, matA, vecB, vecX,
        CUDA_R_64F, alg, spsvDescr, &bufferSize
    ));
    void* d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, bufferSize));
    stats.device_memory += bufferSize;

    // --- Warmup ---
    printf("\n[PHASE] Warmup...\n");
    float analysis_ms, factorization_ms, solve_ms;

    CUDA_CHECK(cudaEventRecord(start, 0));
    CUSPARSE_CHECK(cusparseSpSV_analysis(
        handle, opA, &alpha, matA, vecB, vecX,
        CUDA_R_64F, alg, spsvDescr, d_workspace
    ));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&analysis_ms, start, stop));


    // // --- Incomplete LU(0) Factorization using csrilu02 ---
    // CUDA_CHECK(cudaEventRecord(start, 0));
    // cusparseMatDescr_t matA_descr;
    // CUSPARSE_CHECK(cusparseCreateMatDescr(&matA_descr));
    // CUSPARSE_CHECK(cusparseSetMatType(matA_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    // CUSPARSE_CHECK(cusparseSetMatIndexBase(matA_descr, CUSPARSE_INDEX_BASE_ZERO));

    // csrilu02Info_t info_M = NULL;
    // CUSPARSE_CHECK(cusparseCreateCsrilu02Info(&info_M));

    // int structural_zero;
    // size_t bufferSize_M;
    // void* pBuffer_M = NULL;

    // // Query buffer size for csrilu02
    // CUSPARSE_CHECK(cusparseDcsrilu02_bufferSize(
    //     handle, n_rows, nnz,
    //     matA_descr,
    //     d_values, d_row_ptr, d_col_idx,
    //     info_M, &bufferSize_M
    // ));

    // CUDA_CHECK(cudaMalloc(&pBuffer_M, bufferSize_M));

    // // Analysis phase for csrilu02
    // CUSPARSE_CHECK(cusparseDcsrilu02_analysis(
    //     handle, n_rows, nnz,
    //     matA_descr,
    //     d_values, d_row_ptr, d_col_idx,
    //     info_M,
    //     CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer_M
    // ));

    // // Perform the actual ILU(0) factorization
    // CUSPARSE_CHECK(cusparseDcsrilu02(
    //     handle, n_rows, nnz,
    //     matA_descr,
    //     d_values, d_row_ptr, d_col_idx,
    //     info_M,
    //     CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer_M
    // ));

    // // Check for structural zero pivot
    // CUSPARSE_CHECK(cusparseXcsrilu02_zeroPivot(handle, info_M, &structural_zero));
    // if (structural_zero != -1) {
    //     printf("[csrilu02 WARNING] Structural zero at row %d\\n", structural_zero);
    // }

    // CUDA_CHECK(cudaEventRecord(stop, 0));
    // CUDA_CHECK(cudaEventSynchronize(stop));
    // CUDA_CHECK(cudaEventElapsedTime(&factorization_ms, start, stop));


    CUDA_CHECK(cudaEventRecord(start, 0));
    CUSPARSE_CHECK(cusparseSpSV_solve(
        handle, opA, &alpha, matA, vecB, vecX,
        CUDA_R_64F, alg, spsvDescr
    ));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&solve_ms, start, stop));

    printf("[WARMUP] Analysis: %.2f ms | Solve: %.2f ms\n", analysis_ms, solve_ms);

    // --- Benchmark ---
    printf("\n[PHASE] Benchmarking solve phase (%d runs)...\n", n_runs);
    float total_solve = 0, min_solve = 1e9, max_solve = 0;
    float total_analysis = 0, min_analysis = 1e9, max_analysis = 0;
    float total_factorization = 0, min_factorization = 1e9, max_factorization = 0;
    float total_run_avg = 0;
    get_gpu_stats(&stats);

    for (int i = 0; i < n_runs; i++) {
        CUDA_CHECK(cudaEventRecord(start, 0));
        CUSPARSE_CHECK(cusparseSpSV_analysis(
            handle, opA, &alpha, matA, vecB, vecX,
            CUDA_R_64F, alg, spsvDescr, d_workspace
        ));
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&solve_ms, start, stop));

        CUDA_CHECK(cudaEventRecord(start, 0));
        CUSPARSE_CHECK(cusparseSpSV_solve(
            handle, opA, &alpha, matA, vecB, vecX,
            CUDA_R_64F, alg, spsvDescr
        ));
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&solve_ms, start, stop));

        total_solve += solve_ms;
        total_analysis += analysis_ms;
        total_factorization += factorization_ms;
        total_run_avg = total_solve + solve_ms + factorization_ms + analysis_ms;
        
        if(analysis_ms < min_analysis) min_analysis = analysis_ms;
        if(analysis_ms > max_analysis) max_analysis = analysis_ms;
        if(factorization_ms < min_factorization) min_factorization = factorization_ms;
        if(factorization_ms > max_factorization) max_factorization = factorization_ms;
        if(solve_ms < min_solve) min_solve = solve_ms;
        if(solve_ms > max_solve) max_solve = solve_ms;
        printf("[RUN %02d] Analysis time: %.2f ms | Factorization time: %.2f ms | Solve time: %.2f ms\n", i + 1, analysis_ms, factorization_ms, solve_ms);
    }

    // --- Results ---
    printf("\n============= Results =============\n");
    printf("[TIMING] Solve (avg/min/max): %.2f / %.2f / %.2f ms\n",
           total_solve / n_runs, min_solve, max_solve);
    printf("[TIMING] Total (analysis + solve): %.2f ms\n",
           analysis_ms + total_solve / n_runs);
    get_gpu_stats(&stats);
    printf("[MEMORY] Host: %.2f MB | Device: %.2f MB\n",
           stats.host_memory / (1024.0 * 1024.0),
           stats.device_memory / (1024.0 * 1024.0));

    // --- Cleanup ---
    printf("\n[PHASE] Cleaning up...\n");
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecB));
    CUSPARSE_CHECK(cusparseSpSV_destroyDescr(spsvDescr));
    CUSPARSE_CHECK(cusparseDestroy(handle));

    free(row_ptr); free(col_idx); free(values); free(b);
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_workspace));

    printf("[MEMORY] All resources freed\n");
}

int main(int argc, char** argv) {
    int num_of_runs = 20;
    if (argc < 2) {
        printf("Usage: %s <matrix.mtx> [runs=10]\n", argv[0]);
        return EXIT_FAILURE;
    }
    int runs = (argc > 2) ? atoi(argv[2]) : num_of_runs;
    benchmark_cusparse(argv[1], runs);
    return EXIT_SUCCESS;
}