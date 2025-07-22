#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudss.h>
#include "mmio.h"  // For .mtx file reading
#include <nvml.h>


#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

#define CUDSS_CHECK(err) \
    if (err != CUDSS_STATUS_SUCCESS) { \
        printf("cuDSS error: %d\n", err); \
        exit(EXIT_FAILURE); \
    }

void read_mtx_to_csr(const char* filename, 
                    int** row_ptr, 
                    int** col_idx, 
                    double** values, 
                    int* n_rows, 
                    int* n_cols, 
                    int* nnz) {
    FILE* f = fopen(filename, "r");
    MM_typecode matcode;
    int ret_code;
    
    if ((ret_code = mm_read_banner(f, &matcode)) != 0) {
        printf("Could not read Matrix Market banner\n");
        exit(EXIT_FAILURE);
    }

    if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode)) {
        printf("Only sparse matrices are supported\n");
        exit(EXIT_FAILURE);
    }

    // Read matrix size and non-zeros
    int M, N, nz;
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0) {
        printf("Failed to read matrix size\n");
        exit(EXIT_FAILURE);
    } else
    {
        printf("Matrix size: %d x %d, Non-zeros: %d\n", M, N, nz);
    }
    

    // Temporary COO storage
    int* I = (int*)malloc(nz * sizeof(int));
    printf("Allocated %.6f MB for I\n", 
           (double)nz * sizeof(int) / (1024.0 * 1024.0));
    int* J = (int*)malloc(nz * sizeof(int));
    printf("Allocated %.6f MB for J\n", 
           (double)nz * sizeof(int) / (1024.0 * 1024.0));
    double* val = (double*)malloc(nz * sizeof(double));
    printf("Allocated %.6f MB for val\n", 
           (double)nz * sizeof(double) / (1024.0 * 1024.0));

    // Read COO data
    for (int i = 0; i < nz; i++) {
        fscanf(f, "%d %d %lf\n", &I[i], &J[i], &val[i]);
        I[i]--;  // Convert to 0-based indexing
        J[i]--;
    }
    fclose(f);

    // Convert COO to CSR
    *row_ptr = (int*)calloc(M + 1, sizeof(int));
    printf("Allocated %.6f MB for row_ptr\n", 
           (double)(M + 1) * sizeof(int) / (1024.0 * 1024.0));
    *col_idx = (int*)malloc(nz * sizeof(int));
    printf("Allocated %.6f MB for col_idx\n", 
           (double)nz * sizeof(int) / (1024.0 * 1024.0));
    *values = (double*)malloc(nz * sizeof(double));
    printf("Allocated %.6f MB for values\n", 
           (double)nz * sizeof(double) / (1024.0 * 1024.0));

    // Count non-zeros per row
    for (int i = 0; i < nz; i++)
        (*row_ptr)[I[i] + 1]++;

    // Cumulative sum
    for (int i = 0; i < M; i++)
        (*row_ptr)[i + 1] += (*row_ptr)[i];

    // Fill CSR
    int* row_count = (int*)calloc(M, sizeof(int));
    printf("Allocated %.6f MB for row_count\n", 
           (double)M * sizeof(int) / (1024.0 * 1024.0));
    for (int i = 0; i < nz; i++) {
        int row = I[i];
        int dest = (*row_ptr)[row] + row_count[row];
        (*col_idx)[dest] = J[i];
        (*values)[dest] = val[i];
        row_count[row]++;
    }

    free(I); printf("Freed I of size %.6f MB\n", 
           (double)nz * sizeof(int) / (1024.0 * 1024.0));
    free(J); printf("Freed J of size %.6f MB\n", 
           (double)nz * sizeof(int) / (1024.0 * 1024.0));
    free(val); printf("Freed val of size %.6f MB\n", 
           (double)nz * sizeof(double) / (1024.0 * 1024.0));
    free(row_count); printf("Freed row_count of size %.6f MB\n", 
           (double)M * sizeof(int) / (1024.0 * 1024.0));
    *n_rows = M;
    *n_cols = N;
    *nnz = nz;
}

int power_measure() {
    nvmlReturn_t result;
    nvmlDevice_t device;
    unsigned int power;
    nvmlMemory_t memoryInfo;

    result = nvmlInit();
    if (NVML_SUCCESS != result) {
        printf("nvmlInit failed: %s\n", nvmlErrorString(result));
        return 1;
    }

    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (NVML_SUCCESS != result) {
        printf("nvmlDeviceGetHandleByIndex failed: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        return 1;
    }

    result = nvmlDeviceGetPowerUsage(device, &power);
    if (NVML_SUCCESS != result) {
        printf("nvmlDeviceGetPowerUsage failed: %s\n", nvmlErrorString(result));
    } else {
        printf("GPU Power Usage: %u watts\n", power/1000);
    }

    result = nvmlDeviceGetMemoryInfo(device, &memoryInfo);
    if (NVML_SUCCESS != result) {
        printf("nvmlDeviceGetMemoryInfo failed: %s\n", nvmlErrorString(result));
    } else {
        printf("Total GPU Memory: %llu megabytes\n", (unsigned long long)memoryInfo.total/1000000);
        printf("Used GPU Memory: %llu megabytes\n", (unsigned long long)memoryInfo.used/1000000);
    }

    nvmlShutdown();
    return (int) memoryInfo.used/1000000; // Return used memory in MB
}
void benchmark_cudss(const char* mtx_path, int n_runs) {
    // Read matrix
    int *row_ptr, *col_idx, *d_row_ptr, *d_col_idx;
    double *values, *d_values, *d_b, *d_x;
    int n_rows, n_cols, nnz;

    read_mtx_to_csr(mtx_path, &row_ptr, &col_idx, &values, &n_rows, &n_cols, &nnz);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_row_ptr, (n_rows + 1) * sizeof(int)));
    printf("Allocated %.6f MB for d_row_ptr\n", 
           (double)(n_rows + 1) * sizeof(int) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaMalloc((void**)&d_col_idx, nnz * sizeof(int)));
    printf("Allocated %.6f MB for d_col_idx\n", 
           (double)nnz * sizeof(int) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaMalloc((void**)&d_values, nnz * sizeof(double)));
    printf("Allocated %.6f MB for d_values\n", 
           (double)nnz * sizeof(double) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaMalloc((void**)&d_b, n_rows * sizeof(double)));
    printf("Allocated %.6f MB for d_b\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaMalloc((void**)&d_x, n_rows * sizeof(double)));
    printf("Allocated %.6f MB for d_x\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr, (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    printf("Copying row_ptr to device of size %.6f MB\n", 
           (double)(n_rows + 1) * sizeof(int) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    printf("Copying col_idx to device of size %.6f MB\n", 
           (double)nnz * sizeof(int) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaMemcpy(d_values, values, nnz * sizeof(double), cudaMemcpyHostToDevice));
    printf("Copying values to device of size %.6f MB\n", 
           (double)nnz * sizeof(double) / (1024.0 * 1024.0));

    // Generate random RHS
    double* b = (double*)malloc(n_rows * sizeof(double));
    printf("Allocated %.6f MB for b\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
    for (int i = 0; i < n_rows; i++) 
        b[i] = (double)rand() / RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d_b, b, n_rows * sizeof(double), cudaMemcpyHostToDevice));
    printf("Copying b to device of size %.6f MB\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));

    cudaSetDevice(0);  // Ensure CUDA context is created
    cudaFree(0);       // Safe no-op that forces context creation

    // cuDSS setup
    cudssHandle_t handle;
    cudssConfig_t config;
    cudssData_t data;
    CUDSS_CHECK(cudssCreate(&handle));
    CUDSS_CHECK(cudssConfigCreate(&config));
    CUDSS_CHECK(cudssDataCreate(handle, &data));

    // Create matrix descriptor
    cudssMatrix_t A;
    CUDSS_CHECK(cudssMatrixCreateCsr(
        &A, n_rows, n_cols, nnz,
        d_row_ptr, NULL, d_col_idx, d_values,
        CUDA_R_32I, CUDA_R_64F,
        CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO
    ));

    // Create vector descriptors
    cudssMatrix_t b_mat, x_mat;
    CUDSS_CHECK(cudssMatrixCreateDn(&b_mat, n_rows, 1, n_rows, d_b, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    printf("Created b_mat of size %.6f MB\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
    CUDSS_CHECK(cudssMatrixCreateDn(&x_mat, n_rows, 1, n_rows, d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    printf("Created x_mat of size %.6f MB\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));

    // Warm-up
    int mem1 = power_measure();
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, A, x_mat, b_mat));
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, A, x_mat, b_mat));
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A, x_mat, b_mat));

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float total_ms = 0;

    for (int i = 0; i < n_runs; i++) {
        CUDA_CHECK(cudaEventRecord(start, 0));
        CUDSS_CHECK(cudssExecute(
            handle, CUDSS_PHASE_SOLVE, config,
            data, A, x_mat, b_mat
        ));
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
        printf("Run %d: %.2f ms\n", i+1, ms);
    }

    printf("\nAverage solve time: %.2f ms\n", total_ms / n_runs);
    int mem2 = power_measure();
    printf("Memory usage: %d Megabytes\n", mem2 - mem1);

    // Cleanup
    CUDSS_CHECK(cudssMatrixDestroy(A));
    printf("Freed matrix A of size %.6f MB\n", 
           (double)(n_rows + 1 + nnz) * sizeof(int) / (1024.0 * 1024.0) + 
           (double)nnz * sizeof(double) / (1024.0 * 1024.0));
    CUDSS_CHECK(cudssMatrixDestroy(b_mat));
    printf("Freed matrix b_mat of size %.6f MB\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
    CUDSS_CHECK(cudssMatrixDestroy(x_mat));
    printf("Freed matrix x_mat of size %.6f MB\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
    CUDSS_CHECK(cudssDataDestroy(handle, data));
    printf("Freed data of size %.6f MB\n", 
           (double)(n_rows + 1 + nnz) * sizeof(int) / (1024.0 * 1024.0) + 
           (double)nnz * sizeof(double) / (1024.0 * 1024.0));
    CUDSS_CHECK(cudssConfigDestroy(config));
    CUDSS_CHECK(cudssDestroy(handle));

    free(row_ptr); printf("Freed row_ptr of size %.6f MB\n", 
           (double)(n_rows + 1) * sizeof(int) / (1024.0 * 1024.0));
    free(col_idx); printf("Freed col_idx of size %.6f MB\n", 
           (double)nnz * sizeof(int) / (1024.0 * 1024.0));
    free(values); printf("Freed values of size %.6f MB\n", 
           (double)nnz * sizeof(double) / (1024.0 * 1024.0));
    free(b); printf("Freed b of size %.6f MB\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaFree(d_row_ptr));
    printf("Freed d_row_ptr of size %.6f MB\n", 
           (double)(n_rows + 1) * sizeof(int) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaFree(d_col_idx));
    printf("Freed d_col_idx of size %.6f MB\n", 
           (double)nnz * sizeof(int) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaFree(d_values));
    printf("Freed d_values of size %.6f MB\n", 
           (double)nnz * sizeof(double) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaFree(d_b));
    printf("Freed d_b of size %.6f MB\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaFree(d_x));
    printf("Freed d_x of size %.6f MB\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
}

int main(int argc, char** argv) {
    int number_of_runs = 10; // Default number of runs
    if (argc < 2) {
        printf("Usage: %s <matrix.mtx> [runs=%d]\n", argv[0], number_of_runs);
        return EXIT_FAILURE;
    }
    int runs = (argc > 2) ? atoi(argv[2]) : number_of_runs;
    benchmark_cudss(argv[1], runs);
    return EXIT_SUCCESS;
}