#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudss.h>
#include "mmio.h"  // For .mtx file reading

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
    }

    // Temporary COO storage
    int* I = (int*)malloc(nz * sizeof(int));
    int* J = (int*)malloc(nz * sizeof(int));
    double* val = (double*)malloc(nz * sizeof(double));

    // Read COO data
    for (int i = 0; i < nz; i++) {
        fscanf(f, "%d %d %lf\n", &I[i], &J[i], &val[i]);
        I[i]--;  // Convert to 0-based indexing
        J[i]--;
    }
    fclose(f);

    // Convert COO to CSR
    *row_ptr = (int*)calloc(M + 1, sizeof(int));
    *col_idx = (int*)malloc(nz * sizeof(int));
    *values = (double*)malloc(nz * sizeof(double));

    // Count non-zeros per row
    for (int i = 0; i < nz; i++)
        (*row_ptr)[I[i] + 1]++;

    // Cumulative sum
    for (int i = 0; i < M; i++)
        (*row_ptr)[i + 1] += (*row_ptr)[i];

    // Fill CSR
    int* row_count = (int*)calloc(M, sizeof(int));
    for (int i = 0; i < nz; i++) {
        int row = I[i];
        int dest = (*row_ptr)[row] + row_count[row];
        (*col_idx)[dest] = J[i];
        (*values)[dest] = val[i];
        row_count[row]++;
    }

    free(I); free(J); free(val); free(row_count);
    *n_rows = M;
    *n_cols = N;
    *nnz = nz;
}

void benchmark_cudss(const char* mtx_path, int n_runs) {
    // Read matrix
    int *row_ptr, *col_idx, *d_row_ptr, *d_col_idx;
    double *values, *d_values, *d_b, *d_x;
    int n_rows, n_cols, nnz;

    read_mtx_to_csr(mtx_path, &row_ptr, &col_idx, &values, &n_rows, &n_cols, &nnz);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n_rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, n_rows * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr, (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values, nnz * sizeof(double), cudaMemcpyHostToDevice));

    // Generate random RHS
    double* b = (double*)malloc(n_rows * sizeof(double));
    for (int i = 0; i < n_rows; i++) 
        b[i] = (double)rand() / RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d_b, b, n_rows * sizeof(double), cudaMemcpyHostToDevice));

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
    CUDSS_CHECK(cudssMatrixCreateDn(&x_mat, n_rows, 1, n_rows, d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

    // Warm-up
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

    // Cleanup
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
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <matrix.mtx> [runs=5]\n", argv[0]);
        return EXIT_FAILURE;
    }
    int runs = (argc > 2) ? atoi(argv[2]) : 5;
    benchmark_cudss(argv[1], runs);
    return EXIT_SUCCESS;
}