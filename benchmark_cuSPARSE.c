#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include "mmio.h"  // For .mtx file reading
#include <nvml.h>  // For NVML GPU power/memory measurement

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

#define CUSPARSE_CHECK(err) \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        printf("cuSPARSE error: %d\n", err); \
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
    } else {
        printf("Matrix size: %d x %d, Non-zeros: %d\n", M, N, nz);
    }

    // Temporary COO storage
    int* I = (int*)malloc(nz * sizeof(int));
    printf("Allocated %.6f MB for row indices\n", 
           (double)nz*sizeof(int)/(1024.0*1024.0));
    int* J = (int*)malloc(nz * sizeof(int));
    printf("Allocated %.6f MB for column indices\n", 
           (double)nz*sizeof(int)/(1024.0*1024.0));
    double* val = (double*)malloc(nz * sizeof(double));
    printf("Allocated %.6f MB for values\n", 
           (double)nz*sizeof(double)/(1024.0*1024.0));
    //adding this check to ensure memory allocation was successful
    if (!val) {
        fprintf(stderr, "Failed to allocate %.6f MB for values\n", 
           (double)nz*sizeof(double)/(1024.0*1024.0));
        exit(EXIT_FAILURE);
    }

    // Read COO data
    for (int i = 0; i < nz; i++) {
        fscanf(f, "%d %d %lf\n", &I[i], &J[i], &val[i]);
        I[i]--;  // Convert to 0-based indexing
        J[i]--;
    }
    fclose(f);

    // Convert COO to CSR
    *row_ptr = (int*)calloc(M + 1, sizeof(int));
    printf("Allocated %.6f MB for row pointers\n", 
           (double)(M + 1) * sizeof(int) / (1024.0 * 1024.0));
    *col_idx = (int*)malloc(nz * sizeof(int));
    printf("Allocated %.6f MB for column indices\n", 
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
    printf("Allocated %.6f MB for row count\n", 
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

void benchmark_cusparse(const char* mtx_path, int n_runs) {
    // Read matrix
    int *row_ptr, *col_idx, *d_row_ptr, *d_col_idx;
    double *values, *d_values, *d_b, *d_x;
    int n_rows, n_cols, nnz;

    read_mtx_to_csr(mtx_path, &row_ptr, &col_idx, &values, &n_rows, &n_cols, &nnz);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_row_ptr, (n_rows + 1) * sizeof(int)));
    printf("Allocated %.6f MB for device row pointers - d_row_ptr\n", 
           (double)(n_rows + 1) * sizeof(int) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaMalloc((void**)&d_col_idx, nnz * sizeof(int)));
    printf("Allocated %.6f MB for device column indices - d_col_idx\n", 
           (double)nnz * sizeof(int) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaMalloc((void**)&d_values, nnz * sizeof(double)));
    printf("Allocated %.6f MB for device values - d_values\n", 
           (double)nnz * sizeof(double) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaMalloc((void**)&d_b, n_rows * sizeof(double)));
    printf("Allocated %.6f MB for device RHS vector - d_b\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaMalloc((void**)&d_x, n_rows * sizeof(double)));
    printf("Allocated %.6f MB for device solution vector - d_x\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr, (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    printf("Copying row pointers to device - d_row_ptr of size %.6f MB\n", 
           (double)(n_rows + 1) * sizeof(int) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    printf("Copying column indices to device - d_col_idx of size %.6f MB\n", 
           (double)nnz * sizeof(int) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaMemcpy(d_values, values, nnz * sizeof(double), cudaMemcpyHostToDevice));
    printf("Copying values to device - d_values of size %.6f MB\n", 
           (double)nnz * sizeof(double) / (1024.0 * 1024.0));

    // Generate random RHS
    double* b = (double*)malloc(n_rows * sizeof(double));
    printf("Allocated %.6f MB for host RHS vector - b\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
    for (int i = 0; i < n_rows; i++) 
        b[i] = (double)rand() / RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d_b, b, n_rows * sizeof(double), cudaMemcpyHostToDevice));
    printf("Copying RHS vector to device - d_b of size %.6f MB\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
           
    cudaSetDevice(0); // Ensure cuda context is created
    cudaFree(0); // Safe no-op to ensure context is created

    // cuSPARSE setup
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // Create matrix descriptor
    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &matA, n_rows, n_cols, nnz,
        d_row_ptr, d_col_idx, d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, // originally CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F
    ));

    // Create vector descriptors
    cusparseDnVecDescr_t vecX, vecB;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, n_rows, d_x, CUDA_R_64F));
    printf("Created device vector descriptor vecX of size %.6f MB\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecB, n_rows, d_b, CUDA_R_64F));
    printf("Created device vector descriptor vecB of size %.6f MB\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));

    // Solver setup (for triangular solve)
    cusparseSpSVDescr_t spsvDescr;
    CUSPARSE_CHECK(cusparseSpSV_createDescr(&spsvDescr));

    // Configure solver (assuming lower triangular solve)
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    double alpha = 1.0;
    cusparseSpSVAlg_t alg = CUSPARSE_SPSV_ALG_DEFAULT;
    cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;

    // Buffer size query
    size_t bufferSize;
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        handle, opA, &alpha, matA, vecB, vecX,
        CUDA_R_64F, alg, spsvDescr, &bufferSize
    ));

    // Allocate workspace
    void* d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, bufferSize));
    printf("Allocated %.6f MB for device workspace - d_workspace\n", 
           (double)bufferSize / (1024.0 * 1024.0));

    // Warm-up
    int mem1 = power_measure();
    CUSPARSE_CHECK(cusparseSpSV_analysis(handle, opA, &alpha, matA, vecB, vecX,CUDA_R_64F, alg, spsvDescr, d_workspace));
    CUSPARSE_CHECK(cusparseSpSV_solve(handle, opA, &alpha, matA, vecB, vecX,CUDA_R_64F, alg, spsvDescr));

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float total_ms = 0;

    for (int i = 0; i < n_runs; i++) {
        CUDA_CHECK(cudaEventRecord(start, 0));
        CUSPARSE_CHECK(cusparseSpSV_solve(
            handle, opA, &alpha, matA, vecB, vecX,
            CUDA_R_64F, alg, spsvDescr
        ));
        //SpSV solve uses as triangular solve, so we assume the matrix is triangular
        // If the matrix is not triangular, you would need to adjust the operation and fill mode
        // accordingly.

        //instead of SpSV solve, i need to use analysis and solve separately
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
        printf("Run %d: %.2f ms\n", i+1, ms);
    }

    printf("\nAverage solve time: %.2f ms\n", total_ms / n_runs);
    int mem2 = power_measure();
    printf("Memory used: %d megabytes\n", mem2 - mem1);
    // Cleanup
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    printf("Destroyed sparse matrix descriptor matA of size %.6f MB\n", 
           (double)(n_rows + 1) * sizeof(int) / (1024.0 * 1024.0));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    printf("Destroyed device vector descriptor vecX of size %.6f MB\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecB));
    printf("Destroyed device vector descriptor vecB of size %.6f MB\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
    CUSPARSE_CHECK(cusparseSpSV_destroyDescr(spsvDescr));
    printf("Destroyed sparse solver descriptor spsvDescr of size %.6f MB\n", 
           (double)(n_rows + 1) * sizeof(int) / (1024.0 * 1024.0));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    printf("Destroyed cuSPARSE handle\n");

    free(row_ptr); 
    printf("Freed row_ptr of size %.6f MB\n", 
           (double)(n_rows + 1) * sizeof(int) / (1024.0 * 1024.0));
    free(col_idx);
    printf("Freed col_idx of size %.6f MB\n", 
           (double)nnz * sizeof(int) / (1024.0 * 1024.0)); 
    free(values); 
    printf("Freed values of size %.6f MB\n", 
           (double)nnz * sizeof(double) / (1024.0 * 1024.0));
    free(b);
    printf("Freed host RHS vector b of size %.6f MB\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaFree(d_row_ptr));
    printf("Freed device row pointers d_row_ptr of size %.6f MB\n", 
           (double)(n_rows + 1) * sizeof(int) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaFree(d_col_idx));
    printf("Freed device column indices d_col_idx of size %.6f MB\n", 
           (double)nnz * sizeof(int) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaFree(d_values));
    printf("Freed device values d_values of size %.6f MB\n", 
           (double)nnz * sizeof(double) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaFree(d_b));
    printf("Freed device RHS vector d_b of size %.6f MB\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaFree(d_x));
    printf("Freed device solution vector d_x of size %.6f MB\n", 
           (double)n_rows * sizeof(double) / (1024.0 * 1024.0));
    CUDA_CHECK(cudaFree(d_workspace));
    printf("Freed device workspace d_workspace of size %.6f MB\n", 
           (double)bufferSize / (1024.0 * 1024.0));
}

int main(int argc, char** argv) {
    int number_of_runs = 10; // Default number of runs
    if (argc < 2) {
        printf("Usage: %s <matrix.mtx> [runs=%d]\n", argv[0], number_of_runs);
        return EXIT_FAILURE;
    }
    int runs = (argc > 2) ? atoi(argv[2]) : number_of_runs;
    benchmark_cusparse(argv[1], runs);
    return EXIT_SUCCESS;
}