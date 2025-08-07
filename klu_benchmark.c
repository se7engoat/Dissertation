#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "klu.h"
#include "mmio.h"

#define KLU_CHECK(err, common) \
    if ((err) < 0) { \
        printf("KLU error: %d (status: %d)\n", (err), (common).status); \
        exit(EXIT_FAILURE); \
    }

// Memory tracking structure
typedef struct {
    size_t host_memory;
    size_t device_memory;
} memory_stats_t;

// Function to read matrix in Matrix Market format and convert to KLU's compressed column format
void read_mtx_to_klu(const char* filename, 
                     int** Ap, 
                     int** Ai, 
                     double** Ax, 
                     int* n, 
                     int* nnz,
                     memory_stats_t* mem_stats) {
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

    if (M != N) {
        printf("KLU only works with square matrices\n");
        exit(EXIT_FAILURE);
    }

    // Temporary COO storage
    int* I = (int*)malloc(nz * sizeof(int));
    mem_stats->host_memory += nz * sizeof(int);
    printf("Allocated %.6f MB for I\n", (double)nz * sizeof(int) / (1024.0 * 1024.0));
    
    int* J = (int*)malloc(nz * sizeof(int));
    mem_stats->host_memory += nz * sizeof(int);
    printf("Allocated %.6f MB for J\n", (double)nz * sizeof(int) / (1024.0 * 1024.0));
    
    double* val = (double*)malloc(nz * sizeof(double));
    mem_stats->host_memory += nz * sizeof(double);
    printf("Allocated %.6f MB for val\n", (double)nz * sizeof(double) / (1024.0 * 1024.0));

    // Read COO data
    for (int i = 0; i < nz; i++) {
        fscanf(f, "%d %d %lf\n", &I[i], &J[i], &val[i]);
        I[i]--;  // Convert to 0-based indexing
        J[i]--;
    }
    fclose(f);

    // Convert COO to CSC (KLU uses compressed column format)
    *Ap = (int*)calloc(N + 1, sizeof(int));
    mem_stats->host_memory += (N + 1) * sizeof(int);
    printf("Allocated %.6f MB for Ap\n", (double)(N + 1) * sizeof(int) / (1024.0 * 1024.0));
    
    *Ai = (int*)malloc(nz * sizeof(int));
    mem_stats->host_memory += nz * sizeof(int);
    printf("Allocated %.6f MB for Ai\n", (double)nz * sizeof(int) / (1024.0 * 1024.0));
    
    *Ax = (double*)malloc(nz * sizeof(double));
    mem_stats->host_memory += nz * sizeof(double);
    printf("Allocated %.6f MB for Ax\n", (double)nz * sizeof(double) / (1024.0 * 1024.0));

    // Count non-zeros per column
    for (int i = 0; i < nz; i++)
        (*Ap)[J[i] + 1]++;

    // Cumulative sum
    for (int i = 0; i < N; i++)
        (*Ap)[i + 1] += (*Ap)[i];

    // Fill CSC
    int* col_count = (int*)calloc(N, sizeof(int));
    mem_stats->host_memory += N * sizeof(int);
    printf("Allocated %.6f MB for col_count\n", (double)N * sizeof(int) / (1024.0 * 1024.0));
    
    for (int i = 0; i < nz; i++) {
        int col = J[i];
        int dest = (*Ap)[col] + col_count[col];
        (*Ai)[dest] = I[i];
        (*Ax)[dest] = val[i];
        col_count[col]++;
    }

    free(I); 
    mem_stats->host_memory -= nz * sizeof(int);
    printf("Freed I of size %.6f MB\n", (double)nz * sizeof(int) / (1024.0 * 1024.0));
    
    free(J); 
    mem_stats->host_memory -= nz * sizeof(int);
    printf("Freed J of size %.6f MB\n", (double)nz * sizeof(int) / (1024.0 * 1024.0));
    
    free(val); 
    mem_stats->host_memory -= nz * sizeof(double);
    printf("Freed val of size %.6f MB\n", (double)nz * sizeof(double) / (1024.0 * 1024.0));
    
    free(col_count); 
    mem_stats->host_memory -= N * sizeof(int);
    printf("Freed col_count of size %.6f MB\n", (double)N * sizeof(int) / (1024.0 * 1024.0));
    
    *n = N;
    *nnz = nz;
}

// Timer function
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void benchmark_klu(const char* mtx_path, int n_runs) {
    printf("======= Benchmarking KLU with %d runs on matrix: %s =======\n", n_runs, mtx_path);
    
    memory_stats_t mem_stats = {0, 0};
    
    // Read matrix
    int *Ap, *Ai;
    double *Ax;
    int n, nnz;
    
    read_mtx_to_klu(mtx_path, &Ap, &Ai, &Ax, &n, &nnz, &mem_stats);
    
    // Generate random RHS
    double* b = (double*)malloc(n * sizeof(double));
    mem_stats.host_memory += n * sizeof(double);
    printf("\nAllocated %.6f MB for b\n", (double)n * sizeof(double) / (1024.0 * 1024.0));
    
    double* x = (double*)malloc(n * sizeof(double));
    mem_stats.host_memory += n * sizeof(double);
    printf("Allocated %.6f MB for x\n", (double)n * sizeof(double) / (1024.0 * 1024.0));
    
    for (int i = 0; i < n; i++) {
        b[i] = (double)rand() / RAND_MAX;
        x[i] = 0.0;
    }
    
    // KLU setup
    klu_common Common;
    klu_defaults(&Common);
    
    // Configure KLU parameters (can be adjusted)
    Common.tol = 0.001;      // Partial pivoting tolerance
    Common.btf = 1;          // Use BTF pre-ordering
    Common.ordering = 0;      // 0:AMD, 1:COLAMD, 2:user P and Q, 3:user function
    Common.scale = 2;        // Scale rows by max entry
    
    // Symbolic analysis
    double start_time = get_time();
    klu_symbolic *Symbolic = klu_analyze(n, Ap, Ai, &Common);
    KLU_CHECK(Common.status, Common);
    double analysis_time = get_time() - start_time;
    printf("\nSymbolic analysis time: %.6f ms\n", analysis_time * 1000);
    
    // Numeric factorization
    start_time = get_time();
    klu_numeric *Numeric = klu_factor(Ap, Ai, Ax, Symbolic, &Common);
    KLU_CHECK(Common.status, Common);
    double factor_time = get_time() - start_time;
    printf("Numeric factorization time: %.6f ms\n", factor_time * 1000);
    
    // Initial solve (warm-up)
    start_time = get_time();
    int ok = klu_solve(Symbolic, Numeric, n, 1, b, &Common);
    KLU_CHECK(ok, Common);
    double solve_time = get_time() - start_time;
    printf("Initial solve time: %.6f ms\n", solve_time * 1000);
    
    // Copy solution to x
    // for (int i = 0; i < n; i++) {
    //     x[i] = b[i];
    // }
    
    // Benchmark 
    float total_solve_time = 0, total_factor_time = 0, total_analysis_time = 0;
    float min_analysis_time = 1e9, max_analysis_time = 0;
    float min_factor_time = 1e9, max_factor_time = 0;
    float min_solve_time = 1e9;
    float max_solve_time = 0;
    
    
    for (int i = 0; i < n_runs; i++) {
        // Reset RHS (reusing the same values for consistent benchmarking)
        for (int j = 0; j < n; j++) {
            b[j] = x[j];
        }
        //measure analysis and factorization time on each run
        float analysis_time = 0;
        float factor_time = 0;
        float solve_time = 0;

        start_time = get_time();
        klu_symbolic *Symbolic = klu_analyze(n, Ap, Ai, &Common);
        KLU_CHECK(Common.status, Common);
        analysis_time = get_time() - start_time;

        start_time = get_time();
        klu_numeric *Numeric = klu_factor(Ap, Ai, Ax, Symbolic, &Common);
        KLU_CHECK(Common.status, Common);
        factor_time = get_time() - start_time;
        
        start_time = get_time();
        ok = klu_solve(Symbolic, Numeric, n, 1, b, &Common);
        KLU_CHECK(ok, Common);
        solve_time = get_time() - start_time;
        

        total_solve_time +=  solve_time;
        total_factor_time += factor_time;
        total_analysis_time += analysis_time;

        if(analysis_time < min_analysis_time) min_analysis_time = analysis_time;
        if(analysis_time > max_analysis_time) max_analysis_time = analysis_time;
        if(factor_time < min_factor_time) min_factor_time = factor_time;
        if(factor_time > max_factor_time) max_factor_time = factor_time;
        if(solve_time < min_solve_time) min_solve_time = solve_time;
        if(solve_time > max_solve_time) max_solve_time = solve_time;
        printf("[RUN %02d]: Analysis time: %.6f ms | Factorization time: %.6f ms | Solve time: %.6f ms\n", 
                i + 1, analysis_time * 1000, factor_time * 1000, solve_time * 1000);

    }
    
    // Print statistics
    printf("\n======= KLU Benchmark Results =======\n");
    printf("Matrix: %s (%d x %d, nnz: %d)\n", mtx_path, n, n, nnz);
    printf("Symbolic analysis time: %.6f ms\n", analysis_time * 1000);
    printf("Numeric factorization time: %.6f ms\n", factor_time * 1000);
    printf("\nSolve phase statistics (%d runs):\n", n_runs);
    printf("  Average time: %.6f ms\n", (total_solve_time / n_runs) * 1000);
    printf("  Minimum time: %.6f ms\n", min_solve_time * 1000);
    printf("  Maximum time: %.6f ms\n", max_solve_time * 1000);
    printf("  Total time: %.6f ms\n", total_solve_time * 1000);

    // --- Results ---
    printf("\n============= Results =============\n");
    printf("[TIMING] Solve (avg/min/max): %.6f / %.6f / %.6f ms\n",
            total_solve_time*1000 / n_runs, min_solve_time*1000, max_solve_time*1000);
    printf("[TIMING] Analysis (avg/min/max): %.6f / %.6f / %.6f ms\n",
            total_analysis_time*1000 / n_runs, min_analysis_time*1000, max_analysis_time*1000);
    printf("[TIMING] Factorization (avg/min/max): %.6f / %.6f / %.6f ms\n",
            total_factor_time*1000 / n_runs, min_factor_time*1000, max_factor_time*1000);

    printf("[TIMING] Total (analysis + factorization + solve): %.6f ms\n",
            (total_analysis_time*1000 + total_factor_time*1000 + total_solve_time*1000) / n_runs);

    // Print KLU statistics
    printf("\nKLU Statistics:\n");
    printf("  Structural rank: %d\n", Common.structural_rank);
    printf("  Numerical rank: %d\n", Common.numerical_rank);
    printf("  Singular column: %d\n", Common.singular_col);
    printf("  Off-diagonal pivots: %d\n", Common.noffdiag);
    printf("  Flop count: %.0f\n", Common.flops);
    printf("  Reciprocal condition estimate: %.6g\n", Common.rcond);
    printf("  Condition estimate: %.6g\n", Common.condest);
    printf("  Reciprocal pivot growth: %.6g\n", Common.rgrowth);
    printf("  Memory usage: %.6f MB\n", (double)Common.memusage / (1024.0 * 1024.0));
    printf("  Peak memory usage: %.6f MB\n", (double)Common.mempeak / (1024.0 * 1024.0));
    
    // Cleanup
    klu_free_symbolic(&Symbolic, &Common);
    klu_free_numeric(&Numeric, &Common);
    
    free(Ap); 
    mem_stats.host_memory -= (n + 1) * sizeof(int);
    printf("\nFreed Ap of size %.6f MB\n", (double)(n + 1) * sizeof(int) / (1024.0 * 1024.0));
    
    free(Ai); 
    mem_stats.host_memory -= nnz * sizeof(int);
    printf("Freed Ai of size %.6f MB\n", (double)nnz * sizeof(int) / (1024.0 * 1024.0));
    
    free(Ax); 
    mem_stats.host_memory -= nnz * sizeof(double);
    printf("Freed Ax of size %.6f MB\n", (double)nnz * sizeof(double) / (1024.0 * 1024.0));
    
    free(b); 
    mem_stats.host_memory -= n * sizeof(double);
    printf("Freed b of size %.6f MB\n", (double)n * sizeof(double) / (1024.0 * 1024.0));
    
    free(x); 
    mem_stats.host_memory -= n * sizeof(double);
    printf("Freed x of size %.6f MB\n", (double)n * sizeof(double) / (1024.0 * 1024.0));
    
    printf("\nTotal host memory allocated/freed: %.6f MB\n", 
           (double)mem_stats.host_memory / (1024.0 * 1024.0));
}

int main(int argc, char** argv) {
    int number_of_runs = 100; // Default number of runs
    if (argc < 2) {
        printf("\nUsage: %s <matrix.mtx> [runs=%d]\n", argv[0], number_of_runs);
        return EXIT_FAILURE;
    }
    int runs = (argc > 2) ? atoi(argv[2]) : number_of_runs;
    benchmark_klu(argv[1], runs);
    return EXIT_SUCCESS;
}