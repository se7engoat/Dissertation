import cupy as cp
import cupyx.scipy.sparse as cusp
import cupyx.scipy.sparse.linalg as cuspl
from scipy.io import mmread  # Only for loading .mtx files (data transfer to GPU)
import numpy as np


# 1. Load matrix directly to GPU (no CPU solving)
def load_to_gpu(mtx_path):
    """Load .mtx file directly into GPU memory as CSR matrix"""
    cpu_mat = mmread(mtx_path).tocsr()  # Read on CPU (briefly)
    return cusp.csr_matrix(
        (cp.asarray(cpu_mat.data), 
         cp.asarray(cpu_mat.indices),
         cp.asarray(cpu_mat.indptr)),
        shape=cpu_mat.shape
    )

# 2. cuDSS-accelerated solver (pure GPU)
def gpu_solve(A, b):
    """Solve Ax=b using NVIDIA cuDSS via CuPy"""
    # Explicit cuDSS solver with analysis/factorization control
    solver = cuspl.spsolve(A,b)
    # solver.analyze(A)       # Symbolic analysis (GPU)
    # solver.factorize(A)     # Numerical factorization (GPU)
    # return solver.solve(b)  # Solve phase (GPU)
    print(type(solver), solver)
    return solver

# 3. Benchmark harness
def benchmark(mtx_path, n_runs=5):
    A = load_to_gpu(mtx_path)
    x_true = cp.random.rand(A.shape[0])
    b = A @ x_true  # GPU sparse matvec

    # Warm-up (primes cuDSS)
    _ = gpu_solve(A, b)
    cp.cuda.Stream.null.synchronize()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        x = gpu_solve(A, b)
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))  # ms

        residual = cp.linalg.norm(A @ x - b)
        print(f"Residual: {residual:.2e}")

    print(f"\ncuDSS Performance ({A.shape[0]}x{A.shape[1]}, {A.nnz} nnz):")
    print(f"Avg: {np.mean(times):.2f} ms | Best: {np.min(times):.2f} ms")

# Example usage
if __name__ == "__main__":
    benchmark("/home/killerbean7/Documents/Dissertation/SparseMatrices/1138_bus.mtx")

