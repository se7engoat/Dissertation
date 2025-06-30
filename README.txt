Setup instructions for running CuPy on a linux environment.
Pre-requisites:
Install cuda toolkit
Install cuDSS
Install conda for running venv

Commands for running conda venv:
conda create -n cudss_env python=3.9 -y
conda activate cudss_env
conda deactivate


> pip install cupy-cuda12x scipy 
- check if the cuda version on system is more than v12.0,
otherwise use cupy-cuda11x.


env dependencies to run the python code:
pip install scipy cupy-cuda12x pyJoules pyJoules[nvidia]


Benchmarking using the Cuda code in C:
nvcc benchmark.c -o benchmark -lcudss -lmmio -lcudart

>should give some warnings but don't care
Running object file:
./benchmark <matrix_file.mtx>
