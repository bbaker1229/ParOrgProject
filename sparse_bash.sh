echo "Sequential"
./sparse_mult.ex 

echo "1 thread"
export OMP_NUM_THREADS=1
./omp_sparse.ex

echo "2 threads"
export OMP_NUM_THREADS=2
./omp_sparse.ex

echo "4 threads"
export OMP_NUM_THREADS=4
./omp_sparse.ex

echo "8 threads"
export OMP_NUM_THREADS=8
./omp_sparse.ex

echo "16 threads"
export OMP_NUM_THREADS=16
./omp_sparse.ex

echo "32 threads"
export OMP_NUM_THREADS=32
./omp_sparse.ex

echo "64 threads"
export OMP_NUM_THREADS=64
./omp_sparse.ex

