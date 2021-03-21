all: sparse_mult.ex dense_mult.ex omp_dense.ex omp_sparse.ex cuda_dense.ex cuda_sparse.ex mpi_dense.ex

mpi_dense.ex: mpi_dense.o tools.o
	mpicc -g -Wall mpi_dense.o tools.o -o mpi_dense.ex

cuda_sparse.ex: cuda_sparse.cu
	module load soft/cuda; nvcc cuda_sparse.cu -o cuda_sparse.ex

cuda_dense.ex: cuda_dense.cu
	module load soft/cuda; nvcc cuda_dense.cu -o cuda_dense.ex

omp_sparse.ex: openmp_sparse.o tools.o
	gcc -Wall -g -O3 -fopenmp openmp_sparse.o tools.o -o omp_sparse.ex

omp_dense.ex: openmp_matmult.o tools.o
	gcc -Wall -g -O3 -fopenmp openmp_matmult.o tools.o -o omp_dense.ex

sparse_mult.ex: sparsemult_basic.o tools.o 
	gcc -Wall -g -O3 -fopenmp sparsemult_basic.o tools.o -o sparse_mult.ex

dense_mult.ex: matmult_basic.o tools.o 
	gcc -Wall -g -O3 -fopenmp matmult_basic.o tools.o -o dense_mult.ex 

mpi_dense.o: mpi_dense.c
	mpicc -g -Wall -c mpi_dense.c

sparsemult_basic.o: sparsemult_basic.c 
	gcc -Wall -g -O3 -fopenmp -c sparsemult_basic.c 

matmult_basic.o: matmult_basic.c 
	gcc -Wall -g -O3 -fopenmp -c matmult_basic.c 

openmp_matmult.o: openmp_matmult.c
	gcc -Wall -g -O3 -fopenmp -c openmp_matmult.c

openmp_sparse.o: openmp_sparse.c
	gcc -Wall -g -O3 -fopenmp -c openmp_sparse.c

tools.o: tools.c tools.h
	gcc -Wall -g -O3 -fopenmp -c tools.c 

clean:
	rm *.o *.ex
