all: sparse_mult.ex dense_mult.ex

sparse_mult.ex: sparsemult_basic.o tools.o 
	gcc -Wall -g -O3 -fopenmp sparsemult_basic.o tools.o -o sparse_mult.ex

dense_mult.ex: matmult_basic.o tools.o 
	gcc -Wall -g -O3 -fopenmp matmult_basic.o tools.o -o dense_mult.ex 

sparsemult_basic.o: sparsemult_basic.c 
	gcc -Wall -g -O3 -fopenmp -c sparsemult_basic.c 

matmult_basic.o: matmult_basic.c 
	gcc -Wall -g -O3 -fopenmp -c matmult_basic.c 

tools.o: tools.c tools.h
	gcc -Wall -g -O3 -fopenmp -c tools.c 

clean:
	rm *.o *.ex
