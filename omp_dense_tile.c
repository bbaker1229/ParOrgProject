#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "tools.h"

int main(int argc, char *argv[]) {
    int idim = 1000;
    int jdim = 1000;
    int kdim = 1000;
    int i, j, k, nt, block;
    double t1, times[200];
    float nops, err;
    float *A, *B, *C, *actualC;
    A = (float*) malloc(idim*kdim*sizeof(float));
    B = (float*) malloc(kdim*jdim*sizeof(float));
    C = (float*) malloc(idim*jdim*sizeof(float));
    actualC = (float*) malloc(idim*jdim*sizeof(float));

    // Initialize matrices
    zero_init(idim, jdim, C);
    zero_init(idim, jdim, actualC);
    rand_init(idim, kdim, A);
    rand_init(kdim, jdim, B);

    //printf("A matrix sample: \n");
    //print_sample(idim, kdim, A, 2, 10);
    //printf("B matrix sample: \n");
    //print_sample(kdim, jdim, B, 2, 10);
    // This is the standard matrix multiplication - do not adjust
    matrix_mult(idim, jdim, kdim, A, B, actualC);

    //printf("actualC matrix sample: \n");
    //print_sample(idim, jdim, actualC, 2, 10);
    // Begin test multiplication
    block = atoi(argv[1]);
    printf("%d\n", block);
    #pragma omp parallel
    nt = omp_get_num_threads();
    printf("Running with %d threads\n", nt);
    for(int loop_cnt = 0; loop_cnt < 200; loop_cnt++) {
    t1 = wctime();  // record start time
    #pragma omp parallel shared(A, B, C) private(i, j, k) 
    {
    #pragma omp for schedule(static)
    for(int i1 = 0; i1 < idim; i1 += block)
    for(int k1 = 0; k1 < kdim; k1 += block)
    for(int j1 = 0; j1 < jdim; j1 += block)
    for(i = i1; i < i1 + block; i++) {
        for(k = k1; k < k1 + block; k++) {
	    for(j = j1; j < j1 + block; j++) {
                C[i*jdim+j] += (A[i*kdim+k] * B[k*jdim+j]);
	    }
	}
    }
    }
    t1 = wctime() - t1;  // record elapsed time
    times[loop_cnt] = t1;
    if(loop_cnt != 199)
      zero_init(idim, jdim, C);
    }

    //printf("C matrix sample: \n");
    //print_sample(idim, jdim, C, 2, 10);

    // error calculation
    err = error_calc(idim, jdim, actualC, C);
 
    t1 = 0.0;
    for(i=0; i < 200; i++)
      t1 += times[i];
    t1 /= (float) 200;
    printf("Finished in %lf seconds\n", t1);
    t1 *= (1.e+09);
    nops = (float) 2 * idim * kdim * jdim;
    printf("Performance = %f GFLOPs\n",nops/t1);
    printf("Error: %f\n", err);
    return(0);
}
