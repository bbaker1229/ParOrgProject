#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "tools.h"

int main(int argc, char *argv[]) {
    int idim = 200;
    int jdim = 400;
    int kdim = 1000;
    int i, j, k, nt;
    double t1;
    float nops;
    float **A, **B, **C, **actualC;
    A = (float**) malloc(idim*sizeof(float*));
    B = (float**) malloc(kdim*sizeof(float*));
    C = (float**) malloc(idim*sizeof(float*));
    actualC = (float**) malloc(idim*sizeof(float*));
    for(i=0; i<idim; i++) {
        A[i] = (float*) malloc(kdim*sizeof(float));
        C[i] = (float*) malloc(jdim*sizeof(float));
        actualC[i] = (float*) malloc(jdim*sizeof(float));
    }
    for(i=0; i<kdim; i++) {
        B[i] = (float*) malloc(jdim*sizeof(float));
    }

    zero_init(idim, jdim, C);
    zero_init(idim, jdim, actualC);
    rand_init(idim, kdim, A);
    rand_init(kdim, jdim, B);

    // This is the standard matrix multiplication - do not adjust
    for(i = 0; i < idim; i++) {
        for(k = 0; k < kdim; k++) {
            for(j = 0; j < jdim; j++) {
                actualC[i][j] += A[i][k] * B[k][j];
            }
        }
    }

//    #pragma omp parallel
    nt = omp_get_num_threads();
    printf("Running with %d threads\n", nt);
    t1 = wctime();
//    #pragma omp parallel for
    for(i = 0; i < idim; i++) {
        for(k = 0; k < kdim; k++) {
            for(j = 0; j < jdim; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    t1 = wctime() - t1;

    // error calculation
    float err = 0.0;
    for(i = 0; i < idim; i++) {
        for(j = 0; j < jdim; j++) {
            err += (actualC[i][j] - C[i][j]);
        }
    }

    printf("Finished in %lf seconds\n", t1);
    t1 *= (1.e+09);
    nops = (float) 2 * idim * kdim * jdim;
    printf("Performance = %f GFLOPs\n",nops/t1);
    printf("Error: %f\n", err);
    return(0);
}