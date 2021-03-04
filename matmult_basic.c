#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double wctime();

int main(int argc, char *argv[]) {
    int idim = 2000;
    int jdim = 4000;
    int kdim = 10000;
    int i, j, k, nt;
    float **A, **B, **C;
    A = (float**) malloc(idim*sizeof(float*));
    B = (float**) malloc(kdim*sizeof(float*));
    C = (float**) malloc(idim*sizeof(float*));
    for(i=0; i<idim; i++) {
        A[i] = (float*) malloc(kdim*sizeof(float));
        C[i] = (float*) malloc(jdim*sizeof(float));
    }
    for(i=0; i<kdim; i++) {
        B[i] = (float*) malloc(jdim*sizeof(float));
    }
    float value;
    double t1;
    float nops;

    for(i = 0; i < idim; i++) {
        for(j = 0; j < jdim; j++) {
            C[i][j] = 0.0;
        }
    }

    for(i = 0; i < idim; i++) {
        for(k = 0; k < kdim; k++) {
            value = (float)rand() / (float)RAND_MAX;
            A[i][k] = value;
        }
    }

    for(k = 0; k < kdim; k++) {
        for(j = 0; j < jdim; j++) {
            value = (float)rand() / (float)RAND_MAX;
            B[k][j] = value;
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

    printf("Finished in %lf seconds\n", t1);
    t1 *= (1.e+09);
    nops = (float) 2 * idim * kdim * jdim;
    printf("Performance = %f GFLOPs\n",nops/t1);
    return(0);
}