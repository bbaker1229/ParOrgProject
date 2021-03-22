#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "tools.h"

int main(int argc, char *argv[]) {
    int idim = 200;
    int jdim = 400;
    int kdim = 1000;
    int i, j, k, rowlen, vallen;
    long int newdim;
    float nops, per, err;
    double t1;
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

    per = 0.3;
    printf("Running with %0.1f%% sparsity\n", per * 100);
    newdim = make_sparse_percent(per, idim, kdim, A);
    //printf("A matrix sample: \n");
    //print_matrix(idim, kdim, A);
    //printf("B matrix sample: \n");
    //print_matrix(kdim, jdim, B);
    // This is the standard matrix multiplication - do not adjust
    matrix_mult(idim, jdim, kdim, A, B, actualC);
    //printf("actualC matrix sample: \n");
    //print_matrix(idim, jdim, actualC);

    int *rowval, *colval;
    float *value;
    rowval = (int*) malloc(newdim*sizeof(int));
    colval = (int*) malloc(newdim*sizeof(int));
    value = (float*) malloc(newdim*sizeof(float));
    make_sparse_matrix(idim, kdim, rowval, colval, value, &rowlen, &vallen, A);

    //printf("Values: \n");
    //printf("[ ");
    //for(i=0; i<vallen; i++) {
    //  printf("%f ", value[i]);
    //}
    //printf("]\n");
    //printf("Column Vector: \n");
    //printf("[ ");
    //for(i=0; i<vallen; i++) {
    //  printf("%d ", colval[i]);
    //}
    //printf("]\n");
    //printf("Row Vector: \n");
    //printf("[ ");
    //for(i=0; i<rowlen; i++) {
    //  printf("%d ", rowval[i]);
    //}
    //printf("]\n");

    t1 = wctime();
    for(i=0; i<rowlen-1; i++) {
      for(j=0; j<jdim; j++) {
        for(k=rowval[i]; k<rowval[i+1]; k++) {
	  C[i*jdim+j] += value[k] * B[colval[k]*jdim+j];
	}
      }
    }
    /*for(i=0; i < newdim; i++) {
        for(j=0; j < jdim; j++) {
            C[rowval[i]*jdim+j] += value[i] * B[colval[i]*jdim+j];
        }
    }*/
    t1 = wctime() - t1;

    //printf("C matrix sample: \n");
    //print_matrix(idim, jdim, C);

    // error calculation
    err = error_calc(idim, jdim, actualC, C);

    printf("Finished in %lf seconds\n", t1);
    t1 *= (1.e+09);
    nops = (float) 2 * idim * kdim * jdim;
    printf("Performance = %f GFLOPs\n",nops/t1);
    printf("Error: %f\n", err);
    return(0);
}
