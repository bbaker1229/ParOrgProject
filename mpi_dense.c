#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "tools.h"

int main(int argc, char *argv[]) {
    int idim = 200;
    int jdim = 400;
    int kdim = 1000;
    int i, j, k;
    double t1 = 0.0;
    float nops;
    float *A, *B, *C, *actualC;
    A = (float*) malloc(idim*kdim*sizeof(float));
    B = (float*) malloc(kdim*jdim*sizeof(float));
    C = (float*) malloc(idim*jdim*sizeof(float));
    actualC = (float*) malloc(idim*jdim*sizeof(float));
    
    for(i=0; i<idim; i++) {
      for(j=0; j<jdim; j++) {
        C[i*jdim+j] = 0.0;
        actualC[i*jdim+j] = 0.0;
      }
      for(k=0; k<kdim; k++) {
        A[i*kdim+k] = 0.0;
      }
    }
    for(k=0; k<kdim; k++) {
      for(j=0; j<jdim; j++) {
        B[k*jdim+j] = 0.0;
      }
    }

    for(i=0; i<idim; i++) {
      for(k=0; k<kdim; k++) {
        A[i*kdim+k] = (float)rand() / (float)RAND_MAX;
      }
    }
    for(k=0; k<kdim; k++) {
      for(j=0; j<jdim; j++) {
        B[k*jdim+j] = (float)rand() / (float)RAND_MAX;
      }
    }

    // This is the standard matrix multiplication - do not adjust
    for(i = 0; i < idim; i++) {
        for(k = 0; k < kdim; k++) {
            for(j = 0; j < jdim; j++) {
                actualC[i*jdim+j] += A[i*kdim+k] * B[k*jdim+j];
            }
        }
    }

//    #pragma omp parallel
    int myid, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    /*nt = omp_get_num_threads();
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
*/

    // error calculation
    float err = 0.0;
    for(i = 0; i < idim; i++) {
        for(j = 0; j < jdim; j++) {
            err += (actualC[i*jdim+j] - C[i*jdim+j]);
        }
    }

    printf("Finished in %lf seconds\n", t1);
    t1 *= (1.e+09);
    nops = (float) 2 * idim * kdim * jdim;
    printf("Performance = %f GFLOPs\n",nops/t1);
    printf("Error: %f\n", err);
    MPI_Finalize();
    return(0);
}
