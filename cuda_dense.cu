#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
//#include "tools.h"

__global__ void matrixMultiply(float *A, float *B, float *C, int I, int J, int K) {
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    
    float tmp = 0.0;

    if(row < I && col < J) {
      for(int i=0; i<K; i++) {
        tmp += A[row*K+i] * B[i*J+col];
      }
      C[row*J+col] = tmp;
    }
}

double wctime() {
  // calculate wall time.
  struct timeval tv;
  gettimeofday(&tv, NULL); 
  return (tv.tv_sec + 1E-6 * tv.tv_usec);
}

int main(int argc, char *argv[]) {
    int idim = 200;
    int jdim = 400;
    int kdim = 1000;
    int i, j, k;
    double t1;
    float nops;
    float *A, *B, *C, *actualC, *Ag, *Bg, *Cg;
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

    printf("A matrix sample: \n");
    for(i=0; i<2; i++) {
      printf("[ ");
      for(k=0; k<10; k++) {
        printf("%f ", A[i*kdim+k]);
      }
      printf("]\n");
    }
    printf("\nB matrix sample: \n");
    for(k=0; k<2; k++) {
      printf("[ ");
      for(j=0; j<10; j++) {
        printf("%f ", B[k*jdim+j]);
      }
      printf("]\n");
    }

    // This is the standard matrix multiplication - do not adjust
    for(i = 0; i < idim; i++) {
        for(k = 0; k < kdim; k++) {
            for(j = 0; j < jdim; j++) {
                actualC[i*jdim+j] += A[i*kdim+k] * B[k*jdim+j];
            }
        }
    }

    printf("\nActualC matrix sample: \n");
    for(i=0; i<2; i++) {
      printf("[ ");
      for(j=0; j<10; j++) {
        printf("%f ", actualC[i*jdim+j]);
      }
      printf("]\n");
    }
/*
//    #pragma omp parallel
    nt = omp_get_num_threads();
    printf("Running with %d threads\n", nt);
    t1 = wctime();
//    #pragma omp parallel for
    for(i = 0; i < idim; i++) {
        for(k = 0; k < kdim; k++) {
            for(j = 0; j < jdim; j++) {
                C[i*jdim+j] += A[i*kdim+k] * B[k*jdim+j];
            }
        }
    }
    t1 = wctime() - t1;
*/
    cudaMalloc(&Ag, idim*kdim*sizeof(float));
    cudaMalloc(&Bg, kdim*jdim*sizeof(float));
    cudaMalloc(&Cg, idim*jdim*sizeof(float));
    //cudaMalloc(&I, sizeof(int));
    //cudaMalloc(&J, sizeof(int));
    //cudaMalloc(&K, sizeof(int));
    t1 = wctime();
    cudaMemcpy(Ag, A, idim*kdim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bg, B, kdim*jdim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Cg, C, idim*jdim*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(I, idim, sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(J, jdim, sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(K, kdim, sizeof(int), cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(jdim*idim, 2);
    dim3 blocksPerGrid(1, 1);
    if (idim * jdim > 512) {
      threadsPerBlock.x = 32;
      threadsPerBlock.y = 32;
      blocksPerGrid.x = ceil((double)jdim/(double)threadsPerBlock.x);
      blocksPerGrid.y = ceil((double)idim/(double)threadsPerBlock.y);
    }
    printf("threadsPerBlock: (%d, %d)\n", threadsPerBlock.x, threadsPerBlock.y);
    printf("blocksPerGrid:   (%d, %d)\n", blocksPerGrid.x, blocksPerGrid.y);
    //t1 = wctime();
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(Ag, Bg, Cg, idim, jdim, kdim);
    cudaDeviceSynchronize();
    //t1 = wctime() - t1;
    cudaError_t error = cudaGetLastError();
    if (error) {
      printf("CUDA error: %s \n", cudaGetErrorString(error));
      exit(1);
    }
    cudaMemcpy(C, Cg, idim*jdim*sizeof(float), cudaMemcpyDeviceToHost);
    t1 = wctime() - t1;
    // error calculation
    printf("\nC matrix sample: \n");
    for(i=0; i<2; i++) {
      printf("[ ");
      for(j=0; j<10; j++) {
        printf("%f ", C[i*jdim+j]);
      }
      printf("]\n");
    }
    float err = 0.0, t = 0.0;
    for(i = 0; i < idim; i++) {
        for(j = 0; j < jdim; j++) {
            err += ((actualC[i*jdim+j] - C[i*jdim+j]) * (actualC[i*jdim+j] - C[i*jdim+j]));
	    t += (actualC[i*jdim+j] * actualC[i*jdim+j]);
        }
    }
    err = sqrt(err/t);

    printf("Finished in %lf seconds\n", t1);
    t1 *= (1.e+09);
    nops = (float) 2 * idim * kdim * jdim;
    printf("Performance = %f GFLOPs\n",nops/t1);
    printf("Error: %f\n", err/((float)idim*jdim));
    cudaFree(Ag);
    cudaFree(Bg);
    cudaFree(Cg);
    //cudaFree(I);
    //cudaFree(J);
    //cudaFree(K);
    return(0);
}

