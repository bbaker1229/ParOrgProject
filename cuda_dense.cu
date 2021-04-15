#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>

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

double wctime();
void zero_init(int rdim, int cdim, float *A);
void rand_init(int rdim, int cdim, float *A);
void matrix_mult(int rdim, int cdim, int kdim, float *A, float *B, float *C);
float error_calc(int rdim, int cdim, float *A, float *B);
void print_sample(int rdim, int cdim, float *A, int rsize, int csize);

int main(int argc, char *argv[]) {
    int idim = 1000;
    int jdim = 1000;
    int kdim = 1000;
    double t1, times[20];
    float nops, err;
    float *A, *B, *C, *actualC, *Ag, *Bg, *Cg;
    A = (float*) malloc(idim*kdim*sizeof(float));
    B = (float*) malloc(kdim*jdim*sizeof(float));
    C = (float*) malloc(idim*jdim*sizeof(float));
    actualC = (float*) malloc(idim*jdim*sizeof(float));

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

    //printf("ActualC matrix sample: \n");
    //print_sample(idim, jdim, actualC, 2, 10);

    cudaMalloc(&Ag, idim*kdim*sizeof(float));
    cudaMalloc(&Bg, kdim*jdim*sizeof(float));
    cudaMalloc(&Cg, idim*jdim*sizeof(float));
    
    for(int loop_cnt = 0; loop_cnt < 20; loop_cnt++) {
    t1 = wctime();
    cudaMemcpy(Ag, A, idim*kdim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bg, B, kdim*jdim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Cg, C, idim*jdim*sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(jdim*idim, 2);
    dim3 blocksPerGrid(1, 1);
    if (idim * jdim > 512) {
      threadsPerBlock.x = 32;
      threadsPerBlock.y = 32;
      blocksPerGrid.x = ceil((double)jdim/(double)threadsPerBlock.x);
      blocksPerGrid.y = ceil((double)idim/(double)threadsPerBlock.y);
    }
    //printf("threadsPerBlock: (%d, %d)\n", threadsPerBlock.x, threadsPerBlock.y);
    //printf("blocksPerGrid:   (%d, %d)\n", blocksPerGrid.x, blocksPerGrid.y);
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
    times[loop_cnt] = t1;
    if(loop_cnt != 19)
      zero_init(idim, jdim, C);
    }
    
    //printf("C matrix sample: \n");
    //print_sample(idim, jdim, C, 2, 10);
    
    // error calculation
    err = error_calc(idim, jdim, actualC, C);

    t1 = 0.0;
    for(int i=0; i < 20; i++)
	    t1 += times[i];
    t1 /= (float) 20;
    printf("Finished in %lf seconds\n", t1);
    t1 *= (1.e+09);
    nops = (float) 2 * idim * kdim * jdim;
    printf("Performance = %f GFLOPs\n",nops/t1);
    printf("Error: %f\n", err/((float)idim*jdim));
    cudaFree(Ag);
    cudaFree(Bg);
    cudaFree(Cg);
    return(0);
}

double wctime() {
  // calculate wall time.
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + 1E-6 * tv.tv_usec);
}

void zero_init(int rdim, int cdim, float *A) {
    for(int i = 0; i < rdim; i++) {
        for(int j = 0; j < cdim; j++) {
            A[i*cdim+j] = 0.0;
        }
    }
}

void rand_init(int rdim, int cdim, float *A) {
    for(int i = 0; i < rdim; i++) {
        for(int j = 0; j < cdim; j++) {
            A[i*cdim+j] = (float)rand() / (float)RAND_MAX;
        }
    }
}

void matrix_mult(int rdim, int cdim, int kdim, float *A, float *B, float *C) {
    int i, j, k;
    for(i = 0; i < rdim; i++)
      for(k = 0; k < kdim; k++)
        for(j = 0; j < cdim; j++)
          C[i*cdim+j] += A[i*kdim+k] * B[k*cdim+j];
}

float error_calc(int rdim, int cdim, float *A, float *B) {
    int i, j;
    float err = 0.0, t = 0.0;
    for(i = 0; i < rdim; i++) {
      for(j = 0; j < cdim; j++) {
        err += ((A[i*cdim+j] - B[i*cdim+j]) * (A[i*cdim+j] - B[i*cdim+j]));
        t += (A[i*cdim+j] * A[i*cdim+j]);
      }
    }
    return(sqrt(err/t));
}

void print_sample(int rdim, int cdim, float *A, int rsize, int csize) {
    int i, j;
    for(i = 0; i < rsize; i++) {
      printf("[ ");
      for(j = 0; j < csize; j++) {
        printf("%f ", A[i*cdim+j]);
      }
      printf("]\n");
    }
}

