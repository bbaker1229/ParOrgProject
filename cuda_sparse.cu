#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>

__global__ void sparseMultiply(int *rowvec, int *colvec, float *valvec, float *B, float *C, int I, int J, int K) {
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    
    float tmp = 0.0;

    if(row < I-1 && col < J) {
      for(int i=rowvec[row]; i<rowvec[row+1]; i++) {
        tmp += valvec[i] * B[colvec[i]*J+col];
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
int make_sparse_percent(float per, int rdim, int cdim, float *A);
void make_sparse_matrix(int rdim, int cdim, int *rowval, int *colval, float *value, int *rowval_size, int *val_size, float *A);

int main(int argc, char *argv[]) {
    int idim = 200;
    int jdim = 400;
    int kdim = 1000;
    int rowlen, vallen;
    long int newdim;
    double t1;
    float nops, per, err;
    float *A, *B, *C, *actualC, *Bg, *Cg, *valg;
    int *rowg, *colg;
    A = (float*) malloc(idim*kdim*sizeof(float));
    B = (float*) malloc(kdim*jdim*sizeof(float));
    C = (float*) malloc(idim*jdim*sizeof(float));
    actualC = (float*) malloc(idim*jdim*sizeof(float));

    // Initialize matrices
    zero_init(idim, jdim, C);
    zero_init(idim, jdim, actualC);
    rand_init(idim, kdim, A);
    rand_init(kdim, jdim, B);

    if(argc == 1)
      per = 0.3;
    else
      per = atof(argv[1]);
    printf("Running with %0.1f%% sparsity\n", per * 100);
    newdim = make_sparse_percent(per, idim, kdim, A);
    //printf("A matrix sample: \n");
    //print_sample(idim, kdim, A, 2, 10);
    //printf("B matrix sample: \n");
    //print_sample(kdim, jdim, B, 2, 10);
    // This is the standard matrix multiplication - do not adjust
    matrix_mult(idim, jdim, kdim, A, B, actualC);
    //printf("actualC matrix sample: \n");
    //print_sample(idim, jdim, actualC, 2, 10);

    int *rowval, *colval;
    float *value;
    rowval = (int*) malloc(newdim*sizeof(int));
    colval = (int*) malloc(newdim*sizeof(int));
    value = (float*) malloc(newdim*sizeof(float));
    make_sparse_matrix(idim, kdim, rowval, colval, value, &rowlen, &vallen, A);

    cudaMalloc(&rowg, newdim*sizeof(int));
    cudaMalloc(&colg, newdim*sizeof(int));
    cudaMalloc(&valg, newdim*sizeof(float));
    cudaMalloc(&Bg, kdim*jdim*sizeof(float));
    cudaMalloc(&Cg, idim*jdim*sizeof(float));

    t1 = wctime();
    cudaMemcpy(rowg, rowval, newdim*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(colg, colval, newdim*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(valg, value, newdim*sizeof(float), cudaMemcpyHostToDevice);
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
    sparseMultiply<<<blocksPerGrid, threadsPerBlock>>>(rowg, colg, valg, Bg, Cg, rowlen, jdim, kdim);
    cudaDeviceSynchronize();
    //t1 = wctime() - t1;
    cudaError_t error = cudaGetLastError();
    if (error) {
      printf("CUDA error: %s \n", cudaGetErrorString(error));
      exit(1);
    }
    cudaMemcpy(C, Cg, idim*jdim*sizeof(float), cudaMemcpyDeviceToHost);
    t1 = wctime() - t1;

    //printf("C matrix sample: \n");
    //print_sample(idim, jdim, C, 2, 10);

    // error calculation
    err = error_calc(idim, jdim, actualC, C);

    printf("Finished in %lf seconds\n", t1);
    t1 *= (1.e+09);
    nops = (float) 2 * idim * kdim * jdim;
    printf("Performance = %f GFLOPs\n",nops/t1);
    printf("Error: %f\n", err);
    cudaFree(rowg);
    cudaFree(colg);
    cudaFree(valg);
    cudaFree(Bg);
    cudaFree(Cg);
    return(0);
}

double wctime() {
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

int make_sparse_percent(float per, int rdim, int cdim, float *A) {
    long int maxnums, cnt, check;
    maxnums = (long int) (per * (float)rdim * (float)cdim);
    int *vals;
    vals = (int*) malloc(maxnums*sizeof(int));
    for(long int i = 0; i < maxnums; i++) {
        vals[i] = -1;
    }
    cnt = 0;
    while(cnt < maxnums) {
        long int num = rand() % (rdim * cdim + 1);
        check = 0;
        for(long int i = 0; i < cnt; i++) {
            if(vals[i] == num) {
                check = 1;
                break;
            }
        }
        if(check == 0) {
            vals[cnt] = num;
            cnt++;
        }
    }
    for(int i=0; i < rdim; i++) {
        for(int j=0; j < cdim; j++) {
            for(int k=0; k < cnt; k++) {
                if((i+1)*(j+1) == vals[k]) {
                    A[i*cdim+j] = 0.0;
                }
            }
        }
    }
    return rdim * cdim - maxnums + 1;
}

void make_sparse_matrix(int rdim, int cdim, int *rowval, int *colval, float *value, int *rowval_size, int *val_size, float *A) {
    int cnt = 0, cnt1;
    rowval[0] = 0;
    cnt1 = 1;
    for(int i=0; i < rdim; i++) {
        for(int j=0; j < cdim; j++) {
            if(A[i*cdim+j] != 0.0) {
                rowval[cnt] = i;
                colval[cnt] = j;
                value[cnt] = A[i*cdim+j];
                cnt++;
            }
        }
        rowval[cnt1] = cnt;
        cnt1++;
    }
    *val_size = cnt;
    *rowval_size = cnt1;
}

