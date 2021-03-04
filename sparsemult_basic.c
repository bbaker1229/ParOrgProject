#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double wctime();

void print_matrix(int rdim, int cdim, float **A) {
    for(int i = 0; i < rdim; i++) {
        printf("[ ");
        for(int j = 0; j < cdim; j++) {
            printf("%1.4f ", A[i][j]);
        }
        printf("]\n");
    }
}

void zero_init(int rdim, int cdim, float **A) {
    for(int i = 0; i < rdim; i++) {
        for(int j = 0; j < cdim; j++) {
            A[i][j] = 0.0;
        }
    }
}

void rand_init(int rdim, int cdim, float **A) {
    for(int i = 0; i < rdim; i++) {
        for(int j = 0; j < cdim; j++) {
            A[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }
}

int make_sparse_percent(float per, int rdim, int cdim, float **A) {
    long int maxnums, cnt, check;
    maxnums = (long int) (per * (float)rdim * (float)cdim);
    //printf("%ld\n",maxnums);
    //int vals[maxnums];
    int *vals;
    vals = (int*) malloc(maxnums*sizeof(int));
    for(long int i = 0; i < maxnums; i++) {
        vals[i] = -1;
    }
    //printf("cnt = %d\n", maxnums);
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
    /*
    for(int i = 0; i < cnt; i++) {
        //int num = rand() % (idim * kdim + 1);
        printf("%d ", vals[i]);
    }
    printf("\n");
    print_matrix(rdim, cdim, A);
    printf("\n");
    //print_matrix(kdim, jdim, B);
    //printf("\n");
    */
    for(int i=0; i < rdim; i++) {
        for(int j=0; j < cdim; j++) {
            for(int k=0; k < cnt; k++) {
                if((i+1)*(j+1) == vals[k]) {
                    A[i][j] = 0.0;
                }
            }
        }
    }
    return rdim * cdim - maxnums + 1;
}

void make_sparse_matrix(int rdim, int cdim, int *rowval, int *colval, float *value, float **A) {
    int cnt = 0;
    for(int i=0; i < rdim; i++) {
        for(int j=0; j < cdim; j++) {
            if(A[i][j] != 0.0) {
                rowval[cnt] = i;
                colval[cnt] = j;
                value[cnt] = A[i][j];
                cnt++;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int idim = 200;
    int jdim = 400;
    int kdim = 1000;
    int i, j, k, nt, cnt, maxnums, check;
    long int newdim;
    float nops;
    double t1;
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

    zero_init(idim, jdim, C);
    rand_init(idim, kdim, A);
    rand_init(kdim, jdim, B);

    newdim = make_sparse_percent(0.3, idim, kdim, A);
    //int rowval[newdim], colval[newdim];
    //float value[newdim];
    int *rowval, *colval;
    float *value;
    rowval = (int*) malloc(newdim*sizeof(int));
    colval = (int*) malloc(newdim*sizeof(int));
    value = (float*) malloc(newdim*sizeof(float));
    make_sparse_matrix(idim, kdim, rowval, colval, value, A);

    /*for(i = 0; i < newdim; i++) {
        printf("(%d %d) %f\n", rowval[i], colval[i], value[i]);
    }
    printf("Matrix A:\n");
    print_matrix(idim, kdim, A);
    printf("\n");
    printf("Matrix B:\n");
    print_matrix(kdim, jdim, B);
    printf("\n");
    //Fix this part.  It doesn't work yet.
    printf("\n");*/

    t1 = wctime();
    for(i=0; i < newdim; i++) {
        for(j=0; j < jdim; j++) {
            C[rowval[i]][j] += value[i] * B[colval[i]][j];
        }
    }
    t1 = wctime() - t1;
    /*printf("Matrix C (sparse):\n");
print_matrix(idim, jdim, C);
printf("\n");
zero_init(idim, jdim, C);

    t1 = wctime();
    for(i = 0; i < idim; i++) {
        for(k = 0; k < kdim; k++) {
            for(j = 0; j < jdim; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    t1 = wctime() - t1;
    printf("Matrix C (dense):\n");
    print_matrix(idim, jdim, C);
    */
    printf("Finished in %lf seconds\n", t1);
    t1 *= (1.e+09);
    nops = (float) 2 * idim * kdim * jdim;
    printf("Performance = %f GFLOPs\n",nops/t1);
    return(0);
}