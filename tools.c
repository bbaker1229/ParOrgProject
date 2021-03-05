#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <unistd.h>
#include "tools.h"

double wctime() 
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + 1E-6 * tv.tv_usec);
}

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
