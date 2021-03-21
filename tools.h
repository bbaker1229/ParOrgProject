#ifndef SRC_TOOLS_H_
#define SRC_TOOLS_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <unistd.h>

double wctime();

void print_matrix(int rdim, int cdim, float *A);

void zero_init(int rdim, int cdim, float *A);

void rand_init(int rdim, int cdim, float *A);

int make_sparse_percent(float per, int rdim, int cdim, float *A);

void make_sparse_matrix(int rdim, int cdim, int *rowval, int *colval, float *value, float *A);

void matrix_mult(int rdim, int cdim, int kdim, float *A, float *B, float *C);

float error_calc(int rdim, int cdim, float *A, float *B);

void print_sample(int rdim, int cdim, float *A, int rsize, int csize);

#endif
