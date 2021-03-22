#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "tools.h"

int main(int argc, char *argv[]) {
    int idim = 200;
    int jdim = 400;
    int kdim = 1000;
    int i, j, k, p, nloc, counter;
    double t1 = 0.0;
    float nops, err;
    float *A, *B, *C, *actualC, *myA, *myC, *sendMe;
    MPI_Status status;
    A = (float*) malloc(idim*kdim*sizeof(float));
    B = (float*) malloc(kdim*jdim*sizeof(float));
    C = (float*) malloc(idim*jdim*sizeof(float));
    actualC = (float*) malloc(idim*jdim*sizeof(float));
   
    // Initialize matrices
    zero_init(idim, jdim, C);
    zero_init(idim, jdim, actualC);
    rand_init(idim, kdim, A);
    rand_init(kdim, jdim, B);

    // This is the standard matrix multiplication - do not adjust
    matrix_mult(idim, jdim, kdim, A, B, actualC);

//    #pragma omp parallel
    int myid, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if(myid == 0) {
    //printf("A matrix sample: \n");
    //print_sample(idim, kdim, A, 2, 10);
    //printf("B matrix sample: \n");
    //print_sample(idim, kdim, B, 2, 10);
    //printf("actualC matrix sample: \n");
    //print_sample(idim, jdim, actualC, 2, 10);
    //C = actualC;
    printf("Running with %d procs.\n", nprocs);
    t1 = wctime();
    nloc = (int) (idim + nprocs - 1) / nprocs;
    //printf("proc %d using nloc of %d.\n", myid, nloc);
    myA = (float*) malloc(nloc*kdim*sizeof(float));
    sendMe = (float*) malloc(nloc*kdim*sizeof(float));
    for(i = 0; i < nloc; i++) {
      for(k = 0; k < kdim; k++) {
        myA[i*kdim+k] = A[i*kdim+k];
      }
    }
    //printf("Starting to send to other procs:\n");
    MPI_Bcast(B, kdim*jdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //printf("nloc = %d.\n", nloc);
    for(p = 1; p < nprocs; p++) {
      counter = 0;
      //printf("Prepare proc %d.\n", p);
      for(i = nloc*p; (i < nloc*(p+1)) && (i < idim); i++) {
	j = i - nloc*p;
	//printf("Value of j = %d\n", j);
        for(k = 0; k < kdim; k++) {
          sendMe[j*kdim+k] = A[i*kdim+k];
	  counter++;
	}
      }
      //printf("Sending...\n");
      MPI_Send(sendMe, counter, MPI_FLOAT, p, p, MPI_COMM_WORLD);
      //printf("Sent data to proc %d.\n", p);
    }
      //printf("Calculate data...\n");
      myC = (float*) malloc(nloc*jdim*sizeof(float));
      matrix_mult(nloc, jdim, kdim, myA, B, myC);
      for(i=0; i<nloc; i++) {
        for(j = 0; j<jdim; j++) {
	  C[i*jdim+j] = myC[i*jdim+j];
	}
      }
      //MPI_Barrier(MPI_COMM_WORLD);
      //printf("Starting to recieve data...\n");
      for(p = 1; p < nprocs; p++) {
	MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	MPI_Get_count(&status, MPI_INT, &counter);
	//printf("found counts from proc %d using tag %d.\n", status.MPI_SOURCE, status.MPI_TAG);
        MPI_Recv(myC, counter, MPI_FLOAT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);
	//printf("recieved data from proc %d, in %d counts.\n", status.MPI_SOURCE, counter);
	counter = (int) counter / jdim;
	//printf("nloc = %d.\n", counter);
	//print_sample(counter, jdim, myC, 2, 10);
	for(i=0; i<counter; i++) {
          k = i + nloc * status.MPI_TAG;
          for(j=0; j<jdim; j++) {
            C[k*jdim+j] = myC[i*jdim+j];
	  }
	}
      }
    //MPI_Bcast(A, idim*kdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //MPI_Bcast(B, kdim*jdim, MPI_FLOAT, 100, MPI_COMM_WORLD);
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
    } else {
    //printf("I am here too.  From id: %d\n", myid);
    MPI_Bcast(B, kdim*jdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //printf("Proc %d recieved matrix B.\n", myid);
    nloc = (int) (idim + nprocs - 1) / nprocs;
    if(myid == (nprocs - 1))
      nloc = idim - nloc*(nprocs - 1);
    //printf("proc %d using nloc of %d.\n", myid, nloc);
    myA = (float*) malloc(nloc*kdim*sizeof(float));
    MPI_Recv(myA, nloc*kdim, MPI_FLOAT, 0, myid, MPI_COMM_WORLD, &status);
    //printf("Proc %d recieved part of A.\n", myid);
    //MPI_Bcast(A, idim*kdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //MPI_Bcast(B, kdim*jdim, MPI_FLOAT, 100, MPI_COMM_WORLD);
    myC = (float*) malloc(nloc*jdim*sizeof(float));
    matrix_mult(nloc, jdim, kdim, myA, B, myC);
    MPI_Send(myC, nloc*jdim, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
    //printf("Proc %d sent part of C.\n", myid);
    }
/*
    matrix_mult(nloc, jdim, kdim, myA, B, myC);

    if(myid == 0) {
      // error calculation
      err = error_calc(idim, jdim, actualC, C);

      t1 = wctime() - t1;
      printf("Finished in %lf seconds\n", t1);
      t1 *= (1.e+09);
      nops = (float) 2 * idim * kdim * jdim;
      printf("Performance = %f GFLOPs\n", nops/t1);
      printf("Error: %f\n", err);
    } else {
      1;
    }
*/
    MPI_Finalize();
    return(0);
}
