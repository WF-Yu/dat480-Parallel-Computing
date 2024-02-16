#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>

#define N 1000


int main(int argc, char ** argv) {
	MPI_Init(&argc, &argv);

	//---- Initializations
	int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if (rank == 0)
		printf("Computing approximation to pi using N=%d\n", N);
	
	int i;
	double pi = 0.0;		//The final result
	double exact_pi = 0.0;		//Exact computation of pi, for comparison
	double partial_pi = 0.0; 	//Partial result calculated by each process
	
	//Identify neighbouring processes in the ring
	int left = (size + rank - 1) % size; 	//Neighbor at the left of the ring
	int right = (rank + 1) % size;	//Neighbor at the right of the ring

	//Compute the loop boundaries "start", "end", for each process in SPMD style

	int partitions;	//computer iterations per processes
	int start; 	//compute loop start
	int end; 	//compute loop end
	
	partitions = (N - 1) / size;
	start = 1 + rank * partitions;
	end = start + partitions;

	//The last process may receive fewer iterations 
	if (rank == size - 1)
		end = N + 1;
	//----


	//----Main computation
	for (i = start ; i < end ; i++)
		partial_pi = partial_pi + 1.0 / (1.0 + pow( (((double)i - 0.5) / (double)N), 2.0));
	//----

	//----Ring-style communication
	MPI_Status status;	//MPI variable to track the status of communication
	double passon = partial_pi;	//Variable to send to neighbour to the right
	double addon;			//Variable to receive from neighbour to the left

	MPI_Request sendRequest, recvRequest;
#ifdef BLOCKING
	//----Question 1: What is the problem with blocking communication? Can it cause a deadlock?
	//Blocking communication - Sends are issued first, receives next 
	for (i = 0 ; i < size  ; i++) {
		MPI_Send(&passon, 1, MPI_DOUBLE, right, i, MPI_COMM_WORLD);
		MPI_Recv(&addon, 1, MPI_DOUBLE, left, i, MPI_COMM_WORLD, &status);
		pi = pi + addon;
		passon = addon;
	}
	printf("[BLOCKING] Finished\n");
#elif BLOCKING_SYNCHRONOUS
	//----Question 2: What is the problem with synchronous communication? It does cause a deadlock.
	//Synchronous - Blocking communication - Note the usage of MPI_Ssend 
	for (i = 0 ; i < size  ; i++) {
		MPI_Ssend(&passon, 1, MPI_DOUBLE, right, i, MPI_COMM_WORLD);
		MPI_Recv(&addon, 1, MPI_DOUBLE, left, i, MPI_COMM_WORLD, &status);
		pi = pi + addon;
		passon = addon;
	}
    printf("[BLOCKING_SYNCHRONOUS] Finished\n");

#elif DEADLOCK_AVOIDING_1
	//----Question 3: Provide a solution that avoids deadlocks
	//TODO: Implement the same rotating ring communication.
	//TODO: What type of Send and Recv routines you will use? In which order?
    for (i = 0 ; i < size  ; i++) {
		MPI_Isend(&passon, 1, MPI_DOUBLE, right, i, MPI_COMM_WORLD, &sendRequest);
		MPI_Irecv(&addon, 1, MPI_DOUBLE, left, i, MPI_COMM_WORLD, &recvRequest);
		int flag = 0;
        while(!flag)
            MPI_Test(&recvRequest, &flag, &status);
        
        pi = pi + addon;
        passon = addon;
    }      
	
	//----
    printf("[DEADLOCK_AVOIDING 1] Finished\n");
#elif DEADLOCK_AVOIDING
    for (i = 0 ; i < size  ; i++) {
		MPI_Sendrecv(&passon, 1, MPI_DOUBLE, right, i, &addon, 1, MPI_DOUBLE, left, i, MPI_COMM_WORLD, &status);
		pi = pi + addon;
		passon = addon;
	}
	//----
    printf("[DEADLOCK_AVOIDING 2] Finished\n");
#endif
    
	//Final result is collected on all ranks, rank 0 performs the printing
	if (rank == 0) {
		pi = pi * 4.0 / (double)N;
		exact_pi = 4.0 * atan(1.0);
		printf("Pi = %f, Error = %f\n", pi, fabs(100.0 * (pi - exact_pi)/exact_pi));
	}

	MPI_Finalize();
	return 0;
}
