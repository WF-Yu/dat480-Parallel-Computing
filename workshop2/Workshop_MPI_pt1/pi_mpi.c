#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>

#define N 10000


int main(int argc, char ** argv) {
	//MPI_Init performs setup for the MPI program, including passing command-line arguments to all processes
	MPI_Init(&argc, &argv);
	
	//---- Question 1: Discover the number of processes and rank (ID) of each process 
	int size, rank;
	//TODO: Get the number of processes in variable "size"
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	//TODO: Get the rank of each current process in variable "rank"
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//----
	printf("MPI rank: %d \t MPI size: %d\n", rank, size);

	if (rank == 0)
		printf("Computing approximation to pi using N=%d\n", N);
	
	int i;
	double pi = 0.0;		//The final result
	double exact_pi = 0.0;		//Exact computation of pi, for comparison
	double partial_pi = 0.0; 	//Partial result calculated by each process
	

	//---- Question 2: Compute the loop boundaries "start", "end", for each process in SPMD style

	int partitions; //TODO: compute how many iteretions should be assigned to each process
	int start; 	//TODO: compute loop start
	int end; 	//TODO: compute loop end
	
	//TODO: Correct the loop boundaries for the last process 
    start = rank * (N / size) + 1;
    end = start + N / size;
    if (rank == size - 1) 
        end = N + 1;
    // printf("start: %d\tend: %d\n", start, end);
	//----


	//Main computation
	for (i = start ; i < end ; i++)
		partial_pi = partial_pi + 1.0 / (1.0 + pow( (((double)i - 0.5) / (double)N), 2.0));
	//----
	
	//---- Question 3: Implement communication to ensure that rank 0 collects the final result in variable "pi"
	//TODO: Use MPI_Send and MPI_Recv routines
	//TODO: How many sends will be issued, and how many receives? Which processes need to call them?
	//TODO: Make sure that rank 0 sums all partial sums

	int tag = 0; 			// FIXME: Message tag, will need initialization 
	MPI_Status status;		// MPI variable to collect the status of communication
	double partial_pi_to_recv;	// Temporary store of "partial_pi" received from other processes
			
	if (rank != 0) {
        //TODO: Implement communication - all ranks but rank 0
		MPI_Send(&partial_pi, sizeof(double), MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    }
		
	else {
		//TODO: Implement communication - rank 0
        //TODO: Rank 0 should also sum all partial sums of pi

        MPI_Recv(&partial_pi_to_recv, sizeof(double), MPI_DOUBLE, 1, tag, MPI_COMM_WORLD, &status);
        pi+=partial_pi_to_recv;
        MPI_Recv(&partial_pi_to_recv, sizeof(double), MPI_DOUBLE, 2, tag, MPI_COMM_WORLD, &status);
        pi+=partial_pi_to_recv;
        MPI_Recv(&partial_pi_to_recv, sizeof(double), MPI_DOUBLE, 3, tag, MPI_COMM_WORLD, &status);
        pi+=partial_pi_to_recv;
        
        pi += partial_pi;
	}	
	//----
    
	//Final result is collected at rank 0, which performs the printing
	if (rank == 0) {
		pi = pi * 4.0 / (double)N;
		exact_pi = 4.0 * atan(1.0);
		printf("Pi = %f, Exact pi = %f, Error = %f\n", pi, exact_pi, fabs(100.0 * (pi - exact_pi)/exact_pi));
	}

	//A call to MPI Finalize is necessary for the MPI program to exit without errors (with necessary cleanups)
	MPI_Finalize();
	return 0;
}
