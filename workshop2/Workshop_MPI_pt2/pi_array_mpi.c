#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <sys/time.h>

#define N 10000

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
	//----

	//---- Initialization and distribution of the "random_seq" array
	
	//Rank 0 initializes the array "random_seq"
	int * random_seq;
	if (rank == 0) {
		//Memory alignment to produce equal chunks
		if (N % size == 0)
			random_seq = malloc(N  * sizeof(int));
		else
			random_seq = malloc( (1 + N/size) * size * sizeof(int)); 
		for (i = 0 ; i < N ; i++) 
			random_seq[i] = rand() % N;
	}

	//All ranks shall receive a chunk of the "random_seq" array of size "chunk_size", to store in "random_seq_local"
	int * random_seq_local;
	int chunk_size;
	//The size of "random_seq_local" is computed in "chunk_size"
	if (N % size == 0)
		chunk_size = N / size;
	else
		chunk_size = (N / size) + 1;
	//The local random sequence, "random_seq_local" is allocated on all processes
	random_seq_local = malloc(chunk_size * sizeof(int));

	//----Working set distrubtion: Rank 0 distributes the initial array "random_seq"
#ifdef P2P
	MPI_Status status;
	if (rank == 0) {
		//Rank 0 to send chunks to all processes
		for (i = 1 ; i < size ; i++)
			MPI_Send(&random_seq[i * chunk_size], chunk_size, MPI_INT, i, 55, MPI_COMM_WORLD);
		//Rank 0 to copy its own chunk
		memcpy(random_seq_local, random_seq, chunk_size * sizeof(int));
	}
	else {
		//All ranks to receive chunks from rank 0
		MPI_Recv(random_seq_local, chunk_size, MPI_INT, 0, 55, MPI_COMM_WORLD, &status);
	}
#elif COLL || COLL_OPT
	//----Question 1: Use the appropriate MPI collective to distribute chunks of the initial array "random_seq"
	//TODO: All processes to receive a chunk in the array "random_seq_local"
    MPI_Scatter(random_seq, chunk_size, MPI_INT, random_seq_local, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    
	//----
#endif
	
	//Compute new loop boundaries - take care of the last process with rank=size-1
	int start = 0;
	int end = chunk_size;
    if (rank == size - 1) 
		if (N % size != 0) 
			end = chunk_size - (size * chunk_size - N); 		
	//----
	
	//Main computation: Each process computes "partial_pi"
	for (i = start ; i < end ; i++)
		partial_pi = partial_pi + 1.0 / (1.0 + pow( (((double)random_seq_local[i] - 0.5) / (double)N), 2.0));
	//----


	//Rank 0 collects "partial_pi" and computes "pi" 
#ifdef P2P
	double partial_pi_to_recv;	
	if (rank != 0) 
		//All ranks but rank 0 to send their partial sums
		MPI_Send(&partial_pi, 1, MPI_DOUBLE, 0, 55, MPI_COMM_WORLD);
	else {
		//Rank 0 to receive partial sums and add them to pi
		for (i = 1 ; i < size ; i++) {
			MPI_Recv(&partial_pi_to_recv, 1, MPI_DOUBLE, i, 55, MPI_COMM_WORLD, &status);
			pi = pi + partial_pi_to_recv;
		}
		//Rank 0 to add its own sum to pi
		pi = pi + partial_pi;
	}	
#elif COLL 
	//----Question 2: Use the appropriate MPI collective to reduce partial sums "partial_pi" in "pi" on rank 0
	//TODO: Rank 0 to receive a sum of all partial sums
	MPI_Reduce(&partial_pi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	//----
#endif

	//Rank 0 distributes final result "pi" on all processes
#ifdef P2P
	if (rank != 0) {
		//All ranks but rank 0 to receive value of "pi" from rank 0
		MPI_Recv(&pi, 1, MPI_DOUBLE, 0, 55, MPI_COMM_WORLD, &status);
	}
	else {
		//Rank 0 to send final result "pi" to all other processes 
		for (i = 1 ; i < size ; i++) 
			MPI_Send(&pi, 1, MPI_DOUBLE, i, 55, MPI_COMM_WORLD);
	}
#elif COLL
	//----Question 3: Use the appropriate MPI collective to distribute the full sum from rank 0 to all processes
	//TODO: All processes to receive the full sum in "pi"
	MPI_Bcast(&pi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//----
#endif

#ifdef COLL_OPT
	//----Question 4: Use a single MPI collective to optimize the previous two collective calls (Questions 2 and 3)
	//TODO: All processes to receive the full sum in "pi" from partial sums in "partial_pi"
	MPI_Allreduce(&partial_pi, &pi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	//----
#endif
	//All processes print the final result
	pi = pi * 4.0 / (double)N;
	exact_pi = 4.0 * atan(1.0);
	printf("Rank = %d, Pi = %f, Exact pi = %f, Error = %f\n", rank, pi, exact_pi, fabs(100.0 * (pi - exact_pi)/exact_pi));


	if (rank == 0)
		free(random_seq);
	free(random_seq_local);
	MPI_Finalize();
	return 0;
}
