#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char ** argv) {
	int size, rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//----Question 1: Create new communicators, for even and odd processes
	int color = rank % 2; 		//TODO: Assign a color (even/odd) to each process, according to its rank

	MPI_Comm new_comm;	//TODO: Create a new communicator based on the color of each process
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);
	//----
	

	//----Question 2: A process with a new rank 0 on each communicator should now broadcast its color 
	int new_rank;			//TODO: Find the rank of each process in the new communicator it belongs to
    MPI_Comm_rank(new_comm, &new_rank);
	int communicator_color; 	//TODO: The color of new rank 0 to be broadcasted
	MPI_Comm_size(new_comm, &communicator_color);
	if (new_rank == 0)
		communicator_color = color;
	//TODO: Broadcast the color to all processes in the new communicator
    MPI_Bcast(&communicator_color, 1, MPI_INT, 0, new_comm);
    //----Question 4: using point-to-point communication instead of multiple communicators
    if (new_rank == 0) {
        // for()
            // MPI_send();
    }
    else {
        // MPI_recv();
    }
    
    //----Question 5: using collective communication, without using multiple communicators
    // MPI_Bcast(&result[color],,,,MPI_COMM_WORLD)
    // color received = result[color]

	//----
	
	//----Question 3: Each process should print its rank in the "world" and in the new communicator and the received color
	printf("new rank:%d, old rank:%d, received color:%d \n", new_rank, rank, communicator_color);
	//----
	MPI_Finalize();
	return 0;
}
