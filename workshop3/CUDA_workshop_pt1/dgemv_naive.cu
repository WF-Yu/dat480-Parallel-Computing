#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

//Problem size M x N
#define M_ 10000
#define N_ 10000

//A sample dimension for the block (number of threads per block)
#define BLOCK_DIM 1024	

#define timetosol(ts, tf) (tf.tv_sec - ts.tv_sec) + (tf.tv_usec - ts.tv_usec) * 1e-6

//Function getGPUProperties
//Discovers available GPUs and prints useful attributes
void getGPUProperties() {
	//Discover GPU attributes
	cudaError_t err;
	int devices;	//In case there are more than one GPUs
	cudaDeviceProp prop;
	err = cudaGetDeviceCount(&devices);
	if (!err) {
		for (int i = 0 ; i < devices ; i++) {
			printf("CUDA Device - ID %d\n", i);
			err = cudaGetDeviceProperties(&prop, i);
			if (!err) {
				printf("Max threads per block: 		%d\n", prop.maxThreadsPerBlock);
				printf("Max block dimensions: 		(%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
				printf("Max grid dimensions: 		(%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
				printf("Shared memory per block: 	%.2lfKB\n", prop.sharedMemPerBlock/1024.);
			}
			printf("\n");
		}
	}
	else {
		fprintf(stderr,"Error finding available GPUs, now exiting\n");
		exit(-1);
	}
}

//Function checkCorrectness
//Compares CPU result with GPU result, returns 0 if the result is correct
int checkCorrectness(float * sol_cpu, float * sol_gpu, int M) {
	int i;
	for (i = 0 ; i < M ; i++)
		if (fabs(sol_cpu[i] - sol_gpu[i])>1e-3) {
			printf("FAIL -- at %d, %f %f\n", i, sol_cpu[i], sol_gpu[i]);
			return 1;
		}
	return 0;
}

//Function cpuDGEMV
//Computes Matrix-Vector Multiplication A * x = b on the CPU 
//Input A: array M x N 
//Input x: vector N x 1 
//Output b: vector M x 1 
void cpuDGEMV(float * A, float * x, float * b, int M, int N) {
	int i, j;
	for (i = 0 ; i< M ; i++) {
		b[i] = 0.;
		for (j = 0 ; j < N ; j++)
			b[i] += A[i * N + j] * x[j]; 
	}
}

//TODO - Question 2: Function cudaDGEMV
//------Computes Matrix-Vector Multiplication A * x = b on the GPU
//------Input dev_A: array M x N 
//------Input dev_x: vector N x 1 
//------Output dev_b: vector M x 1 

__global__ void cudaDGEMV(float * A, float * x, float * b, int M, int N) {
	//TODO: Fill in this function to perform matrix-vector multiplication on the GPU
	//TODO: Give a "global" identifier "tid" to each thread, according to its position in the grid and block
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//TODO: Fill in the computations
    if (tid < M) {
        b[tid] = 0.0;
        for (int k = 0; k < N; k++) {
            b[tid] += A[tid*N + k] * x[k];
            
        }
    }
}

int main(int argc, char ** argv) {
    printf("hi");
	//----Problem input M x N----//
	int M = (argc >= 3) ? atoi(argv[1]) : M_;	//Read from input or default
	int N = (argc >= 3) ? atoi(argv[2]) : N_; 
	//---------------------------//


	//----Variables declaration----//
	struct timeval ts, tf;

	int i, j;

	// cudaError_t err;

	//Block & Grid dimensions for the GPU
	unsigned int blockX, blockY, blockZ;
	unsigned int gridX, gridY, gridZ;
	dim3 ThreadsPerBlock;
	dim3 numBlock;
	
	//Matrix A, vectors b, x, to be allocated on the CPU
	float * A;
       	float * x; 
	float * b;
	//Matrix dev_A, vectors dev_b, dev_x, to be allocated on the GPU
	float * dev_A;
	float * dev_x; 
	float * dev_b;
	//Helper vector to store CPU solution for correctness checks
	float * sol;
	//-----------------------------//

	//----Query GPU properties----//
	getGPUProperties();
	//----------------------------//

	//----CPU allocations and initialization----//
	A = (float*)malloc(M * N * sizeof(float)); 	//Matrix A (size M x N)
	x = (float*)malloc(N * sizeof(float));		//Vector x (size N)
	b = (float*)malloc(M * sizeof(float));		//Vector y (size M)

	for (i = 0 ; i < M ; i++) { 				
		for (j = 0 ; j < N ; j++) 
			A[i * N + j] = (rand() % 4 + 1)*0.1;	//Initilize A
	}	
	for (i = 0 ; i < N ; i++)
		x[i] = (rand()%10 + 1) * 0.01;			//Initialize x

	sol = (float*)malloc(M * sizeof(float));
	//------------------------------------------//

	//----DGEMV A * x = b on CPU - Reference run----//
	printf("Running DGEMV with size %d x %d on the CPU - Reference version\n", M, N);
	gettimeofday(&ts, NULL);
	
	cpuDGEMV(A, x, sol, M, N);
	
	gettimeofday(&tf, NULL);
	printf("Time: %.5lf(s)\n", timetosol(ts, tf));
	//------------------------------------//


	//----GPU allocations----//
	//TODO: Question 1 - Allocate matrix dev_A, and vectors dev_x, dev_b on the GPU
		//TODO: Allocate dev_A
		//TODO: Allocate dev_x
		//TODO: Allocate dev_b	
    cudaMalloc((void **)&dev_A, M * N * sizeof(float));
    cudaMalloc((void **)&dev_x, N * sizeof(float));
    cudaMalloc((void **)&dev_b, M * sizeof(float));

	//----------------------//

	//----Perform CPU to GPU transfers----//
	//TODO: Question 1 - Copy the input matrix A and input vector x to the GPU
		//TODO: Copy A to dev_A
		//TODO: Copy x to dev_x
    cudaMemcpy(dev_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

	//------------------------------------//


	//----A "naive" DGEMV A * x = b on GPU----//
	//TODO - Question 3: Select the block dimensions (threads per block)
	//Assume a 1D block
	blockX = 128;		//TODO: Select the number of blocks
	blockY = 1;
    blockZ = 1;
	ThreadsPerBlock = {blockX, blockY, blockZ};

	//TODO - Question 3: Select the grid dimensions (blocks)
	//Assume a 1D grid
	gridX = M / blockX + 1; 	//TODO: Select the number of blocks
	gridY = 1; 
	gridZ = 1;
	numBlock = {gridX, gridY, gridZ};

	printf("Running DGEMV with size %d x %d on the GPU - Naive version\n", M, N);
	gettimeofday(&ts, NULL);
	
	//TODO - Question 3: Call the "cudaDGEMV" function
		//TODO: Call cudaDGEMV
	//----__global__ void cudaDGEMV(float * A, float * x, float * b, int M, int N) {
    cudaDGEMV<<<numBlock, ThreadsPerBlock>>>(dev_A, dev_x, dev_b, M, N);
    
	cudaDeviceSynchronize();	//Block CPU execution until kernel has completed on GPU
	gettimeofday(&tf, NULL);

	//TODO - Question 1: Copy the result from "dev_b" back to the CPU in "b"
		//TODO: Copy dev_b to b
	//----
    cudaMemcpy(b, dev_b, M * sizeof(float), cudaMemcpyDeviceToHost);

    
	printf("Time: %.5lf(s) -- ", timetosol(ts, tf));
	if (!checkCorrectness(sol, b, M)) 
		printf("PASS\n");


	//----Free memory on GPU and CPU----//

	//TODO - Question 1: Free memory on the GPU size
		//TODO: Free dev_A
		//TODO: Free dev_b
		//TODO: Free dev_x
	//----
    cudaFree(dev_A); 
    cudaFree(dev_b); 
    cudaFree(dev_x); 

	//Free buffers on CPU
	free(A);
	free(x);
	free(b);
	free(sol);
	//---------------------------------//

	return 0;

}

