#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <assert.h>
//Problem size M x N
#define M_ 16384
#define N_ 16384

//A sample dimension for the block (number of threads per block)
#ifndef THREADS_PER_BLOCK
	#define THREADS_PER_BLOCK 128
#endif

//A sample dimension for the block size for tiling (x-vector elements copied in GPU shared memory)
#ifndef BLOCK_SIZE
	#define BLOCK_SIZE 2048
#endif

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
				//TODO: Find and print the number of SMs
                
                printf("QUESTION1:%d\n", prop.multiProcessorCount);
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

//Function cudaDGEMV_shmem
//Computes Matrix-Vector Multiplication A * x = b on the GPU
//Input dev_A: array M x N 
//Input dev_x: vector N x 1 
//Output dev_b: vector M x 1 
//Uses GPU shared memory 
//Each thread accesses one row of matrix A
//Vector "x" is buffered in shared memory for fast access

__global__ void cudaDGEMV_shmem(float * A, float * x, float * b, int M, int N) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	float sum = 0.;
	__shared__ float x_shared[BLOCK_SIZE];
	unsigned int end = (N % BLOCK_SIZE == 0) ? N / BLOCK_SIZE : 1 + N / BLOCK_SIZE;
	for (unsigned int i = 0 ; i < end; i++) {
		int block_end = (BLOCK_SIZE % blockDim.x == 0) ? BLOCK_SIZE / blockDim.x : 1 + BLOCK_SIZE / blockDim.x;

		for (unsigned int j = 0; j < block_end ; j++) {
			if (i * BLOCK_SIZE + threadIdx.x + j * blockDim.x < N && threadIdx.x + j * blockDim.x < BLOCK_SIZE) {
				x_shared[threadIdx.x + j * blockDim.x] = x[i * BLOCK_SIZE + threadIdx.x + j * blockDim.x];
			}
		}
	
		__syncthreads();
		if (tid < M) {
			for (unsigned int j = 0 ; j < BLOCK_SIZE ; j++) {
				if (i * BLOCK_SIZE + j < N)
					sum += A[tid * N + i * BLOCK_SIZE + j] * x_shared[j];
			}
		}
		__syncthreads();
	}
	if (tid < M )
		b[tid] = sum;

}


int main(int argc, char ** argv) {
	//----Problem input M x N----//
	int M = (argc >= 3) ? atoi(argv[1]) : M_;	//Read from input or default
	int N = (argc >= 3) ? atoi(argv[2]) : N_; 
	//---------------------------//


	//----Variables declaration----//
	struct timeval ts, tf;

	int i, j;

	cudaError_t err;

	//Block & Grid dimensions for the GPU
	unsigned int blockX, blockY, blockZ;
	unsigned int gridX, gridY, gridZ;
	dim3 threadsPerBlock;
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
	err = cudaMalloc(&dev_A, M * N * sizeof(float));
	if (err != 0) {
		fprintf(stderr, "Error allocating matrix A on GPU\n");
		exit(-1);
	}
	err = cudaMalloc(&dev_x, N * sizeof(float));
	if (err != 0) {
		fprintf(stderr, "Error allocating vector x on GPU\n");
		exit(-1);
	}
	err = cudaMalloc(&dev_b, M * sizeof(float));
	if (err != 0) {
		fprintf(stderr, "Error allocating vector b on GPU\n");
		exit(-1);
	}
	//----------------------//

	//----Perform CPU to GPU transfers----//
	err = cudaMemcpy(dev_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
	if (err != 0) {
		fprintf(stderr, "Error copying matrix A to GPU\n");
		exit(-1);
	}

	err = cudaMemcpy(dev_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	if (err != 0) {
		fprintf(stderr, "Error copying vector x to GPU\n");
		exit(-1);
	}

	//------------------------------------//

	//----Shared-memory DGEMV A * x = b on GPU----//
	//Assume a 1D block
	blockX = THREADS_PER_BLOCK;	//Number of threads per block	
	blockY = 1;
       	blockZ = 1;
	threadsPerBlock = {blockX, blockY, blockZ};

	//Assume a 1D grid
	gridX = (M % THREADS_PER_BLOCK == 0) ? M / THREADS_PER_BLOCK : 1 + M / THREADS_PER_BLOCK; //Number of blocks
	gridY = 1; 
	gridZ = 1;
	numBlock = {gridX, gridY, gridZ};

	printf("Running DGEMV with size %d x %d on the GPU with %d blocks of %d threads each - Shared memory version - Tiled\n", M, N, gridX, blockX);
	gettimeofday(&ts, NULL);

	cudaDGEMV_shmem<<<numBlock, threadsPerBlock>>>(dev_A, dev_x, dev_b, M, N);
	cudaDeviceSynchronize();
	
	gettimeofday(&tf, NULL);

	//----Perform GPU to CPU transfers----//
	cudaMemcpy(b, dev_b, M * sizeof(float), cudaMemcpyDeviceToHost);
	//------------------------------------//

	printf("Time: %.5lf(s) -- ", timetosol(ts, tf));
	if (!checkCorrectness(sol, b, M)) 
		printf("PASS\n");


	//----Free memory on GPU and CPU----//
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

