#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

//Problem size M x N
#define M_ 8192
#define N_ 8192

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
		if (fabs(sol_cpu[i] - sol_gpu[i])>1e-2) {
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

//TODO: Function cudaDGEMV_shmem
//----Computes Matrix-Vector Multiplication A * x = b on the GPU
//----Input dev_A: array M x N 
//----Input dev_x: vector N x 1 
//----Output dev_b: vector M x 1 
//----Uses GPU shared memory 
//----Each thread block accesses one row of matrix A

__global__ void cudaDGEMV_shmem(float * A, float * x, float * b, int M, int N) {
	//TODO: Question 1 - Fill in this function to perform matrix-vector multiplication on the GPU
	//TODO: Each thread block should access a single row of matrix A
	//TODO: We have declared a shared memory array equal to the number of threads in a block
	//TODO: Use this array to reduce intermediate values for the vector "b"
	__shared__ float sum[BLOCK_DIM];	//Declare shmem array equal to number of threads in a block 
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int row = bid * N;
    int step = N / BLOCK_DIM; // number of calculatins in one thread
    sum[tid] = 0.0;
   
    // calculate the sum in each thread
    for (int i = 0; i < step && tid * step + i < N; i++) {
        sum[tid] += A[i + row + tid * step] * x[i + tid * step];
    }
    __syncthreads();
#if LINEAR_REDUCTION
	//TODO: Question 1 - Perform reduction on output vector "b"
	if (tid == 0) {
        b[bid] = 0.0;
        for (int i = 0; i < BLOCK_DIM; i++) 
            b[bid] += sum[i]; // sum up the results from each thread
    }
    
#elif BINARY_REDUCTION
    // printf("hi\n");
	//TODO: Question 2 - Perform a binary reduction on output vector "b"
    for (int step = 1; step <= BLOCK_DIM / 2; step <<= 1) {
        // printf("tid = %d, step = %d\n", tid, step);
        // step == 1, 2, 4, 8, .....
        if ( tid % (step * 2) == 0 && tid + step < BLOCK_DIM) {
            sum[tid] += sum[tid + step];
        }
        __syncthreads();
    }
    b[bid] = sum[0];
#endif
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
	//TODO: Select the block dimensions (threads per block)
	//Assume a 1D block
	blockX = BLOCK_DIM;	//TODO: Select the number of threads per block
	blockY = 1;
    blockZ = 1;
	threadsPerBlock = {blockX, blockY, blockZ};

	//TODO: Select the grid dimensions (blocks)
	//Assume a 1D grid
	gridX = M;	//TODO: Select the number of blocks
	gridY = 1; 
	gridZ = 1;
	numBlock = {gridX, gridY, gridZ};
#ifdef LINEAR_REDUCTION
	printf("Running DGEMV with size %d x %d on the GPU - Shared memory version\n", M, N);
#elif BINARY_REDUCTION
	printf("Running DGEMV with size %d x %d on the GPU - Shared memory version + Binary reduction\n", M, N);
#endif
	gettimeofday(&ts, NULL);

	cudaDGEMV_shmem<<<numBlock,threadsPerBlock>>>(dev_A, dev_x, dev_b, M, N);
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

