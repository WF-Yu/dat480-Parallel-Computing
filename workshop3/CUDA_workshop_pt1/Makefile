NVCC=nvcc
NVCC_FLAGS=-O2 -arch=sm_61
NVCC_INC_FLAGS=-I/chalmers/sw/sup64/cuda_toolkit-11.2.2/include
NVCC_LIB_DIR=-L/chalmers/sw/sup64/cuda_toolkit-11.2.2/lib64
NVCC_LIBS=-lcudart

all: dgemv_naive dgemv_shmem_linear dgemv_shmem_binary dgemv_shmem_tiled

dgemv_naive.o: dgemv_naive.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAGS) -c dgemv_naive.cu -o dgemv_naive.o
dgemv_shmem_linear.o: dgemv_shmem.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAGS) -DLINEAR_REDUCTION -c dgemv_shmem.cu -o dgemv_shmem_linear.o
dgemv_shmem_binary.o: dgemv_shmem.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAGS) -DBINARY_REDUCTION -c dgemv_shmem.cu -o dgemv_shmem_binary.o
dgemv_shmem_tiled.o: dgemv_shmem_tiled.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAGS) -c dgemv_shmem_tiled.cu -o dgemv_shmem_tiled.o

dgemv_naive: dgemv_naive.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAGS) -o dgemv_naive $< $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)
dgemv_shmem_linear: dgemv_shmem_linear.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAGS) -o dgemv_shmem_linear $< $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)
dgemv_shmem_binary: dgemv_shmem_binary.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAGS) -o dgemv_shmem_binary $< $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)
dgemv_shmem_tiled: dgemv_shmem_tiled.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAGS) -o dgemv_shmem_tiled $< $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

build_pt1: dgemv_naive
run_pt1:
	./dgemv_naive

build_pt2: dgemv_shmem_linear dgemv_shmem_binary 
build_pt3: dgemv_shmem_tiled
run_pt2:
	./dgemv_shmem_linear
	./dgemv_shmem_binary
run_pt3:
	./dgemv_shmem_tiled

clean:
	rm *.o dgemv_naive dgemv_shmem_linear dgemv_shmem_binary dgemv_shmem_tiled
