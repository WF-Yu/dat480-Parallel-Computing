NVCC=nvcc
NVCC_FLAGS=-O2 -arch=sm_61
NVCC_INC_FLAGS=-I/chalmers/sw/sup64/cuda_toolkit-11.2.2/include
NVCC_LIB_DIR=-L/chalmers/sw/sup64/cuda_toolkit-11.2.2/lib64
NVCC_LIBS=-lcudart

all: dgemv_resource_management dgemv_streams

dgemv_resource_management.o: dgemv_resource_management.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAGS) -c dgemv_resource_management.cu -o dgemv_resource_management.o
dgemv_resource_management: dgemv_resource_management.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAGS) -o dgemv_resource_management $< $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

dgemv_streams.o: dgemv_streams.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAGS) -c dgemv_streams.cu -o dgemv_streams.o
dgemv_streams: dgemv_streams.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAGS) -o dgemv_streams $< $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)


build_pt1: dgemv_resource_management

run_pt1:
	./dgemv_resource_management

build_pt2: dgemv_streams
run_pt2:
	./dgemv_streams
clean:
	rm *.o dgemv_resource_management dgemv_streams
