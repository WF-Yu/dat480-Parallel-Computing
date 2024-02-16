CC=gcc
CFLAGS=-O0 -fopenmp
LIBS=-lm 

all: nbody_spmd nbody_loop nbody_sync nbody_simd fib_tasks
build_pt1: nbody_spmd 
build_pt2: nbody_loop
build_pt3: nbody_sync
build_pt4: nbody_simd
build_pt5: fib_tasks

nbody_spmd: OMP_pt1_nbody.c
	$(CC) $(CFLAGS) OMP_pt1_nbody.c -o nbody_spmd $(LIBS) 

nbody_loop: OMP_pt2_nbody.c
	$(CC) $(CFLAGS) OMP_pt2_nbody.c -o nbody_loop $(LIBS) 

nbody_sync: OMP_pt3_nbody.c
	$(CC) $(CFLAGS) OMP_pt3_nbody.c -o nbody_sync $(LIBS) 

nbody_simd: OMP_pt4_nbody.c
	$(CC) $(CFLAGS) OMP_pt4_nbody.c -o nbody_simd $(LIBS) 

fib_tasks: OMP_pt5_fib.c
	$(CC) $(CFLAGS) OMP_pt5_fib.c -o fib_tasks $(LIBS) 

run_pt1:
	OMP_NUM_THREADS=4 ./nbody_spmd
run_pt2:
	OMP_NUM_THREADS=4 ./nbody_loop
run_pt3:
	OMP_NUM_THREADS=4 ./nbody_sync
run_pt4:
	OMP_NUM_THREADS=4 ./nbody_simd
run_pt5:
	OMP_NUM_THREADS=4 ./fib_tasks

run: run_pt1 run_pt2 run_pt3 run_pt4 run_pt5
	

default: all
clean:
	rm -f nbody_spmd nbody_loop nbody_sync nbody_simd fib_tasks
