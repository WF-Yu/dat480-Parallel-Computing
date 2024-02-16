CC=gcc
MPICC=mpicc
CFLAGS=-O3 
LIBS=-lm 

all: pi_serial pi_mpi pi_ring_blocking pi_ring_synchronous pi_ring_deadlockavoiding
build_pi: pi_serial pi_mpi
build_pi_ring: pi_ring_blocking pi_ring_synchronous pi_ring_deadlockavoiding

pi_serial: pi_serial.c
	$(CC) $(CFLAGS) pi_serial.c -o pi_serial $(LIBS)
pi_mpi: pi_mpi.c
	$(MPICC) $(CFLAGS) pi_mpi.c -o pi_mpi $(LIBS)
run_pi:
	./pi_serial
	mpirun -np 4 ./pi_mpi

pi_ring_blocking: pi_ring_mpi.c
	$(MPICC) $(CFLAGS) pi_ring_mpi.c -o pi_ring_blocking $(LIBS) -DBLOCKING
pi_ring_synchronous: pi_ring_mpi.c
	$(MPICC) $(CFLAGS) pi_ring_mpi.c -o pi_ring_synchronous $(LIBS) -DBLOCKING_SYNCHRONOUS
pi_ring_deadlockavoiding: pi_ring_mpi.c
	$(MPICC) $(CFLAGS) pi_ring_mpi.c -o pi_ring_deadlockavoiding $(LIBS) -DDEADLOCK_AVOIDING_1

run_pi_ring:
	# mpirun -np 4 ./pi_ring_blocking
	# mpirun -np 4 ./pi_ring_synchronous
	mpirun -np 4 ./pi_ring_deadlockavoiding


default: all
clean:
	rm -f pi_serial pi_mpi pi_ring_blocking pi_ring_synchronous pi_ring_deadlockavoiding
