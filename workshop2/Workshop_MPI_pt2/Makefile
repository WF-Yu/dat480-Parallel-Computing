CC=gcc
MPICC=mpicc
CFLAGS=-O3 
LIBS=-lm 

all: pi_array_serial pi_array_mpi_p2p pi_array_mpi_coll pi_array_mpi_coll_opt comms_mpi
build_pi_array: pi_array_serial pi_array_mpi_p2p pi_array_mpi_coll pi_array_mpi_coll_opt
build_comms: comms_mpi

pi_array_serial: pi_array_serial.c
	$(CC) $(CFLAGS) pi_array_serial.c -o pi_array_serial $(LIBS)
pi_array_mpi_p2p: pi_array_mpi.c 
	$(MPICC) $(CFLAGS) pi_array_mpi.c -o pi_array_mpi_p2p $(LIBS) -DP2P
pi_array_mpi_coll: pi_array_mpi.c
	$(MPICC) $(CFLAGS) pi_array_mpi.c -o pi_array_mpi_coll $(LIBS) -DCOLL
pi_array_mpi_coll_opt: pi_array_mpi.c
	$(MPICC) $(CFLAGS) pi_array_mpi.c -o pi_array_mpi_coll_opt $(LIBS) -DCOLL_OPT

run_pi_array:
	./pi_array_serial
	mpirun -np 4 ./pi_array_mpi_p2p
	mpirun -np 4 ./pi_array_mpi_coll
	mpirun -np 4 ./pi_array_mpi_coll_opt

comms_mpi: comms_mpi.c
	$(MPICC) $(CFLAGS) comms_mpi.c -o comms_mpi $(LIBS)

run_comms: 
	mpirun -np 7 --oversubscribe ./comms_mpi

default: all
clean:
	rm -f pi_array_serial pi_array_mpi_p2p pi_array_mpi_coll pi_array_mpi_coll_opt comms_mpi
