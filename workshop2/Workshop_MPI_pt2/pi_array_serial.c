#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10000

int main(int argc, char ** argv) {
	double pi, exact_pi;
	int i;
	int * random_seq = malloc(N * sizeof(int));
	for (i = 0 ; i < N ; i++)
		random_seq[i] = rand()%N;
	printf("Computing approximation to pi using a random array of N=%d numbers\n", N);
	pi = 0.0;
	for (i = 0 ; i < N ; i++) {
		pi = pi + 1.0 / (1.0 + pow( (((double)random_seq[i] - 0.5) / (double)N), 2.0));
	}
	pi = pi * 4.0 / (double)N;
	exact_pi = 4.0 * atan(1.0);
	printf("Pi = %f, Error = %f\n", pi, fabs(100.0 * (pi - exact_pi) / exact_pi));

	return 0;
}
