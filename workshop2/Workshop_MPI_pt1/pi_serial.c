#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10000

int main(int argc, char ** argv) {
	double pi, exact_pi;
	int i;
	printf("Computing approximation to pi using N=%d\n", N);
	pi = 0.0;
	for (i = 1 ; i <= N ; i++) {
		pi = pi + 1.0 / (1.0 + pow( (((double)i - 0.5) / (double)N), 2.0));
	}
	pi = pi * 4.0 / (double)N;
	exact_pi = 4.0 * atan(1.0);
	printf("Pi = %f, Exact pi = %f, Error = %f\n", pi, exact_pi, fabs(100.0 * (pi - exact_pi) / exact_pi));

	return 0;
}
