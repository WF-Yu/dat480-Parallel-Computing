#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#define ERR 1e-3
#define DIFF(x,y) ((x-y)<0? y-x : x-y)
#define FPNEQ(x,y) (DIFF(x,y)>ERR ? 1 : 0)
int test(int N, float * sol, float * p, float * ax, float * ay) {
  int i;
  for (i = 0 ; i < N ; i++) {
    if (FPNEQ(sol[i],p[i])) 
      return 0;
  }
  for (i = 0 ; i < N ; i++) {
    if (FPNEQ(sol[i+N],ax[i])) 
      return 0;
  }
  for (i = 0 ; i < N ; i++) {
    if (FPNEQ(sol[i+2 * N], ay[i]))
      return 0;
  }
  return 1;
}

int main(int argc, char** argv) {
  // Initialize
  int pow = (argc > 1)? atoi(argv[1]) : 14;
  int N = 1 << pow;
  int i, j;
  float OPS = 20. * N * N * 1e-9;
  float EPS2 = 1e-6;
  float* x =  (float*)malloc(N * sizeof(float));
  float* y =  (float*)malloc(N * sizeof(float));
  float* m =  (float*)malloc(N * sizeof(float));
  float* p =  (float*)malloc(N * sizeof(float));
  float* ax = (float*)malloc(N * sizeof(float));
  float* ay = (float*)malloc(N * sizeof(float));
  for (i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48() / N;
    p[i] = ax[i] = ay[i] =  0;
  }

  printf("Running for problem size N: %d\n", N);

  //Timers
  double ts, tf;

  //Serial version 
  printf("Running serial......................................\n");
  ts = omp_get_wtime();
  for (i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float xi = x[i];
    float yi = y[i];
    for (j=0; j<N; j++) {
      float dx = x[j] - xi;
      float dy = y[j] - yi;
      float R2 = dx * dx + dy * dy + EPS2;
      float invR = 1.0f / sqrtf(R2);
      float invR3 = m[j] * invR * invR * invR;
      pi += m[j] * invR;
      axi += dx * invR3;
      ayi += dy * invR3;
    }
    p[i] = pi;
    ax[i] = axi;
    ay[i] = ayi;
  }
  tf = omp_get_wtime();
  printf("Time: %.4lfs\n", tf - ts);

  //Copying solution for correctness check
  float* sol = (float*)malloc(3 * N * sizeof(float));
  memcpy(sol, p, N * sizeof(float));
  memcpy(sol + N, ax, N * sizeof(float));
  memcpy(sol+ 2 * N, ay, N * sizeof(float));


  //TODO: SYNC - Question 1 - Parallelize the inner loop with OMP reduction

  printf("Running parallel (inner loop with parallel for reduction).....\n");
  ts = omp_get_wtime();
 
 for (i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float xi = x[i];
    float yi = y[i];
    
#pragma omp parallel for reduction (+ : pi, axi, ayi) 
        for (int j=0; j<N; j++) {       //FIXME: Parallelize using OMP loop parallelization
        float dx = x[j] - xi;
        float dy = y[j] - yi;
        float R2 = dx * dx + dy * dy + EPS2;
        float invR = 1.0f / sqrtf(R2);
        float invR3 = m[j] * invR * invR * invR;
        pi += m[j] * invR;
        axi += dx * invR3;
        ayi += dy * invR3;
        }
    
    p[i] = pi;
    ax[i] = axi;
    ay[i] = ayi;
  }

  tf = omp_get_wtime();
  if(test(N, sol, p, ax, ay)) 
    printf("Time: %.4lfs -- PASS\n", tf - ts);
  else
    printf("FAIL\n");

   //TODO: SYNC - Question 1 - Parallelize the inner loop with atomics/locks/critical section
 
  printf("Running parallel (inner loop with parallel for and atomic).....\n");
  ts = omp_get_wtime();
 
 for (i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float xi = x[i];
    float yi = y[i];
#pragma omp parallel for 
    for (j=0; j<N; j++) {       //FIXME: Parallelize using OMP loop parallelization
      float dx = x[j] - xi;
      float dy = y[j] - yi;
      float R2 = dx * dx + dy * dy + EPS2;
      float invR = 1.0f / sqrtf(R2);
      float invR3 = m[j] * invR * invR * invR;
#pragma omp atomic
      pi += m[j] * invR;
#pragma omp atomic
      axi += dx * invR3;
#pragma omp atomic
      ayi += dy * invR3;
    }
    p[i] = pi;
    ax[i] = axi;
    ay[i] = ayi;
  }

  tf = omp_get_wtime();
  if(test(N, sol, p, ax, ay)) 
    printf("Time: %.4lfs -- PASS\n", tf - ts);
  else
    printf("FAIL\n");
 
  
   printf("Running parallel (inner loop with parallel for and critical).....\n");
  ts = omp_get_wtime();
 
 for (i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float xi = x[i];
    float yi = y[i];
#pragma omp parallel for 
    for (j=0; j<N; j++) {       //FIXME: Parallelize using OMP loop parallelization
      float dx = x[j] - xi;
      float dy = y[j] - yi;
      float R2 = dx * dx + dy * dy + EPS2;
      float invR = 1.0f / sqrtf(R2);
      float invR3 = m[j] * invR * invR * invR;
    
      #pragma omp critical
      {
        pi += m[j] * invR;
        axi += dx * invR3;
        ayi += dy * invR3;
      }
    }
    p[i] = pi;
    ax[i] = axi;
    ay[i] = ayi;
  }

  tf = omp_get_wtime();
  if(test(N, sol, p, ax, ay)) 
    printf("Time: %.4lfs -- PASS\n", tf - ts);
  else
    printf("FAIL\n");
  
  
   printf("Running parallel (inner loop with parallel for and locks).....\n");
  ts = omp_get_wtime();
 
 for (i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float xi = x[i];
    float yi = y[i];
    
    omp_lock_t lck;
    omp_init_lock(&lck);
#pragma omp parallel for shared (lck)
    for (j=0; j<N; j++) {       //FIXME: Parallelize using OMP loop parallelization
      float dx = x[j] - xi;
      float dy = y[j] - yi;
      float R2 = dx * dx + dy * dy + EPS2;
      float invR = 1.0f / sqrtf(R2);
      float invR3 = m[j] * invR * invR * invR;
      
      omp_set_lock(&lck);
      pi += m[j] * invR;
      axi += dx * invR3;
      ayi += dy * invR3;
      omp_unset_lock(&lck);
      
    }
    p[i] = pi;
    ax[i] = axi;
    ay[i] = ayi;
  }

  tf = omp_get_wtime();
  if(test(N, sol, p, ax, ay)) 
    printf("Time: %.4lfs -- PASS\n", tf - ts);
  else
    printf("FAIL\n");
  
  free(x);
  free(y);
  free(m);
  free(p);
  free(ax);
  free(ay);
  free(sol);
  return 0;
}

