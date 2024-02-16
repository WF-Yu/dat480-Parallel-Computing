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
  printf("N=%d\n", N);
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


  //TODO: SPMD - Question 1 - Parallelize the outer loop 

  printf("Running parallel (outer loop).......................\n");
  ts = omp_get_wtime();
#pragma omp parallel private (i)
  {
      int thread_count = omp_get_num_threads();
      // printf("thread_count = %d\n", thread_count);
      int partition_size = N / thread_count;
      int thread_id = omp_get_thread_num();
      int start = thread_id*partition_size;
      int end = ((thread_id+1)==thread_count)?N:start+partition_size;
      
      // printf("start:%d end:%d\n", start, end);
    for (i=start; i<end; i++) {            //FIXME: Parallelize
        float pi = 0;
        float axi = 0;
        float ayi = 0;
        float xi = x[i];
        float yi = y[i];
        for (int j=0; j<N; j++) {
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
  }
  
  tf = omp_get_wtime();

  if(test(N, sol, p, ax, ay)) 
    printf("Time: %.4lfs -- PASS\n", tf - ts);
  else
    printf("FAIL\n");

  //TODO: SPMD - Question 2 - Parallelize the inner loop 
// version with no false sharing
  printf("Running parallel (inner loop).......................\n");
  ts = omp_get_wtime();
  for (i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float xi = x[i];
    float yi = y[i];
    float pii[N];
    float axii[N];
    float ayii[N];
    
    #pragma omp parallel shared (pi, axi, ayi)
    {
        int thread_count = omp_get_num_threads();
        // printf("thread_count = %d\n", thread_count);
        int partition_size = N / thread_count;
        int thread_id = omp_get_thread_num();
        int start = thread_id*partition_size;
        int end = ((thread_id+1)==thread_count)?N:start+partition_size;
        
        for (int j=start; j<end; j++) {       //FIXME: Parallelize
            float dx = x[j] - xi;
            float dy = y[j] - yi;
            float R2 = dx * dx + dy * dy + EPS2;
            float invR = 1.0f / sqrtf(R2);
            float invR3 = m[j] * invR * invR * invR;
            // #pragma omp critical
            // {
            
                //pi += m[j] * invR;
                //axi += dx * invR3;
                //ayi += dy * invR3;
                pii[j] = m[j] * invR;
                axii[j] = dx * invR3;
                ayii[j] = dy * invR3;
            // }
            
        }
    }
    for (int a = 0; a<N; a++){
        pi+=pii[a];
        axi+=axii[a];
        ayi+=ayii[a];
        
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

  //TODO: SPMD - Question 3 - Parallelize the inner loop and avoid false sharing

  printf("Running parallel (inner loop without false sharing).\n");
  ts = omp_get_wtime();
 // I put the one with false sharing here
 for (i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float xi = x[i];
    float yi = y[i];
    
    float pii[4]={0};
    float axii[4]={0};
    float ayii[4]={0};

#pragma omp parallel shared (pi, axi, ayi)
    {
    int thread_count = omp_get_num_threads();
        // printf("thread_count = %d\n", thread_count);
    int partition_size = N / thread_count;
    int thread_id = omp_get_thread_num();
    int start = thread_id*partition_size;
    int end = ((thread_id+1)==thread_count)?N:start+partition_size;
    for (int j=start; j<end; j++) {       //FIXME: Parallelize without false sharing
      float dx = x[j] - xi;
      float dy = y[j] - yi;
      float R2 = dx * dx + dy * dy + EPS2;
      float invR = 1.0f / sqrtf(R2);
      float invR3 = m[j] * invR * invR * invR;
      pii[thread_id] += m[j] * invR;
      axii[thread_id] += dx * invR3;
      ayii[thread_id] += dy * invR3;
    }
    }
    p[i]=0;
    ax[i]=0;
    ay[i]=0;
    for (int a = 0; a < 4; a++){
        p[i] += pii[a];
        ax[i] += axii[a];
        ay[i] = ayii[a];
    }
    
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

