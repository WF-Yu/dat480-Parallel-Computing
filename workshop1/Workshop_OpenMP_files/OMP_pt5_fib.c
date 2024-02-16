#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

//Serial version
size_t fib(int n) {
    size_t l, r;
    if (n < 2) 
        return n;
    l = fib(n - 1);
    r = fib(n - 2);

    return l + r;
}

//TODO: TASKS - Questions 1-4 - Parallelize with tasks
size_t fib_parallel(int n) {
    size_t l, r;
    if (n < 2) 
        return n;
#pragma omp task shared(l) 
    l = fib(n - 1);
#pragma omp task shared(r)
    r = fib(n - 2);
#pragma omp taskwait
    return l + r;
}

//TODO: TASKS - Bonus question - Parallelize with tasks and control over task creation
size_t fib_parallel_control(int n) {
    size_t l, r;
    if (n < 2) 
        return n;

#pragma omp task shared(l) final(n < 41)
    l = fib_parallel_control(n - 1);
#pragma omp task shared(r) final(n < 41)
    r = fib_parallel_control(n - 2);
#pragma omp taskwait
    return l + r;
}


int main(int argc, char** argv) {
  // Initialize
  int N = (argc > 1) ? atoi(argv[1]) : 42 ;
  size_t res;
  
  printf("Running for problem size N: %d\n", N);

  //Timers
  double ts, tf;

  //Serial version 
  printf("Running serial.....................................\n");
  ts = omp_get_wtime();
  
  res = fib(N);
  
  tf = omp_get_wtime();
  printf("Time: %.4lfs\n", tf - ts);

  //Copying solution for correctness check
  size_t serial_res = res;


  //TODO: TASKS - Questions 1-4 - Parallelize with tasks

  printf("Running parallel with tasks........................\n");

  ts = omp_get_wtime();

#pragma omp parallel
  {
    #pragma omp single
    res = fib_parallel(N);        //FIXME: Parallelize with tasks
  }
  tf = omp_get_wtime();
  
  if (res == serial_res)
    printf("Time: %.4lfs -- PASS\n", tf - ts);
  else
    printf("FAIL\n");


  //TODO: TASKS - Bonus question - Parallelize with tasks and control over task creation


  printf("Running parallel with tasks and control............\n");

  ts = omp_get_wtime();

  res = fib_parallel_control(N);    // FIXME: Parallelize with tasks

  tf = omp_get_wtime();
 
  if (res == serial_res)
    printf("Time: %.4lfs -- PASS\n", tf - ts);
  else
    printf("FAIL: Parallel-%lu Serial- %lu\n", res, serial_res);

  return 0;
}

