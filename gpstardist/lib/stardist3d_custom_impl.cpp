#include <stdarg.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <limits>
#include <vector>
#include <array>
#include <string>
#include <chrono>
#include <cstdint>
#include <signal.h>


#define DIM 3

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// global termination flag, to make long running loops interruptable
int IS_TERMINATED = 0;
typedef void (*sighandler_t)(int);

void my_signal_handler( int signum ) {
  IS_TERMINATED = 1;
  std::cout<<"AA ";

}


int round_to_int(float r) {
  return (int)lrint(r);
}
/*
int _sum_buffer(const bool * const buffer, const int N){
  int res = 0;
  for (int i=0; i <N; ++i)
	res += buffer[i];
  return res;
}
*/
