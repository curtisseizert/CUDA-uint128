#include <iostream>
#include <stdint.h>
#include <math.h>
#include <string>

#include <CUDASieve/cudasieve.hpp>
#include <CUDASieve/host.hpp>

#include "cuda_uint128.h"

uint128_t calc(char * argv);
__global__
void atimesbequalsc(uint64_t * a, uint64_t * b, uint128_t * c);
__global__
void squarerootc(uint128_t * c, uint64_t * a);

int main(int argc, char * argv[])
{
  uint128_t x;
  if(argc == 2){
    x = calc(argv[1]);
  }

  uint64_t * d_a, *d_b;
  uint128_t * d_c;
  size_t len_a, len_b;

  d_a = CudaSieve::getDevicePrimes(pow(2,55)+pow(2,32), pow(2,55) + pow(2,33), len_a, 0);
  d_b = CudaSieve::getDevicePrimes(pow(2,60), pow(2,60) + pow(2,32), len_b, 0);

  cudaMalloc(&d_c, len_a * sizeof(uint128_t));
  std::cout << "Number of elements : " << len_a << std::endl;

  KernelTime timer;

  timer.start();

  atimesbequalsc<<<len_a/256, 256>>>(d_a, d_b, d_c);

  timer.stop();
  timer.displayTime();

  KernelTime timer2;

  timer2.start();

  squarerootc<<<390625, 256>>>(d_c, d_a);

  timer2.stop();
  timer2.displayTime();

  std::cout << x << " " << uint128_t::u128_to_double(x) << std::endl;

  uint128_t::sqrt(x);

  return 0;
}

__global__
void atimesbequalsc(uint64_t * a, uint64_t * b, uint128_t * c)
{
  uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  c[tidx] = uint128_t::mul128(a[tidx], b[tidx]);
}

__global__
void squarerootc(uint128_t * c, uint64_t * a)
{
  uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  a[tidx] = uint128_t::sqrt(c[tidx]);
  if(uint128_t::mul128(a[tidx], a[tidx]) > c[tidx] || uint128_t::mul128((a[tidx] + 1), (a[tidx] + 1)) <= c[tidx])
    printf("%llu  %f  %llu\n", a[tidx], uint128_t::u128_to_float(c[tidx]), c[tidx].hi);
}

uint128_t calc(char * argv) // for getting values bigger than the 32 bits that system() will return;
{
  uint128_t value;
  size_t len = 0;
  char * line = NULL;
  FILE * in;
  char cmd[256];

  sprintf(cmd, "calc %s | awk {'print $1'}", argv);

  in = popen(cmd, "r");
  getline(&line, &len, in);
  std::string s = line;

  value = uint128_t::stou128(s);

  return value;
}
