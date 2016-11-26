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
__global__
void divide_and_check(uint128_t x);


int main(int argc, char * argv[])
{
  uint128_t x;
  if(argc == 2){
    x = calc(argv[1]);
  }

  KernelTime timer;

  timer.start();

  divide_and_check<<<1, 256>>>(x);

  cudaDeviceSynchronize();
  timer.stop();
  timer.displayTime();

  return 0;
}

__global__
void atimesbequalsc(uint64_t * a, uint64_t * b, uint128_t * c)
{
  uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  c[tidx] = mul128(a[tidx], b[tidx]);
}

__global__
void squarerootc(uint128_t * c, uint64_t * a)
{
  uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  a[tidx] = _isqrt(c[tidx]);
  if(mul128(a[tidx], a[tidx]) > c[tidx] || mul128((a[tidx] + 1), (a[tidx] + 1)) <= c[tidx])
    printf("%llu  %f  %llu\n", a[tidx], u128_to_float(c[tidx]), c[tidx].hi);
}

__global__
void divide_and_check(uint128_t x)
{
  uint64_t tidx = 2 + threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ uint64_t r[256];
  __shared__ uint128_t z[256];

  uint128_t y = uint128_t::div128to128(x, tidx, &r[threadIdx.x]);
  z[threadIdx.x] = mul128(y, tidx);
  z[threadIdx.x] = add128(z, r[threadIdx.x]);

  if(z[threadIdx.x].lo != x.lo) printf("%llu\t%llu\n", x.lo, z[threadIdx.x].lo);
  __syncthreads();
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

  value = string_to_u128(s);

  return value;
}
