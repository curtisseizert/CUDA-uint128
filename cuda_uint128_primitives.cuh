/*

This header file contains some higher level primitives to simplify certain
operations, mostly (probaly exclusively) that I need to use elsewhere.


*/

#include <cuda_runtime.h>

#ifndef _UINT128_T_CUDA_H
#include "cuda_uint128.h"
#endif

#ifndef _CUDA_UINT128_PRIMITIVES
#define _CUDA_UINT128_PRIMITIVES

namespace cu128_internal{

/// This is a slightly modified version of Mark Harris's optimized reduction
/// kernel (http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf)
template <typename T, uint32_t blockSize>
__global__ void reduce_to_128_kernel(T * a, size_t len, uint128_t * partial_sums)
{
  __shared__ uint128_t s_part[blockSize];
  uint64_t i = threadIdx.x + blockIdx.x * blockSize;
  uint64_t gridSize = blockSize * gridDim.x;

  s_part[threadIdx.x] = 0;

  while(i < len){
    s_part[threadIdx.x] += a[i];
    i += gridSize;
  }

  __syncthreads();

  if(threadIdx.x < 256)
    if(blockSize >= 512)
      s_part[threadIdx.x] += s_part[threadIdx.x + 256];
  __syncthreads();
  if(threadIdx.x < 128)
    if(blockSize >= 256)
      s_part[threadIdx.x] += s_part[threadIdx.x + 128];
  __syncthreads();
  if(threadIdx.x < 64)
    if(blockSize >= 128)
      s_part[threadIdx.x] += s_part[threadIdx.x + 64];
  __syncthreads();
  if(threadIdx.x < 32)
    if(blockSize >= 64)
      s_part[threadIdx.x] += s_part[threadIdx.x + 32];
  __syncthreads();
  if(threadIdx.x < 32)
    if(blockSize >= 32)
      s_part[threadIdx.x] += s_part[threadIdx.x + 16];
  __syncthreads();
  if(threadIdx.x < 32)
    if(blockSize >= 16)
      s_part[threadIdx.x] += s_part[threadIdx.x + 8];
  __syncthreads();
  if(threadIdx.x < 32)
    if(blockSize >= 8)
      s_part[threadIdx.x] += s_part[threadIdx.x + 4];
  __syncthreads();
  if(threadIdx.x < 32)
    if(blockSize >= 4)
      s_part[threadIdx.x] += s_part[threadIdx.x + 2];
  __syncthreads();
  if(threadIdx.x < 32)
    if(blockSize >= 2)
      s_part[threadIdx.x] += s_part[threadIdx.x + 1];
  __syncthreads();

  if(threadIdx.x == 0)
    partial_sums[blockIdx.x] = s_part[0];
}

} // namespace cu128_internal

using namespace cu128_internal;
namespace cuda128{

inline uint128_t reduce64to128(uint64_t * a, size_t len)
{
  uint128_t sum, * partial_sums;
  const uint32_t blockSize = 256;
  uint32_t blocks = (len >> 10) / blockSize + 1;

  cudaMalloc(&partial_sums, blocks*sizeof(uint128_t));

  reduce_to_128_kernel<uint64_t, blockSize><<<blocks, blockSize>>>(a, len, partial_sums);
  if(blocks > 1)
    reduce_to_128_kernel<uint128_t, blockSize><<<1, blockSize>>>(partial_sums, blocks, partial_sums);

  cudaMemcpy(&sum, partial_sums, sizeof(uint128_t), cudaMemcpyDeviceToHost);

  return sum;
}

inline uint128_t reduce64to128(int64_t * a, size_t len)
{
  uint128_t sum, * partial_sums;
  const uint32_t blockSize = 256;
  uint32_t blocks = (len >> 10) / blockSize + 1;

  cudaMalloc(&partial_sums, blocks*sizeof(uint128_t));

  reduce_to_128_kernel<int64_t, blockSize><<<blocks, blockSize>>>(a, len, partial_sums);
  if(blocks > 1)
    reduce_to_128_kernel<uint128_t, blockSize><<<1, blockSize>>>(partial_sums, blocks, partial_sums);

  cudaMemcpy(&sum, partial_sums, sizeof(uint128_t), cudaMemcpyDeviceToHost);

  return sum;
}

inline uint128_t reduce128to128(uint128_t * a, size_t len)
{
  uint128_t sum, * partial_sums;
  const uint32_t blockSize = 256;
  uint32_t blocks = (len >> 10) / blockSize + 1;

  cudaMalloc(&partial_sums, blocks*sizeof(uint128_t));

  reduce_to_128_kernel<uint128_t, blockSize><<<blocks, blockSize>>>(a, len, partial_sums);
  if(blocks > 1)
    reduce_to_128_kernel<uint128_t, blockSize><<<1, blockSize>>>(partial_sums, blocks, partial_sums);

  cudaMemcpy(&sum, partial_sums, sizeof(uint128_t), cudaMemcpyDeviceToHost);

  return sum;
}

} // namespace cuda128

#endif
