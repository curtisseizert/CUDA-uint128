#include <stdint.h>
#include <cuda.h>
#include <math.h>

#ifdef __CUDA_ARCH__
#include <math_functions.h>
#endif

// this will likely have to exist only as a header in order to allow
// the compiler to do its thing effectively

#ifndef _UINT128_T_CUDA
#define _UINT128_T_CUDA

#include "uint128_t.cuh"

__host__ __device__ bool uint128_t::isEqualTo(uint128_t a, uint128_t b)
{
  if(a.lo == b.lo && a.hi == b.hi) return 1;
  else return 0;
}

__host__ __device__ bool uint128_t::isNotEqualTo(uint128_t a, uint128_t b)
{
  if(a.lo != b.lo || a.hi != b.hi) return 1;
  else return 0;
}

__host__ __device__ bool uint128_t::isGreaterThan(uint128_t a, uint128_t b)
{
  if(a.hi < b.hi) return 0;
  if(a.hi > b.hi) return 1;
  if(a.lo <= b.lo) return 0;
  else return 1;
}

__host__ __device__ bool uint128_t::isLessThan(uint128_t a, uint128_t b)
{
  if(a.hi < b.hi) return 1;
  if(a.hi > b.hi) return 0;
  if(a.lo < b.lo) return 1;
  else return 0;
}

__host__ __device__ bool uint128_t::isGreaterThanOrEqual(uint128_t a, uint128_t b)
{
  if(a.hi < b.hi) return 0;
  if(a.hi > b.hi) return 1;
  if(a.lo < b.lo) return 0;
  else return 1;
}

__host__ __device__ bool uint128_t::isLessThanOrEqual(uint128_t a, uint128_t b)
{
  if(a.hi < b.hi) return 1;
  if(a.hi > b.hi) return 0;
  if(a.lo <= b.lo) return 1;
  else return 0;
}

__host__ __device__ uint128_t & uint128_t::operator=(const uint128_t & n)
{
  lo = n.lo;
  hi = n.hi;
  return * this;
}

// Code taken from Hacker's Delight:
// http://www.hackersdelight.org/HDcode/divlu.c.
// License permits inclusion here per:
// http://www.hackersdelight.org/permissions.htm
//
// Actually taken from libdivide, which took the
// code from Hacker's Delight

__host__ __device__ uint64_t uint128_t::div128(uint128_t x, uint64_t v, uint64_t * r)
{
  const uint64_t b = 1ull << 32;
  uint64_t  un1, un0,
            vn1, vn0,
            q1, q0,
            un64, un21, un10,
            rhat;
  int s;

  if(x.hi >= v){
    if( r != NULL) *r = (uint64_t) -1;
    return  (uint64_t) -1;
  }

#ifdef __CUDA_ARCH__
  s = __clzll(v);
#else
  s = clzll(v);
#endif

  if(s > 0){
    v = v << s;
    un64 = (x.hi << s) | ((x.lo >> (64 - s)) & (-s >> 31));
    un10 = x.lo << s;
  }else{
    un64 = x.lo | x.hi;
    un10 = x.lo;
  }

  vn1 = v >> 32;
  vn0 = v & 0xffffffff;

  un1 = un10 >> 32;
  un0 = un10 & 0xffffffff;

  q1 = un64/vn1;
  rhat = un64 - q1*vn1;

again1:
  if (q1 >= b || q1*vn0 > b*rhat + un1){
    q1 -= 1;
    rhat = rhat + vn1;
    if(rhat < b) goto again1;
   }

   un21 = un64*b + un1 - q1*v;

   q0 = un21/vn1;
   rhat = un21 - q0*vn1;
again2:
  if(q0 >= b || q0 * vn0 > b*rhat + un0){
    q0 = q0 - 1;
    rhat = rhat + vn1;
    if(rhat < b) goto again2;
  }

  if(r != NULL) *r = (un21*b + un0 - q0*v) >> s;
  return q1*b + q0;
}


__host__ __device__ uint128_t uint128_t::add128(uint128_t x, uint64_t y)
{
  uint128_t res;
#ifdef __CUDA_ARCH__
  asm(  "add.cc.u64    %0 %2 %4\n\t"
        "addc.u64      %1 %3 0\n\t"
        : "=l" (res.lo) "=l" (res.hi)
        : "l" (x.lo) "l" (x.hi)
          "l" (y));
#else
  asm(  "add    %3, %0\n\t"
        "adc    %5, %1\n\t"
        : "=r" (res.lo), "=r" (res.hi)
        : "%0" (x.lo), "%1" (x.hi),
          "r" (y), "r" (0ull)
        :  "cc");
#endif
  return res;
}

__host__ __device__ uint128_t uint128_t::add128(uint128_t x, uint128_t y)
{
   uint128_t res;
 #ifdef __CUDA_ARCH__
   asm( "add.cc.u64    %0 %2 %4;\n\t"
        "addc.u64      %1 %3 %5\n\t"
        : "=l" (res.lo), "=l" (res.hi)
        : "l" (x.lo), "l" (x.hi),
          "l" (y.lo), "l" (y.hi));
#else
    asm("add    %3, %0\n\t"
        "adc    %5, %1\n\t"
        : "=r" (res.lo), "=r" (res.hi)
        : "%0" (x.lo), "%1" (x.hi),
          "r" (y.lo), "r" (y.hi)
        :  "cc");
#endif
    return res;
}

__host__ __device__ uint128_t uint128_t::mul128(uint64_t x, uint64_t y)
{
  uint128_t res;
#ifdef __CUDA_ARCH__
  asm(  "mul.lo.u64    %0 %2 %3\n\t"
        "mul.hi.u64    %1 %2 %3\n\t"
        : "=l" (res.lo) "=l" (res.hi)
        : "l" (x)
          "l" (y));
#else
  asm ("mulq %3\n\t"
   : "=a" (res.lo), "=d" (res.hi)
   : "%0" (x), "rm" (y));
#endif
  return res;
}

__host__ inline uint64_t uint128_t::clzll(uint64_t x)
{
  uint64_t res;
  asm("lzcnt %1, %0" : "=l" (res) : "l" (x));
  return res;
}

__host__ __device__ uint128_t uint128_t::sub128(uint128_t x, uint128_t y)
{
  uint128_t res;

  res.lo = x.lo - y.lo;
  res.hi = x.hi - y.hi;
  if(x.lo < y.lo) res.hi--;

  return res;
}

__host__ __device__ uint64_t uint128_t::sqrt(uint128_t & x)
{
  int32_t i = 64;
  if(x.hi > pow(2, 58)) return 0;

#ifdef __CUDA_ARCH__
  i -= __clzll(x.hi)/2;
#else
  i -= clzll(x.hi)/2;
#endif
  uint128_t cmp;
  uint64_t res = 1ull << i, err = 1, err_last = 0;
  while(err != err_last){
    err_last = err;
    cmp = mul128(res,res);
    if(cmp > x){
      cmp -= x;
      err = cmp/(2 * res + err);
      res -= err;
    }
    else if(cmp < x){
      cmp = x - cmp;
      err = cmp/(2 * res + err);
      res += err;
    }
    else break;
  }
  return res;
}

#endif
