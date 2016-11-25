/*

  This header file contains definitions for inline functions and templates
  defining the uint128_t class for both device and host functions.  All of the
  usual operators are overloaded, including (for host) ostream insert.  Strings
  can also be converted to uint128_t (again, host only) in order to provide a
  way to get uint128_t variables from command line arguments.  If you think of
  something better than what is here, or something doesn't work, please let me
  know!!

  cuda_uint128.h (c) Curtis Seizert 2016

*/

#ifndef _UINT128_T_CUDA_H
#define _UINT128_T_CUDA_H

#include <iostream>
#include <iomanip>
#include <cinttypes>
#include <cuda.h>
#include <math.h>
#include <string>

#ifdef __CUDA_ARCH__
#include <math_functions.h>
#endif

class uint128_t{
public:
  uint64_t lo = 0, hi = 0; // d == most significant bits

  __host__ __device__ uint128_t(){};


                    ////////////////
                    //  Operators //
                    ////////////////


  template<typename T>
  __host__ __device__ uint128_t(const T & a){this->lo = a;}

  __host__ __device__ static inline uint64_t u128tou64(uint128_t x){return x.lo;}

  __host__ __device__ uint128_t & operator=(const uint128_t & n)
  {
    lo = n.lo;
    hi = n.hi;
    return * this;
  }

// operator overloading
  template <typename T>
  __host__ __device__ uint128_t & operator=(const T n){hi = 0; lo = n; return * this;}

  template <typename T>
  __host__ __device__ friend uint128_t operator+(uint128_t a, const T & b){return add128(a, b);}

  template <typename T>
  __host__ __device__ inline uint128_t & operator+=(const T & b)
  {
    uint128_t temp = (uint128_t) b;
   #ifdef __CUDA_ARCH__
    asm(  "add.cc.u64    %0 %2 %4;\n\t"
          "addc.u64      %1 %3 %5\n\t"
          : "=l" (lo), "=l" (hi)
          : "l" (lo), "l" (hi),
            "l" (temp.lo), "l" (temp.hi));
    return *this;
  #else
    asm(  "add    %q2, %q0\n\t"
          "adc    %q3, %q1\n\t"
          : "+r" (lo), "+r" (hi)
          : "r" (temp.lo), "r" (temp.hi)
          : "cc");
    return *this;
  #endif
  }

  template <typename T>
  __host__ __device__ inline uint128_t & operator-=(const T & b)
  {
    uint128_t temp = (uint128_t)b;
    if(lo < temp.lo) hi--;
    lo -= temp.lo;
    return * this;
  }

  template <typename T>
  __host__ __device__ inline uint128_t & operator>>=(const T & b)
  {
    lo = (lo >> b) | (hi << (int)(b - 64));
    (b < 64) ? hi >>= b : hi = 0;
    return *this;
  }

  template <typename T>
  __host__ __device__ inline uint128_t & operator<<=(const T & b)
  {
    hi = (hi << b) | (lo << (int)(b - 64));
    (b < 64) ? lo <<= b : lo = 0;
    return *this;
  }

  template <typename T>
  __host__ __device__ friend inline uint128_t operator>>(uint128_t a, const T & b){a >>= b; return a;}

  template <typename T>
  __host__ __device__ friend inline uint128_t operator<<(uint128_t a, const T & b){a <<= b; return a;}

  __host__ __device__ inline uint128_t & operator--(){return *this -=1;}
  __host__ __device__ inline uint128_t & operator++(){return *this +=1;}

  template <typename T>
  __host__ __device__ friend uint128_t operator-(uint128_t a, const T & b){return sub128(a, (uint128_t)b);}

  template <typename T>
  __host__ __device__ friend T operator/(uint128_t x, const T & v){return div128(x, (uint64_t)v);}

  template <typename T>
  __host__ __device__ friend T operator%(uint128_t x, const T & v)
  {
    uint64_t res;
    div128(x, v, &res);
    return (T)res;
  }

  __host__ __device__ friend bool operator<(uint128_t a, uint128_t b){return isLessThan(a, b);}
  __host__ __device__ friend bool operator>(uint128_t a, uint128_t b){return isGreaterThan(a, b);}
  __host__ __device__ friend bool operator<=(uint128_t a, uint128_t b){return isLessThanOrEqual(a, b);}
  __host__ __device__ friend bool operator>=(uint128_t a, uint128_t b){return isGreaterThanOrEqual(a, b);}
  __host__ __device__ friend bool operator==(uint128_t a, uint128_t b){return isEqualTo(a, b);}
  __host__ __device__ friend bool operator!=(uint128_t a, uint128_t b){return isNotEqualTo(a, b);}

  template <typename T>
  __host__ __device__ friend uint128_t operator|(uint128_t a, const T & b){return bitwiseOr(a, (uint128_t)b);}

  template <typename T>
  __host__ __device__ uint128_t & operator|=(const T & b){*this = *this | b; return *this;}

  template <typename T>
  __host__ __device__ friend uint128_t operator&(uint128_t a, const T & b){return bitwiseAnd(a, (uint128_t)b);}

  template <typename T>
  __host__ __device__ uint128_t & operator&=(const T & b){*this = *this & b; return *this;}

  template <typename T>
  __host__ __device__ friend uint128_t operator^(uint128_t a, const T & b){return bitwiseXor(a, (uint128_t)b);}

  template <typename T>
  __host__ __device__ uint128_t & operator^=(const T & b){*this = *this ^ b; return *this;}

  __host__ __device__ friend uint128_t operator~(uint128_t a){return bitwiseNot(a);}


                      ////////////////////
                      //    Comparisons
                      ////////////////////


  __host__ __device__ static  bool isLessThan(uint128_t a, uint128_t b)
  {
    if(a.hi < b.hi) return 1;
    if(a.hi > b.hi) return 0;
    if(a.lo < b.lo) return 1;
    else return 0;
  }

  __host__ __device__ static  bool isLessThanOrEqual(uint128_t a, uint128_t b)
  {
    if(a.hi < b.hi) return 1;
    if(a.hi > b.hi) return 0;
    if(a.lo <= b.lo) return 1;
    else return 0;
  }

  __host__ __device__ static  bool isGreaterThan(uint128_t a, uint128_t b)
  {
    if(a.hi < b.hi) return 0;
    if(a.hi > b.hi) return 1;
    if(a.lo <= b.lo) return 0;
    else return 1;
  }

  __host__ __device__ static  bool isGreaterThanOrEqual(uint128_t a, uint128_t b)
  {
    if(a.hi < b.hi) return 0;
    if(a.hi > b.hi) return 1;
    if(a.lo < b.lo) return 0;
    else return 1;
  }

  __host__ __device__ static  bool isEqualTo(uint128_t a, uint128_t b)
  {
    if(a.lo == b.lo && a.hi == b.hi) return 1;
    else return 0;
  }

  __host__ __device__ static  bool isNotEqualTo(uint128_t a, uint128_t b)
  {
    if(a.lo != b.lo || a.hi != b.hi) return 1;
    else return 0;
  }

  __host__ __device__ static uint128_t min(uint128_t a, uint128_t b)
  {
    return a < b ? a : b;
  }

  __host__ __device__ static uint128_t max(uint128_t a, uint128_t b)
  {
    return a > b ? a : b;
  }


                      //////////////////////
                      //   bit operations
                      //////////////////////

/// This relies on a non-archaic host x86 architecture for the lzcnt instruction
/// and counts leading zeros for 64 bit unsigned integers.  It is used internally
/// in a few of the functions defined below.
  __host__ __device__ static inline uint64_t clzll(uint64_t x)
  {
    uint64_t res;
  #ifdef __CUDA_ARCH__
    res = __clzll(x);
  #else
    asm("lzcnt %1, %0" : "=l" (res) : "l" (x));
  #endif
    return res;
  }

/// This just makes it more convenient to count leading zeros for uint128_t
  __host__ __device__ static inline uint64_t clz128(uint128_t x)
  {
    uint64_t res;

    res = x.hi != 0 ? clzll(x.hi) : 64 + clzll(x.lo);

    return res;
  }

  __host__ __device__ static  uint128_t bitwiseOr(uint128_t a, uint128_t b)
  {
    a.lo |= b.lo;
    a.hi |= b.hi;
    return a;
  }

  __host__ __device__ static  uint128_t bitwiseAnd(uint128_t a, uint128_t b)
  {
    a.lo &= b.lo;
    a.hi &= b.hi;
    return a;
  }

  __host__ __device__ static  uint128_t bitwiseXor(uint128_t a, uint128_t b)
  {
    a.lo ^= b.lo;
    a.hi ^= b.hi;
    return a;
  }

  __host__ __device__ static  uint128_t bitwiseNot(uint128_t a)
  {
    a.lo = ~a.lo;
    a.hi = ~a.hi;
    return a;
  }

                          //////////////////
                          //   arithmetic
                          //////////////////

  __host__ __device__ static inline uint128_t add128(uint128_t x, uint128_t y)
  {
   #ifdef __CUDA_ARCH__
    uint128_t res;
    asm(  "add.cc.u64    %0 %2 %4;\n\t"
          "addc.u64      %1 %3 %5\n\t"
          : "=l" (res.lo), "=l" (res.hi)
          : "l" (x.lo), "l" (x.hi),
            "l" (y.lo), "l" (y.hi));
    return res;
  #else
    asm(  "add    %q2, %q0\n\t"
          "adc    %q3, %q1\n\t"
          : "+r" (x.lo), "+r" (x.hi)
          : "r" (y.lo), "r" (y.hi)
          : "cc");
    return x;
  #endif
  }

  __host__ __device__ static inline uint128_t add128(uint128_t x, uint64_t y)
  {
  #ifdef __CUDA_ARCH__
    uint128_t res;
    asm(  "add.cc.u64    %0 %2 %4\n\t"
          "addc.u64      %1 %3 0\n\t"
          : "=l" (res.lo) "=l" (res.hi)
          : "l" (x.lo) "l" (x.hi)
            "l" (y));
    return res;
  #else
    asm(  "add    %q2, %q0\n\t"
          "adc    $0, %q1\n\t"
          : "+r" (x.lo), "+r" (x.hi)
          : "r" (y)
          : "cc");
    return x;
  #endif
  }

  __host__ __device__ static inline uint128_t mul128(uint64_t x, uint64_t y)
  {
    uint128_t res;
  #ifdef __CUDA_ARCH__
    // asm(  "mul.lo.u64    %0 %2 %3\n\t"
    //       "mul.hi.u64    %1 %2 %3\n\t"
    //       : "=l" (res.lo) "=l" (res.hi)
    //       : "l" (x)
    //         "l" (y));
    res.lo = x * y;
    res.hi = __mul64hi(x, y);
  #else
    asm( "mulq %3\n\t"
         : "=a" (res.lo), "=d" (res.hi)
         : "%0" (x), "rm" (y));
  #endif
    return res;
  }

  __host__ __device__ static inline uint128_t mul128(uint128_t x, uint64_t y)
  {
    uint128_t res;
  #ifdef __CUDA_ARCH__
    asm(  "mul.lo.u64     %0  %2  %4\n\t"
          "mul.hi.u64     %1  %2  %4\n\t"
          "mad.lo.u64     %1  %3  %4  %1\n\t"
          : "=l" (res.lo) "=l" (res.hi)
          : "l" (x.lo) "l" (x.hi)
            "l" (y));
  #else
    asm( "mulq %3\n\t"
         : "=a" (res.lo), "=d" (res.hi)
         : "%0" (x.lo), "rm" (y));
    res.hi += x.hi * y;
  #endif
    return res;
  }

// taken from libdivide's adaptation of this implementation origininally in
// Hacker's Delight: http://www.hackersdelight.org/hdcodetxt/divDouble.c.txt
// License permits inclusion here per:
// http://www.hackersdelight.org/permissions.htm
  __host__ __device__ static inline uint64_t div128(uint128_t x, uint64_t v, uint64_t * r = NULL) // x / v
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

    s = clzll(v);

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

  __host__ __device__ static inline uint128_t div128to128(uint128_t x, uint64_t v, uint64_t * r = NULL)
  {
    uint128_t res;

    res.hi = x.hi/v;
    x.hi %= v;
    res.lo = div128(x, v, r);

    return res;
  }

  __host__ __device__ static inline uint128_t sub128(uint128_t x, uint128_t y) // x - y
  {
    uint128_t res;

    res.lo = x.lo - y.lo;
    res.hi = x.hi - y.hi;
    if(x.lo < y.lo) res.hi--;

    return res;
  }

///
/// This integer square root operation needs some love.  Because it is garbage.
///
/*
  __host__ __device__ static  uint64_t sqrt(const uint128_t & x)
  {
    int32_t i = 64;
    if(x.hi > pow(2, 58)) return 0;

  // #ifdef __CUDA_ARCH__
  //   i -= __clzll(x.hi)/2;
  // #else
    i -= clzll(x.hi)/2;
  // #endif
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
*/
  __host__ __device__ static uint64_t sqrt(const uint128_t & x)
  {
    uint64_t res0 = 0, res1 = 0;

    #ifdef __CUDA_ARCH__
    res0 = sqrtf(u128_to_float(x));
    while(res0 != res1){
      res1 = (res0 + x/res0) >> 1;
      res0 = (res1 + x/res1) >> 1;
      res1 = (res0 + x/res0) >> 1;
      res0 = (res1 + x/res1) >> 1;
    }
    #else
    res0 = std::sqrt(u128_to_float(x));
    while(res0 != res1){
      res1 = (res0 + x/res0) >> 1;
      res0 = (res1 + x/res1) >> 1;
      res1 = (res0 + x/res0) >> 1;
      res0 = (res1 + x/res1) >> 1;
    }
    #endif

    return res0;
  }

  // __host__ __device__ static  uint64_t cbrt(const uint128_t & x)
  // {
  //   uint128_t cmp;
  //   uint64_t res, res2, err=1, err0=0, i;
  //   if(x.hi > pow(2, 32)) return 0;
  //
  //   i = 128 - clzll(x.hi);
  //   cmp = x;
  //   cmp >>= (2 * i / 3);
  //   res = cmp.lo;
  //
  //   while(err0 != err){
  //     err0 = err;
  //     res2 = res * res;
  //     cmp = mul128(res2, res);
  //     if(cmp > x){
  //       cmp -= x;
  //       err = cmp / (4 * res);
  //       err /= res;
  //       res -= err;
  //       res -= 1;
  //     }else if(cmp < x){
  //       cmp = x - cmp;
  //       err = cmp / (4 * res);
  //       err /= res;
  //       res += err;
  //       res += 1;
  //     }else break;
  //     if(err <= 1) break;
  // #ifndef __CUDA_ARCH__
  //     std::cout << "\t" << x << "\t" << res << "\t" << err << "\t" << cmp << std::endl;
  // #endif
  //   }
  //
  //   return res;
  // }



                            /////////////////
                            //  typecasting
                            /////////////////

  __host__ static inline uint128_t stou128(std::string s)
  {
    uint128_t res = 0;
    for(std::string::iterator iter = s.begin(); iter != s.end() && (int) *iter >= 48; iter++){
      res = mul128(res, 10);
      res += (uint16_t) *iter - 48;
    }
    return res;
  }

  __host__ __device__ static inline double u128_to_double(uint128_t x)
  {
    double dbl;
    #ifdef __CUDA_ARCH__
    if(x.hi == 0) return __ull2double_rd(x.lo);
    #else
    if(x.hi == 0) return (double) x.lo;
    #endif
    uint64_t r = clzll(x.hi);
    x <<= r;

    #ifdef __CUDA_ARCH__
    dbl = __ull2double_rd(x.hi);
    #else
    dbl = (double) x.lo;
    #endif

    dbl *= (1ull << (64-r));

    return dbl;
  }

  __host__ __device__ static inline double u128_to_float(uint128_t x)
  {
    float flt;
    #ifdef __CUDA_ARCH__
    if(x.hi == 0) return __ull2float_rd(x.lo);
    #else
    if(x.hi == 0) return (float) x.lo;
    #endif
    uint64_t r = clzll(x.hi);
    x <<= r;

    #ifdef __CUDA_ARCH__
    flt = __ull2float_rd(x.hi);
    #else
    flt = (float) x.hi;
    #endif

    flt *= (1ull << (64-r));
    flt *= 2;

    return flt;
  }

  __host__ __device__ static inline uint128_t double_to_u128(double dbl)
  {
    uint128_t x;
    if(dbl < 1 || dbl > 1e39) return 0;
    else{

  #ifdef __CUDA_ARCH__
      uint32_t shft = __double2uint_rd(log2(dbl));
      uint64_t divisor = 1ull << shft;
      dbl /= divisor;
      x.lo = __double2ull_rd(dbl);
      x <<= shft;
  #else
      uint32_t shft = (uint32_t) log2(dbl);
      uint64_t divisor = 1ull << shft;
      dbl /= divisor;
      x.lo = (uint64_t) dbl;
      x <<= shft;
  #endif
      return x;
    }
  }

  __host__ __device__ static inline uint128_t float_to_u128(float flt)
  {
    uint128_t x;
    if(flt < 1 || flt > 1e39) return 0;
    else{

  #ifdef __CUDA_ARCH__
      uint32_t shft = __double2uint_rd(log2(flt));
      uint64_t divisor = 1ull << shft;
      flt /= divisor;
      x.lo = __double2ull_rd(flt);
      x <<= shft;
  #else
      uint32_t shft = (uint32_t) log2(flt);
      uint64_t divisor = 1ull << shft;
      flt /= divisor;
      x.lo = (uint64_t) flt;
      x <<= shft;
  #endif
    return x;
    }
  }

                              //////////////
                              //  iostream
                              //////////////


  __host__ friend inline std::ostream & operator<<(std::ostream & out, uint128_t x)
  {
    if(x.hi != 0){
      uint64_t left, right, divide = 1000000000000000000; // 10^18
      right = x % divide;
      left = x / divide;
      out << left << std::setfill('0') << std::setw(18) << right;
    }else{
      out << x.lo;
    }
    return out;
  }

}; // class uint128_t

#endif
