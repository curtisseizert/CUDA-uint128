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
#include <cmath>
#include <string>
#include <vector>
#include <iterator>

#ifdef __CUDA_ARCH__
#include <math_functions.h>
#endif

#ifdef __has_builtin
# define uint128_t_has_builtin(x) __has_builtin(x)
#else
# define uint128_t_has_builtin(x) 0
#endif

class uint128_t{
public:
#ifdef __CUDA_ARCH__ // dynamic initialization not supported in some device code
  uint64_t lo, hi;
#else
  uint64_t lo = 0, hi = 0;
#endif
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  uint128_t(){};


                    ////////////////
                    //  Operators //
                    ////////////////

  template<typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  uint128_t(const T & a)
  {
    this->lo = (uint64_t) a & (uint64_t)-1;
    this->hi = 0;
  }

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
 static inline uint64_t u128tou64(uint128_t x){return x.lo;}

 #ifdef __CUDA_ARCH__
   __host__ __device__
 #endif
   uint128_t & operator=(const uint128_t & n)
  {
    lo = n.lo;
    hi = n.hi;
    return * this;
  }

// operator overloading
  template <typename T>
  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    uint128_t & operator=(const T n){hi = 0; lo = n; return * this;}

  template <typename T>
  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    friend uint128_t operator+(uint128_t a, const T & b){return add128(a, b);}

  template <typename T>
  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    inline uint128_t & operator+=(const T & b)
  {
    uint128_t temp = (uint128_t) b;
   #ifdef __CUDA_ARCH__
    asm(  "add.cc.u64    %0, %2, %4;\n\t"
          "addc.u64      %1, %3, %5;\n\t"
          : "=l" (lo), "=l" (hi)
          : "l" (lo), "l" (hi),
            "l" (temp.lo), "l" (temp.hi));
    return *this;
  #elif __x86_64__
    asm(  "add    %q2, %q0\n\t"
          "adc    %q3, %q1\n\t"
          : "+r" (lo), "+r" (hi)
          : "r" (temp.lo), "r" (temp.hi)
          : "cc");
    return *this;
  #elif __aarch64__
    asm(  "adds   %0, %2, %4\n\t"
          "adc    %1, %3, %5\n\t"
          : "=&r" (lo), "=r" (hi)
          : "r" (lo), "r" (hi),
            "r" (temp.lo), "r" (temp.hi)
          : "cc");
    return *this;
  #else
  # error Architecture not supported
  #endif
  }

  template <typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  inline uint128_t & operator-=(const T & b)
  {
    uint128_t temp = (uint128_t)b;
    if(lo < temp.lo) hi--;
    lo -= temp.lo;
    return * this;
  }

  template <typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  inline uint128_t & operator>>=(const T & b)
  {
    lo = (lo >> b) | (hi << (int)(b - 64));
    (b < 64) ? hi >>= b : hi = 0;
    return *this;
  }

  template <typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  inline uint128_t & operator<<=(const T & b)
  {
    hi = (hi << b) | (lo << (int)(b - 64));
    (b < 64) ? lo <<= b : lo = 0;
    return *this;
  }

template <typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  friend inline uint128_t operator>>(uint128_t a, const T & b){a >>= b; return a;}

template <typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  friend inline uint128_t operator<<(uint128_t a, const T & b){a <<= b; return a;}

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  inline uint128_t & operator--(){return *this -=1;}
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  inline uint128_t & operator++(){return *this +=1;}

  template <typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
    friend uint128_t operator-(uint128_t a, const T & b){return sub128(a, (uint128_t)b);}

  template <typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
    friend uint128_t operator*(uint128_t a, const T & b){return mul128(a, (uint64_t)b);}

  template <typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
    friend T operator/(uint128_t x, const T & v){return div128to64(x, (uint64_t)v);}

  template <typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  friend T operator%(uint128_t x, const T & v)
  {
    uint64_t res;
    div128to64(x, v, &res);
    return (T)res;
  }

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  friend bool operator<(uint128_t a, uint128_t b){return isLessThan(a, b);}
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  friend bool operator>(uint128_t a, uint128_t b){return isGreaterThan(a, b);}
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  friend bool operator<=(uint128_t a, uint128_t b){return isLessThanOrEqual(a, b);}
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  friend bool operator>=(uint128_t a, uint128_t b){return isGreaterThanOrEqual(a, b);}
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  friend bool operator==(uint128_t a, uint128_t b){return isEqualTo(a, b);}
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
friend bool operator!=(uint128_t a, uint128_t b){return isNotEqualTo(a, b);}

template <typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  friend uint128_t operator|(uint128_t a, const T & b){return bitwiseOr(a, (uint128_t)b);}

template <typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  uint128_t & operator|=(const T & b){*this = *this | b; return *this;}

template <typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  friend uint128_t operator&(uint128_t a, const T & b){return bitwiseAnd(a, (uint128_t)b);}

template <typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  uint128_t & operator&=(const T & b){*this = *this & b; return *this;}

template <typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  friend uint128_t operator^(uint128_t a, const T & b){return bitwiseXor(a, (uint128_t)b);}

template <typename T>
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  uint128_t & operator^=(const T & b){*this = *this ^ b; return *this;}

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  friend uint128_t operator~(uint128_t a){return bitwiseNot(a);}


                      ////////////////////
                      //    Comparisons
                      ////////////////////


  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    static  bool isLessThan(uint128_t a, uint128_t b)
  {
    if(a.hi < b.hi) return 1;
    if(a.hi > b.hi) return 0;
    if(a.lo < b.lo) return 1;
    else return 0;
  }

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    static  bool isLessThanOrEqual(uint128_t a, uint128_t b)
  {
    if(a.hi < b.hi) return 1;
    if(a.hi > b.hi) return 0;
    if(a.lo <= b.lo) return 1;
    else return 0;
  }

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    static  bool isGreaterThan(uint128_t a, uint128_t b)
  {
    if(a.hi < b.hi) return 0;
    if(a.hi > b.hi) return 1;
    if(a.lo <= b.lo) return 0;
    else return 1;
  }

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    static  bool isGreaterThanOrEqual(uint128_t a, uint128_t b)
  {
    if(a.hi < b.hi) return 0;
    if(a.hi > b.hi) return 1;
    if(a.lo < b.lo) return 0;
    else return 1;
  }

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    static  bool isEqualTo(uint128_t a, uint128_t b)
  {
    if(a.lo == b.lo && a.hi == b.hi) return 1;
    else return 0;
  }

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    static  bool isNotEqualTo(uint128_t a, uint128_t b)
  {
    if(a.lo != b.lo || a.hi != b.hi) return 1;
    else return 0;
  }

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    friend uint128_t min(uint128_t a, uint128_t b)
  {
    return a < b ? a : b;
  }

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    friend uint128_t max(uint128_t a, uint128_t b)
  {
    return a > b ? a : b;
  }


                      //////////////////////
                      //   bit operations
                      //////////////////////

/// This counts leading zeros for 64 bit unsigned integers.  It is used internally
/// in a few of the functions defined below.
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  static inline int clz64(uint64_t x)
  {
    int res;
  #ifdef __CUDA_ARCH__
    res = __clzll(x);
  #elif __GNUC__ || uint128_t_has_builtin(__builtin_clzll)
    res = __builtin_clzll(x);
  #elif __x86_64__
    asm("bsr %1, %0\nxor $0x3f, %0" : "=r" (res) : "rm" (x) : "cc", "flags");
  #elif __aarch64__
    asm("clz %0, %1" : "=r" (res) : "r" (x));
  #else
  # error Architecture not supported
  #endif
    return res;
  }

/// This just makes it more convenient to count leading zeros for uint128_t
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  friend inline uint64_t clz128(uint128_t x)
  {
    uint64_t res;

    res = x.hi != 0 ? clz64(x.hi) : 64 + clz64(x.lo);

    return res;
  }

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  static  uint128_t bitwiseOr(uint128_t a, uint128_t b)
  {
    a.lo |= b.lo;
    a.hi |= b.hi;
    return a;
  }

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  static  uint128_t bitwiseAnd(uint128_t a, uint128_t b)
  {
    a.lo &= b.lo;
    a.hi &= b.hi;
    return a;
  }

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  static  uint128_t bitwiseXor(uint128_t a, uint128_t b)
  {
    a.lo ^= b.lo;
    a.hi ^= b.hi;
    return a;
  }

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  static  uint128_t bitwiseNot(uint128_t a)
  {
    a.lo = ~a.lo;
    a.hi = ~a.hi;
    return a;
  }

                          //////////////////
                          //   arithmetic
                          //////////////////

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  static inline uint128_t add128(uint128_t x, uint128_t y)
  {
  #ifdef __CUDA_ARCH__
    uint128_t res;
    asm(  "add.cc.u64    %0, %2, %4;\n\t"
          "addc.u64      %1, %3, %5;\n\t"
          : "=l" (res.lo), "=l" (res.hi)
          : "l" (x.lo), "l" (x.hi),
            "l" (y.lo), "l" (y.hi));
    return res;
  #elif __x86_64__
    asm(  "add    %q2, %q0\n\t"
          "adc    %q3, %q1\n\t"
          : "+r" (x.lo), "+r" (x.hi)
          : "r" (y.lo), "r" (y.hi)
          : "cc");
    return x;
  #elif __aarch64__
    uint128_t res;
    asm(  "adds   %0, %2, %4\n\t"
          "adc    %1, %3, %5\n\t"
          : "=&r" (res.lo), "=r" (res.hi)
          : "r" (x.lo), "r" (x.hi),
            "ri" (y.lo), "ri" (y.hi)
          : "cc");
    return res;
  #else
  # error Architecture not supported
  #endif
  }

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  static inline uint128_t add128(uint128_t x, uint64_t y)
  {
  #ifdef __CUDA_ARCH__
    uint128_t res;
    asm(  "add.cc.u64    %0, %2, %4;\n\t"
          "addc.u64      %1, %3, 0;\n\t"
          : "=l" (res.lo) "=l" (res.hi)
          : "l" (x.lo) "l" (x.hi)
            "l" (y));
    return res;
  #elif __x86_64__
    asm(  "add    %q2, %q0\n\t"
          "adc    $0, %q1\n\t"
          : "+r" (x.lo), "+r" (x.hi)
          : "r" (y)
          : "cc");
    return x;
  #elif __aarch64__
    uint128_t res;
    asm(  "adds   %0, %2, %4\n\t"
          "adc    %1, %3, xzr\n\t"
          : "=&r" (res.lo), "=r" (res.hi)
          : "r" (x.lo), "r" (x.hi),
            "ri" (y)
          : "cc");
    return res;
  #else
  # error Architecture not supported
  #endif
  }

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  static inline uint128_t mul128(uint64_t x, uint64_t y)
  {
    uint128_t res;
  #ifdef __CUDA_ARCH__
    res.lo = x * y;
    res.hi = __mul64hi(x, y);
  #elif __x86_64__
    asm( "mulq %3\n\t"
         : "=a" (res.lo), "=d" (res.hi)
         : "%0" (x), "rm" (y));
  #elif __aarch64__
    asm( "mul %0, %2, %3\n\t"
         "umulh %1, %2, %3\n\t"
         : "=&r" (res.lo), "=r" (res.hi)
         : "r" (x), "r" (y));
  #else
  # error Architecture not supported
  #endif
    return res;
  }

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  static inline uint128_t mul128(uint128_t x, uint64_t y)
  {
    uint128_t res;
  #ifdef __CUDA_ARCH__
    res.lo = x.lo * y;
    res.hi = __mul64hi(x.lo, y);
    res.hi += x.hi * y;
  #elif __x86_64__
    asm( "mulq %3\n\t"
         : "=a" (res.lo), "=d" (res.hi)
         : "%0" (x.lo), "rm" (y));
    res.hi += x.hi * y;
  #elif __aarch64__
    res.lo = x.lo * y;
    asm( "umulh %0, %1, %2\n\t"
         : "=r" (res.hi)
         : "r" (x.lo), "r" (y));
    res.hi += x.hi * y;
  #else
  # error Architecture not supported
  #endif
    return res;
  }

// taken from libdivide's adaptation of this implementation origininally in
// Hacker's Delight: http://www.hackersdelight.org/hdcodetxt/divDouble.c.txt
// License permits inclusion here per:
// http://www.hackersdelight.org/permissions.htm
#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  static inline uint64_t div128to64(uint128_t x, uint64_t v, uint64_t * r = NULL) // x / v
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

    s = clz64(v);

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

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    static inline uint128_t div128to128(uint128_t x, uint64_t v, uint64_t * r = NULL)
  {
    uint128_t res;

    res.hi = x.hi/v;
    x.hi %= v;
    res.lo = div128to64(x, v, r);

    return res;
  }
/*
  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    static inline uint64_t div128to64(uint128_t x, uint128_t v, uint128_t * r = NULL)
  {
    if(v.hi == 0) return div128to64(x, v.lo, r->lo);
    uint64_t res;
  }
*/
  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    static inline uint128_t sub128(uint128_t x, uint128_t y) // x - y
  {
    uint128_t res;

    res.lo = x.lo - y.lo;
    res.hi = x.hi - y.hi;
    if(x.lo < y.lo) res.hi--;

    return res;
  }

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    friend inline uint128_t sub128(uint128_t x, uint128_t y);

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
  friend inline uint64_t _isqrt(uint64_t x)
  {
    uint64_t res0 = 0;
  #ifdef __CUDA_ARCH__
    res0 = sqrtf(x);
  #else
    res0 = sqrt(x);
    for(uint16_t i = 0; i < 8; i++)
      res0 = (res0 + x/res0) >> 1;
  #endif
    return res0;
  }

                      //////////////////
                      ///    roots
                      //////////////////

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    static inline uint64_t _isqrt(const uint128_t & x) // this gives errors above 2^124
  {
    uint64_t res0 = 0;

    if(x == 0 || x.hi > 1ull << 60)
      return 0;

    #ifdef __CUDA_ARCH__
    res0 = sqrtf(u128_to_float(x));
    #else
    res0 = std::sqrt(u128_to_float(x));
    #endif
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
    for(uint16_t i = 0; i < 8; i++)
      res0 = (res0 + x/res0) >> 1;

    return res0;
  }

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    friend inline uint64_t _icbrt(const uint128_t & x)
  {
    uint64_t res0 = 0;

  #ifdef __CUDA_ARCH__
    res0 = cbrtf(u128_to_float(x));
  #else
    res0 = std::cbrt(u128_to_float(x));
  #endif
  #ifdef __CUDA_ARCH__
    #pragma unroll
  #endif
    for(uint16_t i = 0; i < 47; i++) // there needs to be an odd number of iterations
                                     // for the case of numbers of the form x^2 - 1
                                     // where this will not converge
      res0 = (res0 + div128to128(x,res0)/res0) >> 1;
    return res0;
  }

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
  // this function is to avoid off by 1 errors from nesting integer square roots
    friend inline uint64_t _iqrt(const uint128_t & x)
  {
    uint64_t res0 = 0, res1 = 0;

    res0 = _isqrt(_isqrt(x));
    res1 = (res0 + div128to128(x,res0*res0)/res0) >> 1;
    res0 = (res1 + div128to128(x,res1*res1)/res1) >> 1;

    return res0 < res1 ? res0 : res1;
  }


                            /////////////////
                            //  typecasting
                            /////////////////
#ifdef __CUDA_ARCH__
  __host__
#endif
 static inline uint128_t string_to_u128(std::string s)
  {
    uint128_t res = 0;
    for(std::string::iterator iter = s.begin(); iter != s.end() && (int) *iter >= 48; iter++){
      res = mul128(res, 10);
      res += (uint16_t) *iter - 48;
    }
    return res;
  }

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    friend inline double u128_to_double(uint128_t x)
  {
    double dbl;
    #ifdef __CUDA_ARCH__
    if(x.hi == 0) return __ull2double_rd(x.lo);
    #else
    if(x.hi == 0) return (double) x.lo;
    #endif
    uint64_t r = clz64(x.hi);
    x <<= r;

    #ifdef __CUDA_ARCH__
    dbl = __ull2double_rd(x.hi);
    #else
    dbl = (double) x.lo;
    #endif

    dbl *= (1ull << (64-r));

    return dbl;
  }

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    friend inline float u128_to_float(uint128_t x)
  {
    float flt;
    #ifdef __CUDA_ARCH__
    if(x.hi == 0) return __ull2float_rd(x.lo);
    #else
    if(x.hi == 0) return (float) x.lo;
    #endif
    uint64_t r = clz64(x.hi);
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

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    static inline uint128_t double_to_u128(double dbl)
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

  #ifdef __CUDA_ARCH__
    __host__ __device__
  #endif
    static inline uint128_t float_to_u128(float flt)
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

#ifdef __CUDA_ARCH__
  __host__
#endif
  friend inline std::ostream & operator<<(std::ostream & out, uint128_t x)
  {
    std::vector<uint16_t> rout;
    uint64_t v = 10, r = 0;
    if (x == 0) {
      out << "0";
      return out;
    }
    do {
      x = div128to128(x, v, &r);
      rout.push_back(r);
    } while(x != 0);
    for(std::reverse_iterator<std::vector<uint16_t>::iterator> rit = rout.rbegin(); rit != rout.rend(); rit++){
      out << *rit;
    }
    return out;
  }

}; // class uint128_t

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  inline uint128_t mul128(uint64_t x, uint64_t y)
{
  return uint128_t::mul128(x, y);
}

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  inline uint128_t mul128(uint128_t x, uint64_t y)
{
  return uint128_t::mul128(x, y);
}

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  inline uint64_t div128to64(uint128_t x, uint64_t v, uint64_t * r = NULL)
{
  return uint128_t::div128to64(x, v, r);
}

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  inline uint128_t div128to128(uint128_t x, uint64_t v, uint64_t * r = NULL)
{
  return uint128_t::div128to128(x, v, r);
}

#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  inline uint128_t add128(uint128_t x, uint128_t y)
{
  return uint128_t::add128(x, y);
}


#ifdef __CUDA_ARCH__
  __host__ __device__
#endif
  inline uint128_t sub128(uint128_t x, uint128_t y) // x - y
{
  return x - y;
}

#ifdef __CUDA_ARCH__
  __host__
#endif
 inline uint128_t string_to_u128(std::string s)
 {
   return uint128_t::string_to_u128(s);
 }

#ifdef __CUDA_ARCH__
 __host__ __device__
#endif
 inline uint64_t _isqrt(const uint128_t & x)
{
 return uint128_t::_isqrt(x);
}

#endif
