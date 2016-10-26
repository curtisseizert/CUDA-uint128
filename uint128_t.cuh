/*

This is the header file to include in headers that require declaration of
uint128_t.


*/

#ifndef _UINT128_T_CUDA_H
#define _UINT128_T_CUDA_H

#include <iostream>
#include <iomanip>
#include <cinttypes>

class uint128_t{
private:
  uint64_t lo = 0, hi = 0; // d == most significant bits
public:
  __host__ __device__ uint128_t(){};

  template<typename T>
  __host__ __device__ uint128_t(const T & a){this->lo = a;}

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

  template <uint64_t> __host__ __device__ uint128_t & operator+=(const uint64_t & b);
  template <uint32_t> __host__ __device__ uint128_t & operator+=(const uint32_t & b);

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

// comparisons
  __host__ __device__ static  bool isLessThan(uint128_t a, uint128_t b);
  __host__ __device__ static  bool isLessThanOrEqual(uint128_t a, uint128_t b);
  __host__ __device__ static  bool isGreaterThan(uint128_t a, uint128_t b);
  __host__ __device__ static  bool isGreaterThanOrEqual(uint128_t a, uint128_t b);
  __host__ __device__ static  bool isEqualTo(uint128_t a, uint128_t b);
  __host__ __device__ static  bool isNotEqualTo(uint128_t a, uint128_t b);

// bitwise arithmetic
  __host__ __device__ static  uint128_t bitwiseOr(uint128_t a, uint128_t b);
  __host__ __device__ static  uint128_t bitwiseAnd(uint128_t a, uint128_t b);
  __host__ __device__ static  uint128_t bitwiseXor(uint128_t a, uint128_t b);
  __host__ __device__ static  uint128_t bitwiseNot(uint128_t a);

// arithmetic
  __host__ __device__ static  uint128_t add128(uint128_t x, uint128_t y);
  __host__ __device__ static  uint128_t add128(uint128_t x, uint64_t y);
  __host__ __device__ static   uint128_t mul128(uint64_t x, uint64_t y);
  __host__ __device__ static   uint64_t div128(uint128_t x, uint64_t v, uint64_t * r = NULL); // x / v
  __host__ __device__ static  uint128_t sub128(uint128_t x, uint128_t y); // x - y
  __host__ __device__ static  uint64_t sqrt(const uint128_t & x);
  __host__ __device__ static  uint64_t cbrt(const uint128_t & x);

// bit operations
  __host__ __device__ uint64_t static clzll(uint64_t a);

// iostream
  __host__ friend std::ostream & operator<<(std::ostream & out, uint128_t x)
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
