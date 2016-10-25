/*

This is the header file to include in headers that require declaration of
uint128_t.


*/

#ifndef _UINT128_T_CUDA_H
#define _UINT128_T_CUDA_H

class uint128_t{
private:
  uint64_t lo = 0, hi = 0; // d == most significant bits
public:
  __host__ __device__ uint128_t(){};

  template<typename T>
  __host__ __device__ uint128_t(const T & a){this->lo = a;}


// operator overloading
  template <typename T>
  __host__ __device__ uint128_t & operator=(const T n){this->lo = n; return * this;}

  template <typename T>
  __host__ __device__ friend uint128_t operator+(uint128_t a, const T & b){return add128(a, b);}

  template <typename T>
  __host__ __device__ uint128_t operator+=(const T & b){return add128(*this, b);}

  template <typename T>
  __host__ __device__ uint128_t operator-=(const T & b){return sub128(*this, b);}

  __host__ __device__ uint128_t & operator=(const uint128_t & n);

  __host__ __device__ friend uint128_t operator-(uint128_t a, uint128_t b){return sub128(a, b);}
  __host__ __device__ friend uint64_t operator/(uint128_t x, const uint64_t & v){return div128(x, v);}
  __host__ __device__ friend bool operator<(uint128_t a, uint128_t b){return isLessThan(a, b);}
  __host__ __device__ friend bool operator>(uint128_t a, uint128_t b){return isGreaterThan(a, b);}
  __host__ __device__ friend bool operator<=(uint128_t a, uint128_t b){return isLessThanOrEqual(a, b);}
  __host__ __device__ friend bool operator>=(uint128_t a, uint128_t b){return isGreaterThanOrEqual(a, b);}
  __host__ __device__ friend bool operator==(uint128_t a, uint128_t b){return isEqualTo(a, b);}
  __host__ __device__ friend bool operator!=(uint128_t a, uint128_t b){return isNotEqualTo(a, b);}

// comparisons
  __host__ __device__ static  bool isLessThan(uint128_t a, uint128_t b);
  __host__ __device__ static  bool isLessThanOrEqual(uint128_t a, uint128_t b);
  __host__ __device__ static  bool isGreaterThan(uint128_t a, uint128_t b);
  __host__ __device__ static  bool isGreaterThanOrEqual(uint128_t a, uint128_t b);
  __host__ __device__ static  bool isEqualTo(uint128_t a, uint128_t b);
  __host__ __device__ static  bool isNotEqualTo(uint128_t a, uint128_t b);

// arithmetic
  __host__ __device__ static  uint128_t add128(uint128_t x, uint128_t y);
  __host__ __device__ static  uint128_t add128(uint128_t x, uint64_t y);
  __host__ __device__ static  uint128_t mul128(uint64_t x, uint64_t y);
  __host__ __device__ static  uint64_t div128(uint128_t x, uint64_t v, uint64_t * r = NULL); // x / v
  __host__ __device__ static  uint128_t sub128(uint128_t x, uint128_t y); // x - y
  __host__ __device__ uint64_t static  sqrt(uint128_t & x);

  __host__ uint64_t static  clzll(uint64_t a);
}; // class uint128_t

#endif
