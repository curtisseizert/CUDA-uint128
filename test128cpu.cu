#include <stdint.h>
#include <iostream>
#include <omp.h>

#include "cuda_uint128.h"

int main()
{
  uint128_t x = (uint128_t) 1 << 90;

  #pragma omp parallel for
  for(uint64_t v = 2; v < 1u << 30; v++){
    uint64_t r;
    uint128_t y = uint128_t::div128to128(x, v, &r);
    uint128_t z = mul128(y, v) + r;

    if(z != x) std::cout << z << std::endl;

  }
  //
  // std::cout << x << " " << y << " " << r << std::endl;
  // std::cout << z << std::endl;
  //
  // v = _isqrt(x - v);
  // z = mul128(v, v);
  // std::cout << z << " " << v << std::endl;

  return 0;

}
