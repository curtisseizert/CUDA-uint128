#include <stdint.h>
#include <iostream>
#include <omp.h>

#include "cuda_uint128.h"

uint128_t calc(char * argv);

int main(int argc, char ** argv)
{
  uint128_t x = (uint128_t) 1 << 90;

  if(argc == 2)
    x = calc(argv[1]);

  // #pragma omp parallel for
  // for(uint64_t v = 2; v < 1u << 30; v++){
  //   uint64_t r;
  //   uint128_t y = uint128_t::div128to128(x, v, &r);
  //   uint128_t z = mul128(y, v) + r;
  //
  //   if(z != x) std::cout << z << std::endl;
  //
  // }
  //
  // std::cout << x << " " << y << " " << r << std::endl;
  // std::cout << z << std::endl;
  //
  // v = _isqrt(x - v);
  // z = mul128(v, v);
  // std::cout << z << " " << v << std::endl;

  std::cout << _iqrt(x) << std::endl;

  return 0;

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
