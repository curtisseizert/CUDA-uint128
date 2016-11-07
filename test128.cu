#include <iostream>
#include <stdint.h>
#include <math.h>
#include <string>

#include "uint128_t.cuh"

uint128_t calc(char * argv);

int main(int argc, char * argv[])
{
  uint128_t x;
  if(argc == 2){
    x = calc(argv[1]);
  }

  std::cout << x << std::endl;


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

  value = uint128_t::stou128_t(s);

  return value;
}
