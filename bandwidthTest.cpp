#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

// cl /O2 bandwidthTest.cpp
int main()
{
  std::size_t const unit = 1000;
  std::size_t const size = unit * 1000 * 1000;

  char * const data[2] = {
    new char[size],
    new char[size],
  };

  std::clock_t const start = std::clock();
  std::memcpy(data[1], data[0], size);
  std::clock_t const end = std::clock();
  double const time = (end - start) / double(CLOCKS_PER_SEC);
  std::printf("%g MB/s\n", unit / time);

  return 0;
}
