#include <iostream>

int main()
{
  typedef unsigned long long int ulong_t;

  // ビットが立っている部分集合を列挙する
  ulong_t const d = 0x7;

  ulong_t n = 0;
  do {
    std::cout << n << std::endl;
    n = (n - d) & d;
  } while (n);

  return 0;
}
