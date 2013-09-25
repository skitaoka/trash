#include <iostream>

template <typename T>
T gcd(T m, T n)
{
  if (!(m < n)) {
    std::swap(m, n);
  }

  while (T const r = m % n) {
    m = n;
    n = r;
  }

  return n;
}

template <typename T>
T lcm(T m, T n)
{
  return m * n / gcd(m, n);
}

int main()
{
  std::ios::sync_with_stdio(false);

  std::cout << gcd(2, 3) << std::endl;
  std::cout << gcd(11221*2, 1233*2) << std::endl;
  std::cout << gcd(11*3, 14*3) << std::endl;
  std::cout << gcd(11*8, 14*8) << std::endl;
  std::cout << lcm(123, 32) << std::endl;
  std::cout << lcm(2, 5) << std::endl;

  return 0;
}
