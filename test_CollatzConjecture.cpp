/**
** コラッツ予想を試すプログラム
** 入力例)
** 5
** 2
** 3
** 2398
** 2385
** 10000
**/

#include <iostream>
#include <iterator>
#include <algorithm>
#include <vector>

template <int Sz = 1000000>
struct solver
{
  solver(): cache_(Sz)
  {
  }

  int operator () (int const n)
  {
    if (n <= 1) {
      return 0;
    }

    if (n < Sz) {
      int const count = cache_[n];
      if (count) {
        return count;
      }
    }

    int const count
      = (n&1) 
      ? 1 + (*this)(n * 3 + 1)
      : 1 + (*this)(n >> 1);

    if (n < Sz) {
      cache_[n] = count;
    }

    return count;
  }

  std::vector<int> cache_;
};

int main()
{
  std::ios::sync_with_stdio(false);

  int n;
  std::cin >> n;
  std::vector<int> data(n);

  std::copy(
    std::istream_iterator<int>(std::cin),
    std::istream_iterator<int>(), data.begin());

  std::transform(data.begin(), data.end(), data.begin(), solver<>());

  std::copy(data.begin(), data.end(),
    std::ostream_iterator<int>(std::cout, "\n"));

  return 0;
}
