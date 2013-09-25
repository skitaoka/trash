#include <cassert>
#include <iostream>
#include <vector>
#include <iterator>

namespace aux
{
  template <typename Iterator>
  typename Iterator::value_type
    find_median(Iterator begin, Iterator end)
  {
    typename Iterator::distance_type const
      size = std::distance(begin, end);
    assert(size);

    bool const isEven = (size & 1) == 0;
    typename Iterator const median = begin + size / 2;

    while (begin < end) {
      Iterator::value_type const v = *begin;
      Iterator i = begin - 1;
      Iterator j = end;
      for (;;) {
        do { ++i; } while ((*i < v));
        do { --j; } while ((*j > v) && (j > begin));

        if (i < j) {
          std::iter_swap(i, j);
        } else {
          break;
        }
      }
      if (i <= median) {
        begin = i + 1;
      }
      if (median <= i) {
        end   = i - 1;
      }
    }

    if (isEven) {
      return (*(median-1) + *median) / 2;
    } else {
      return *median;
    }
  }
}

int main()
{
  std::vector<int> v;
  v.push_back(170);
  v.push_back(100);
  v.push_back(110);
  v.push_back(120);
  v.push_back(180);
  v.push_back(150);
  v.push_back(140);
  v.push_back(130);
  v.push_back(160);
  v.push_back(190);

  std::cout << aux::find_median(v.begin(), v.end()) << std::endl;

  return 0;
}

