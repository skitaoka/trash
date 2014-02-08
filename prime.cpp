#include <vector>
#include <cstdio>

int main(int argc, char * argv[])
{
  // エラトステネスのふるいで 2 より大きい素数を求める。
  std::vector<bool> table(10000000U, true);
  table[0] = false;
  table[1] = false;
  for (std::size_t i = 2, size = table.size(); i < size; ++i) {
    if (table[i]) {
      for (std::size_t j = i + i; j < size; j += i) {
        table[j] = false;
      }
    }
  }

  // 出力
  for (std::size_t i = 2, size = table.size(), nl = 0; i < size; ++i) {
    if (table[i]) {
      if (++nl > 10U) {
        nl = 1U;
        std::printf("\n");
      }
      std::printf(" %10d", i);
    }
  }

  return 0;
}
