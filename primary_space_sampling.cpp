#include <cmath>
#include <ctime>

#include <iterator>
#include <algorithm>
#include <numeric>
#include <random>

#include <iostream>

int main()
{
  std::ios::sync_with_stdio(false);

  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  std::mt19937 engine(::time(nullptr));

  std::vector<int> histgram(100);
  for (int i = 0; i < 10000; ++i) {
  	// 1,000 次元ベクトルを生成して原点からの距離を測る
    double distance = 0.0;
    for (int k = 0; k < 1000; ++k) {
      double const x = distribution(engine);
      distance = x * x;
    }
    distance = std::sqrt(distance);

    // 原点からの距離のヒストグラムをつくる
    ++histgram[static_cast<int>(distance * histgram.size())];
  }

  std::copy(histgram.begin(), histgram.end(),
    std::ostream_iterator<int>(std::cout, "\n"));

  std::cout << "sum: " << std::accumulate(histgram.begin(), histgram.end(), 0) << '\n';

  return 0;
}
