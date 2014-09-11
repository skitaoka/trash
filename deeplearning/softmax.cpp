//
// softmax 関数による線形多クラス分類器を学習するアルゴリズムのサンプル実装
//
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

namespace
{
  std::size_t const M = 12; // 教師データ数
  std::size_t const N =  4; // 特徴ベクトルの次元
  std::size_t const K =  4; // 出力ベクトルの次元
  
  double const tolerance = 1e-5;

  template <std::size_t N, std::size_t K, typename T>
  std::vector<T> calc(T const x[N], T const w[K][N+1])
  {
    std::vector<T> z(K);
    for (std::size_t k = 0; k < K; ++k) {
      z[k] = std::inner_product(x, x+N, w[k], w[k][N]);
    }

    T const a = *std::max_element(std::begin(z), std::end(z));
    std::transform(std::begin(z), std::end(z), std::begin(z),
      [a](T const x) { return std::exp(x - a); });

    double const sum = std::accumulate(std::begin(z), std::end(z), 0.0);
    std::transform(std::begin(z), std::end(z), std::begin(z),
      std::bind1st(std::multiplies<double>(), (sum > 0.0) ? 1.0 / sum : 0.0));

    return z;
  }

  // cross entropy
  template <std::size_t M, std::size_t N, std::size_t K, typename T>
  T cross_entropy(T const x[M][N], T const y[M][K], T const w[K][N+1])
  {
    T retval = 0.0;
    for (std::size_t i = 0; i < M; ++i) {
      std::vector<T> const z = ::calc<N, K>(x[i], w);
      for (std::size_t k = 0; k < K; ++k) {
        retval += y[i][k] * std::log(z[k]);
      }
    }
    return -retval / M;
  }
}

int main()
{
  double const x[M][N] = {
    {0, 1, 1, 1},
    {1, 0, 1, 1},
    {1, 1, 0, 1},
    {1, 1, 1, 0},
    {0, 1, 2, 1},
    {-1, 0, 1, 1},
    {1, 1, 0, 2},
    {1, -2, 1, 0},
    {0, 2, 1, 1},
    {1, 0, 1, 2},
    {1, 2, 0, 1},
    {1, 1, 2, 0},
  };
  double const t[M][K] = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1},
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1},
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1},
  };

  // 重みベクトル
  double w[K][N+1];

  for (std::size_t epoch = 1; epoch <= 100000; ++epoch) {
    for (std::size_t i = 0; i < M; ++i) {
      std::vector<double> const z = ::calc<N, K>(x[i], w);

      // 勾配を更新
      for (std::size_t k = 0; k < K; ++k) {
        double const s = (t[i][k] - z[k]);
        for (std::size_t n = 0; n < N; ++n) {
          w[k][n] += s * x[i][n];
        }{
          w[k][N] += s;
        }
      }

#if 0
      for (std::size_t k = 0; k < K; ++k) {
        std::cout << '\t' << w[k][N];
      }
      std::cout << '\n';
#endif
    }

    // オフセット項をスケーリング
    double maxb = -std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < K; ++k) {
      if (maxb < w[k][N]) {
        maxb = w[k][N];
      }
    }
    for (std::size_t k = 0; k < K; ++k) {
      w[k][N] -= maxb;
    }

    if (epoch % 100 == 0) {
      double const entropy = ::cross_entropy<M, N, K>(x, t, w);
      if (entropy < tolerance) {
        std::cout << '[' << epoch << "]: " << entropy << '\n';
        break;
      }
    }
  }


  return 0;
}
