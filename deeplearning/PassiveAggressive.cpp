#include "OnlineLearning.hpp"

int main(int argc, char* argv[])
{
  std::size_t const epoch = 1;
  std::size_t const num_samples = 100;

  // 教師データ
  std::vector<std::vector<double>> x(num_samples);
  std::vector<int> t(num_samples); // [-1, 1]

  double const C = 1.0; // パラメータ
  std::vector<double> w;
  for (std::size_t n = 0; n < epoch; ++n) {
    for (std::size_t i = 0, size = x.size(); i < size; ++i) {
      double const m = t[i] * aka::inner_product(w, x[i]);
      if (m > 1.0) {
        continue;
      }
      {
        double const a = aka::inner_product(x[i], x[i]);
        double const alpha = (1 - m) / (a + 1 / C);
        double const s = alpha * t[i];
        aka::transform(w, x[i],
          [s](double const w, double const x)
        {
          return w + s * x;
        });
      }
    }
  }

  return 0;
}

