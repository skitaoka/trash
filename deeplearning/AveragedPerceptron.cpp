#include "OnlineLearning.hpp"

int main(int argc, char* argv[])
{
  std::size_t const epoch = 1;
  std::size_t const num_samples = 100;

  // 教師データ
  std::vector<std::vector<double>> x(num_samples);
  std::vector<int> t(num_samples); // [-1, 1]

  double const alpha = 0.1; // ステップ幅
  std::vector<double> w;
  std::vector<double> w_avg;
  for (std::size_t n = 0; n < epoch; ++n) {
    for (std::size_t i = 0, size = x.size(); i < size; ++i) {
      double const m = t[i] * aka::inner_product(w, x[i]); // margin
      if (m > 1.0) {
        continue;
      }
      {
        double const s = alpha * t[i];
        aka::transform(w, x[i],
          [s](double const w, double const x)
        {
          return w + s * x;
        });
      }
      {
        size_t const c = n * size + i;
        double const s = 1.0 / (1 + c);
        aka::transform(w_avg, w,
          [c,s](double const w, double const w_avg)
        {
          return (w + w_avg * c) * s;
        });
      }
    }
  }

  return 0;
}

