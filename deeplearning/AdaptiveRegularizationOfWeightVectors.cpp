#include <algorithm>
#include <vector>
#include <numeric>

namespace aka
{
  template <typename T>
  inline T inner_product(std::vector<T> const & a, std::vector<T> const & b)
  {
    return std::inner_product(a.begin(), a.end(), b.begin(), T());
  }

  template <typename T>
  inline T quadratic_form(std::vector<T> const & a, std::vector<T> const & x)
  {
    return std::inner_product(a.begin(), a.end(), x.begin(), T(), std::plus<T>(),
      [](T const a, T const x) { return a * x * x; });
  }

  template <typename T, typename Fn>
  inline void transform(std::vector<T> & a, std::vector<T> const & b, Fn fn)
  {
    std::transform(a.begin(), a.end(), b.begin(), fn);
  }

  template <typename T, typename Fn>
  inline void transform(std::vector<T> & a, std::vector<T> const & b, std::vector<T> const & c, Fn fn)
  {
    for (std::size_t i = 0, size = a.size(); i < size; ++i) {
      a[i] = fn(a[i], b[i], c[i]);
    }
  }
}

int main(int argc, char* argv[])
{
  std::size_t const epoch = 1;
  std::size_t const num_samples = 100;

  // 教師データ
  std::vector<std::vector<double>> x(num_samples);
  std::vector<int> t(num_samples); // [-1, 1]

  double const r = 1.0; // パラメータ
  std::vector<double> u; // 平均
  std::vector<double> v; // 分散
  for (std::size_t n = 0; n < epoch; ++n) {
    for (std::size_t i = 0, size = x.size(); i < size; ++i) {
      double const m = t[i] * aka::inner_product(u, x[i]); // margin
      if (m > 1.0) {
        continue;
      }

      double const c = aka::quadratic_form(v, x[i]); // confidnece
      double const beta = 1 / (c + r);
      double const alpha = t[i] * (1 - m) * beta;

      // update mean
      //   mu <- mu + alpha sigma t x
      //     where alpha = max(0, 1 - t x^T mu) beta,
      //           beta = 1 / (x^T sigma x + r).
      aka::transform(u, v, x[i],
        [alpha](double const u, double const v, double const x)
      {
        double const r = v * x;
        return u + alpha * r;
      });

      // update variance:
      //   sigma <- sigma - beta sigma x x^T sigma
      aka::transform(v, x[i],
        [beta](double const v, double const x)
      {
        double const r = v * x;
        return v - beta * r * r;
      });
    }
  }

  return 0;
}
