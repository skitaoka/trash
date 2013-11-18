//
// 高速な exp の実装 (標準の exp に対して相対誤差で 2 桁ほどの精度)
//   cf. http://www.radiumsoftware.com/0303.html#030301
//
#include <cmath>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <random>

namespace sota
{
  namespace math
  {
    namespace details
    {
      template <typename T>
      struct exp_traits;

      template <>
      struct exp_traits<float>
      {
        typedef int type;
        inline static float a() { return (1<<23)/std::log(2.0f); }
        inline static float b() { return 0x3F800000; }
      };

      template <>
      struct exp_traits<double>
      {
        typedef long long int type;
        inline static double a() { return (1LL<<52) / std::log(2.0); }
        inline static double b() { return 0x3FF0000000000000LL; }
      };
    }

    template <typename T>
    T exp(T const x)
    {
      typename details::exp_traits<T>::type const
        e = static_cast<typename details::exp_traits<T>::type>(
          details::exp_traits<T>::a() * x + details::exp_traits<T>::b());
      return (T&)e;
    }
  }
}

int main()
{
  std::ios::sync_with_stdio(false);

  std::cout << sota::math::exp( 0.0f) << std::endl;
  std::cout << sota::math::exp(-1.0f) << std::endl;
  std::cout << sota::math::exp( 1.0f) << std::endl;

  std::cout << sota::math::exp( 0.0) << std::endl;
  std::cout << sota::math::exp(-1.0) << std::endl;
  std::cout << sota::math::exp( 1.0) << std::endl;

  std::random_device rnd;
  std::vector<int> seeds(10);
  std::generate(seeds.begin(), seeds.end(), std::ref(rnd));

  std::mt19937 engine(std::seed_seq(seeds.begin(), seeds.end()));
  std::uniform_real_distribution<double> distribution(-10.0, 10.0);

  std::cout << std::scientific;
  for (int i = 0; i < 10; ++i) {
    double const x = distribution(engine);
    double const t = std::exp(x);
    double const a = sota::math::exp(x);
    double const b = sota::math::exp(static_cast<float>(x));

    std::cout << x << " : " << t << ",\n\t"
      << a << " (" << std::abs((a-t)/t) << "),\n\t"
      << b << " (" << std::abs((b-t)/t) << ")\n";
  }

  return 0;
}
