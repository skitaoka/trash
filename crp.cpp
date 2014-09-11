//
// 中華料理店過程 (Chinese Restaurant Process) からサンプリングする
// cf. http://www.singularpoint.org/blog/math/stat/crp-generation/
//
#include <algorithm>
#include <iterator>
#include <random>
#include <sstream>
#include <iostream>

namespace reactor
{
  class string_to
  {
  public:
    inline string_to(char const * const s)
      : s_(s)
    {
    }
  
  public:
#define REACTOR_STRINGTO_IMPL(T)\
    inline operator T() const\
    {\
      std::istringstream in(s_);\
      T retval;\
      in >> retval;\
      return retval;\
    }
    REACTOR_STRINGTO_IMPL(int);
    REACTOR_STRINGTO_IMPL(long);
    REACTOR_STRINGTO_IMPL(long long);
    REACTOR_STRINGTO_IMPL(unsigned);
    REACTOR_STRINGTO_IMPL(unsigned long);
    REACTOR_STRINGTO_IMPL(unsigned long long);
    REACTOR_STRINGTO_IMPL(float);
    REACTOR_STRINGTO_IMPL(double);
    REACTOR_STRINGTO_IMPL(long double);
#undef REACTOR_STRINGTO_IMPL
    
  private:
    char const * const s_;
  };

  reactor::string_to cast(char const * const s)
  {
    return reactor::string_to(s);
  }
}

int main(int argc, char* argv[])
{
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " sample-size alpha\n";
    return 1;
  }

  std::size_t const sample_size = reactor::cast(argv[1]);
  double const alpha = reactor::cast(argv[2]);

  std::mt19937_64 engine(std::random_device("")());
  std::normal_distribution<> dnorm; // G0

  std::vector<double> samples;
  samples.reserve(sample_size);

  // 初期値を標準正規分布からサンプリング
  // NOTE: ここでは標準正規分布を G0 としている。
  samples.push_back(dnorm(engine));
  for (std::size_t i = 1; i < sample_size; ++i) {
    samples.push_back(
      // α/(α+i) の確率で標準正規分布からサンプリング
      // それ以外はすでに得られたサンプルから一様サンプリング
      std::bernoulli_distribution(alpha / (alpha + i))(engine)
      ? dnorm(engine)
      : samples[std::uniform_int_distribution<std::size_t>(0, i-1)(engine)]);
  }

  std::copy(std::begin(samples), std::end(samples),
    std::ostream_iterator<double>(std::cout, "\n"));

  return 0;
}
