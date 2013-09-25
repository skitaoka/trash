//
// 有意に勝率が高くなったと判定できる勝利回数を試行回数ごとに計算する。
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// 累乗
double power(double const x, int const n)
{
  double retval = 1.0;
  for (int i = 0; i < n; ++i) {
    retval *= x;
  }
  return retval;
}

// 階乗
double fact(int const n)
{
  double retval = 1.0;
  for (int i = 2; i <= n; ++i) {
    retval *= i;
  }
  return retval;
}

// 累乗/階乗
double power_by_fact(double const x, int const n)
{
  if (n) {
    double retval = 1.0;
    for (int i = 1; i <= n; ++i) {
      retval *= x / i;
    }
    return retval;
  } else {
    return 1.0;
  }
}

// 組合せ
double combi(int const n, int const k)
{
  double retval = 1.0;
  for (int i = n - k + 1; i <= n; ++i) {
    retval *= i;
  }
  return retval / fact(k);
}

// p=1/2 の二項分布の累積関数
double binormal_cfd(int const n, int const k)
{
  double retval = 0.0;
  for (int i = k; i <= n; ++i) {
    retval += combi(n, i) * power(0.5, i);
  }
  return retval;
}

// p=1/2 のポアソン分布の累積関数
double poisson_cfd(int const n, int const k)
{
  double const lambda = n * 0.5;

  double retval = 0.0;
  for (int i = k; i <= n; ++i) {
    double a = 1.0;
    retval += power_by_fact(lambda, i);
  }
  return std::exp(-lambda) * retval;
}

int main(int argc, char * argv[])
{
  // 試行回数
  int const count[] = {1, 5, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,};
  int const size = static_cast<int>(sizeof(count) / sizeof(count[0]));

  double const p[] = {0.10, 0.05, 0.01,};
  int const num_pattern = static_cast<int>(sizeof(p) / sizeof(p[0]));

  std::printf("試行回数");
  for (int j = 0; j < num_pattern; ++j) {
    std::printf(", %g", p[j]);
  }
  std::printf("\n");
  for (int i = 0; i < size; ++i) {
    // 有意に勝率が高くなったと判定できる勝利回数
    std::vector<int> nwon(num_pattern);

    int const n = count[i]; // 試行回数
    for (int k = n/2; k <= n; ++k) {
#if 0
	  // 二項分布で計算
      double const q = binormal_cfd(n, k);
#else
	  // ポアソン分布で計算
	  double const q = poisson_cfd(n, k);
#endif
      for (int j = 0; j < num_pattern; ++j) {
        if (!nwon[j] && (q < p[j])) {
          nwon[j] = k;
        }
      }
      if (nwon.back()) {
        break;
      }
    }

    std::printf("%d", n);
    for (int j = 0; j < num_pattern; ++j) {
      std::printf(", %d", nwon[j]);
    }
    std::printf("\n");
    std::fflush(stdout);
  }

  return 0;
}
