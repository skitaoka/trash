//
// 集合内のユニークな要素の数を推定する乱択アルゴリズム　ＨｙｐｅｒLogLog (k=1 の場合の実装)
//   ※ k は使用するハッシュ関数の数
//
// 1-HyperLogLog:
//   各要素のハッシュ値を計算する。
//   ハッシュ値の Base-2 Rank を計算する。
//     (Base-2 Rank は、ハッシュ値の先頭に並んだ 0 ビットの数)
//   集合内で最大の　Base-2 Rank を p とする。
//   2^p をユニークな要素数の推定値とする。
//   # 2^max{__clz(hash(x[i]))}_{i=1}^{N}
//
// k-HyperLogLog (k = 2^b): 
//   各要素のハッシュ値を計算する。
//   ハッシュ値の先頭 b bit が同じ要素の集合それぞれについて Base-2 Rank を計算する。
//      ※先頭の b bit は同じ値なので、その部分を除いたビットで計算する。
//   それぞれの集合内で最大の Base-2 Rank を M[i] (i in [0,k-1]) とする。
//   2^M[i] の　nbchm (normalized bias corrected harmonic mean) をとって推定値とする。
//   # nbchm([e | e <- 2^max{R[j]}]),
//   #   where R[j] = [r | r <- __clz(h[i] & msk1) - b, j = (h[i] >> (B-b)) & msk2],
//   #         h[i] = hash(x[i]),
//   #         msk1 = (1 << (B-b)) - 1.
//   #         msk2 = (1 <<    b ) - 1,
//   #         B is the number of bits (B=32/64 if data type of hash is int32/int64).
//
// build: nvcc -arch=sm_30 -O2 -Xcompiler "/wd4819 /openmp" HyperLogLog.cu
//
#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>

#include <intrin.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/iterator/counting_iterator.h>

#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <thrust/random.h>

// 真値の計算用
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <omp.h>

namespace reactor
{
  __device__ __host__
    int hash(int const x)
  {
#if 0
    int retval = 0;
    for (int i = 0; i < 32; ++i) {
      int const b = (x>>i)&1;
      retval = 31 * retval + b;
    }
    return retval;
#else
    return x;
#endif
  }

  struct cpu { static char const * name() { return "cpu"; } };
  struct gpu { static char const * name() { return "gpu"; } };

  template <typename Container>
  struct arch;

  template <typename T>
  struct arch<thrust::device_vector<T>>
  {
    typedef reactor::gpu type;
  };

  template <typename T>
  struct arch<thrust::host_vector<T>>
  {
    typedef reactor::cpu type;
  };

  template <typename T>
  struct clz;

  template <>
  struct clz<gpu>: public thrust::unary_function<int,int>
  {
    __device__
      int operator () (int const x) const
    {
      return __clz(reactor::hash(x));
    }
  };

  template <>
  struct clz<cpu>: public thrust::unary_function<int,int>
  {
    __host__
      int operator () (int const x) const
    {
      return 31-__lzcnt(reactor::hash(x));
    }
  };

  struct runif: public thrust::unary_function<std::size_t, int>
  {
    __device__ __host__
      int operator () (std::size_t const n) const
      {
        thrust::default_random_engine engine;
        engine.discard(n);
        return thrust::uniform_int_distribution<>()(engine);
      }
  };

  struct perf_counter
  {
  public:
    inline perf_counter()
      : start(::omp_get_wtime())
    {
    }

    inline double get() const
    {
      return ::omp_get_wtime() - start;
    }

  private:
    double const start;
  };
}

template <typename Container>
void brute_force(Container data)
{
  reactor::perf_counter time;
  thrust::sort(data.begin(), data.end());
  auto const it = thrust::unique(data.begin(), data.end());
  int const expect = thrust::distance(data.begin(), it);
  std::cout << "expect = " << expect << " (" << reactor::arch<Container>::type::name() << ") ... " << time.get() << " sec.\n";
}

template <typename Container>
void hyper_log_log(Container data)
{
  typedef typename reactor::arch<Container>::type arch_type;

  reactor::perf_counter time;

  // ハッシュ値を計算して、上位ビットの 0 の数を数える
  thrust::transform(data.begin(), data.end(), data.begin(), reactor::clz<arch_type>());

  // 集合内で最大の 0 の数　p を得る
  int const p = thrust::reduce(data.begin(), data.end(), 0, thrust::maximum<int>());
  int const actual = 1<<p;

  std::cout << "actual = " << actual << " (" << arch_type::name() << ") ... " << time.get() << " sec.\n";
}

int main(int argc, char * argv[])
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " size\n";
    return 1;
  }


  int size;
  {
    std::istringstream in(argv[1]);
    in >> size;
  }

  // データを生成
  thrust::host_vector<int> h_data(size);
#if 0
  {
    std::mt19937_64 engine(std::random_device("")());
    thrust::generate(h_data.begin(), h_data.end(), [&engine]()->int
    {
      return std::uniform_int_distribution<>()(engine);
    });
  }
#else
  thrust::transform(
    thrust::make_counting_iterator(std::size_t(0)),
    thrust::make_counting_iterator(h_data.size()),
    h_data.begin(), reactor::runif());
#endif
  thrust::copy_n(h_data.begin(),
    std::min<std::size_t>(10, h_data.size()),
    std::ostream_iterator<int>(std::cout, "\n"));

  // GPU に転送
  thrust::device_vector<int> g_data = h_data;

  // まずは、比較用に真値を計算する
  brute_force(h_data);
  brute_force(g_data);

  // HyperLogLog で計算する
  hyper_log_log(std::move(h_data));
  hyper_log_log(std::move(g_data));

  return 0;
}

/*
cf. http://www.slideshare.net/iwiwi/minhash

Minhash は Jaccard 係数を近似する確率的手法。
　　Jaccard 係数: J(S1, S2) = |S1 cap S2| / |S1 cup S2|
  　　集合の類似度を測る指標として使われる。

Minhash algorithm
  I1 = {h(a) | a in S1} where h(a) 要素 a のハッシュ値を計算するハッシュ関数
  I2 = {h(a) | a in S2}
  「I1 の要素の最小値 == I2 の要素の最小値」となる確率は Jaccard 係数に等しい

k-Minhash algorithm
  K 個のハッシュ関数を用意して、
  　　J(S1, S2) = [ハッシュ値の最小値が一致した数]/K
  として Jaccard 係数を近似する。
  推定値は unbiased で 分散は J(S1, S2) (1 - J(S1, S2)) / K になる。

※ハッシュ値によるテクニックは、集合をランダムシャッフルして最初の要素が一致するか確認しているのと一緒。
  ランダムな要素の最初の要素が一致する確率は P = |S1 cap S2| / |S1 cup S2|
  これって Jaccard 係数と一緒じゃん！

HyperLogLog は集合のユニーク要素数を近似する確率的手法
  std::sort(std::begin(s), std::end(s));
  std::distance(std::begin(s), std::unique(std::begin(s), std::end(s)));

HyperLogLog algorithm
  I = {h(a) | a in S}
  R = {base2_rank(i) | i in I} where base2_rank(i) は i の先頭ビットに 0 が並んでいる数をかえす。
  p = max{R}
  推定値 = 2^p where 0 の並びの個数が p 以上になるのは確率 1/2^p だから

k-HyperLogLog algorithm
  データを K 分割して、それぞれのデータについて HyperLogLog を計算する。
  K 個の推定値の normalized bias corrected harmonic mean をとって最終的な推定値とする。
*/
