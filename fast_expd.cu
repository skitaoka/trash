//
// 高速な倍精度 exp の実装
//   cf. http://www.slideshare.net/herumi/exp-9499790
//
// 測定結果 (wall-clock time, N=1,000,000, GeForce GTX 680):
//   nvcc -O2 -m64 -arch=sm_30 -Xcompiler "/openmp /O2" fast_expd.cu
//   データがソート済みの場合 (テーブル参照でキャッシュにヒットしやすい)
//       exp   = 0.985646 ms
//     __expf  = 0.192752 ms
//      fexpdc = 0.332516 ms
//      fexpdg = 0.335972 ms
//       cpu   = 4.23478 ms
//   データがソートされてない場合
//       exp   = 0.708805 ms
//     __expf  = 0.205807 ms
//      fexpdc = 0.676168 ms
//      fexpdg = 0.491863 ms
//       cpu   = 4.2966 ms
//
// 測定結果 (wall-clock time, N=1,000,000, Tesla K20c):
//   nvcc -O2 -m64 -arch=sm_35 -Xcompiler "/openmp /O2" fast_expd.cu
//   データがソート済みの場合 (テーブル参照でキャッシュにヒットしやすい)
//       exp   = 0.118695 ms
//     __expf  = 0.094613 ms
//      fexpdc = 0.096657 ms
//      fexpdg = 0.113040 ms
//       cpu   = 5.612460 ms
//   データがソートされてない場合
//       exp   = 0.119024 ms
//     __expf  = 0.094577 ms
//      fexpdc = 0.562242 ms
//      fexpdg = 0.272528 ms
//       cpu   = 5.730750 ms
//
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <ctime>
#include <vector>
#include <omp.h>

union di
{
  unsigned long long i;
  double             d;
};

__device__ __constant__ unsigned long long c_tbl[2048];
__device__              unsigned long long g_tbl[2048];

__host__ __device__ double fexpd(double const x, unsigned long long const * __restrict__ const tbl)
{
  if (x <= -708.39641853226408) {
    return 0;
  }

	di di;

  if (x >= 709.78271289338397) {
    di.i = 0x7ff0000000000000ULL;
    return di.d;
  }

/*
  di.d = x * ((1UL<<11)/log(2.0)) + (3ULL<<51);
  unsigned long long const iax = tbl[di.i & ((1U<<11)-1)];
  double const t = (di.d - (3ULL<<51)) * (log(2.0)/(1UL<<11)) - x;
  unsigned long long const u = ((di.i + ((1UL<<(11+10)) - (1UL<<11))) >> 11) << 52;
  double const y = (3.0000000027955394 - t) * (t * t) * 0.16666666685227835064 - t + 1.0;
*/
  di.d = x * 2954.639443740597 + 6755399441055744ULL;
  unsigned long long const iax = tbl[di.i & 2047];
  double             const t   = (di.d - 6755399441055744ULL) * 0.0003384507717577858 - x;
  unsigned long long const u   = ((di.i + 2095104) >> 11) << 52;
  double             const y   = (3.0000000027955394 - t) * (t * t) * 0.16666666685227835064 - t + 1.0;

  di.i = u | iax;
  return y * di.d;
}

__global__ void
  exp1_kernel(double * const y, double const * const x, int const n)
{
  int const i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    y[i] = exp(x[i]);
  }
}

__global__ void
  exp2_kernel(double * const y, double const * const x, int const n)
{
  int const i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    y[i] = __expf(x[i]);
  }
}

__global__ void
  exp3_kernel(double * const y, double const * const x, int const n)
{
  int const i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    y[i] = fexpd(x[i], c_tbl);
  }
}

__global__ void
  exp4_kernel(double * const y, double const * const x, int const n)
{
  int const i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    y[i] = fexpd(x[i], g_tbl);
  }
}

void _check(cudaError_t const e)
{
  if (e) {
    std::fprintf(stderr, "%s\n", ::cudaGetErrorString(e));
    std::exit(1);
  }
}


#define N 10000000
#define M 100

int main()
{
  _check(::cudaSetDevice(0));

  {
    unsigned long long h_tbl[2048];
    for (int i = 0; i < 2048; ++i) {
      di v; v.d = std::pow(2.0, i * (1.0/2048));
      h_tbl[i] = v.i & ((1ULL<<52)-1);
    }
    ::cudaMemcpyToSymbol(c_tbl, h_tbl, 2048 * sizeof(unsigned long long));
    ::cudaMemcpyToSymbol(g_tbl, h_tbl, 2048 * sizeof(unsigned long long));
    ::cudaDeviceSynchronize();
  }

  std::vector<double> h_x(N);
  for (int i = 0; i < N; ++i) {
    h_x[i] = (rand() * (2.0/double(RAND_MAX)) - 1.0);
  }
  //std::sort(h_x.begin(), h_x.end());

  double * g_x; ::cudaMalloc(&g_x, N * sizeof(double));
  double * g_y; ::cudaMalloc(&g_y, N * sizeof(double));

  ::cudaMemcpy(g_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice);

  float elapsed_time_ms1;
  float elapsed_time_ms2;
  float elapsed_time_ms3;
  float elapsed_time_ms4;
  float elapsed_time_ms5;

  cudaEvent_t start; ::cudaEventCreate(&start);
  cudaEvent_t stop ; ::cudaEventCreate(&stop );

  std::vector<double> h_1(M);
  {
    double const start_time = ::omp_get_wtime();
    exp1_kernel<<<(N+1023)/1024, 1024>>>(g_y, g_x, N);
    ::cudaDeviceSynchronize();
    elapsed_time_ms1 = ::omp_get_wtime() - start_time;
    ::cudaMemcpy(h_1.data(), g_y, M * sizeof(double), cudaMemcpyDeviceToHost);
  }

  std::vector<double> h_2(M);
  {
    double const start_time = ::omp_get_wtime();
    exp2_kernel<<<(N+1023)/1024, 1024>>>(g_y, g_x, N);
    ::cudaDeviceSynchronize();
    elapsed_time_ms2 = ::omp_get_wtime() - start_time;
    ::cudaMemcpy(h_2.data(), g_y, M * sizeof(double), cudaMemcpyDeviceToHost);
  }

  std::vector<double> h_3(M);
  {
    double const start_time = ::omp_get_wtime();
    exp3_kernel<<<(N+1023)/1024, 1024>>>(g_y, g_x, N);
    ::cudaDeviceSynchronize();
    elapsed_time_ms3 = ::omp_get_wtime() - start_time;
    ::cudaMemcpy(h_3.data(), g_y, M * sizeof(double), cudaMemcpyDeviceToHost);
  }

  std::vector<double> h_4(M);
  {
    double const start_time = ::omp_get_wtime();
    exp4_kernel<<<(N+1023)/1024, 1024>>>(g_y, g_x, N);
    ::cudaDeviceSynchronize();
    elapsed_time_ms4 = ::omp_get_wtime() - start_time;
    ::cudaMemcpy(h_4.data(), g_y, M * sizeof(double), cudaMemcpyDeviceToHost);
  }

  std::vector<double> h_y(N);
  {
    double const start_time = ::omp_get_wtime();

#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
      h_y[i] = std::exp(h_x[i]);
    }

    elapsed_time_ms5 = ::omp_get_wtime() - start_time;
  }

  for (int i = 0; i < M; ++i) {
    double const y = h_y[i];
    std::printf("% .6lf:  |%g|  |%g|  |%g|  |%g|\n",
      h_x[i],
      std::abs((h_1[i] - y) / y),
      std::abs((h_2[i] - y) / y),
      std::abs((h_3[i] - y) / y),
      std::abs((h_4[i] - y) / y));
  }

  std::printf("  exp   = %g ms\n", elapsed_time_ms1 * 1000);
  std::printf("__expf  = %g ms\n", elapsed_time_ms2 * 1000);
  std::printf(" fexpdc = %g ms\n", elapsed_time_ms3 * 1000);
  std::printf(" fexpdg = %g ms\n", elapsed_time_ms4 * 1000);
  std::printf("  cpu   = %g ms\n", elapsed_time_ms5 * 1000);

  return 0;
}
