//
// nvcc -O2 -arch=sm_30 -Xcompiler "/openmp /wd4819" gaussian_elimination.cu
//
#include <random>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>

namespace {

  __host__ __device__ __forceinline __forceinline__
    double normalize_number(double const x)
  {
    return (abs(x) > DBL_EPSILON) ? x : 0.0;
  }

  __host__ __device__ __forceinline __forceinline__
    int idx(int const i, int const j, int const n)
  {
    return j*n+i;
  }

  __host__ void __forceinline __forceinline__
    _check(cudaError_t const e)
  {
    if (e) {
      std::fprintf(stderr, "%s\n", ::cudaGetErrorString(e));
      std::exit(1);
    }
  }

  // Mathematica フォーマットで解ベクトルを表示する.
  void show_source_vector(std::vector<double> const & a, int const n)
  {
    std::cout << "{";
    for (int i = 0; i < n; ++i) {
      if (i) {
        std::cout << ",";
      }
      std::cout << "{" << a[idx(i,n,n)] << "}";
    }
    std::cout << "}\n";
  }

  // Mathematica フォーマットで線形方程式を表示する.
  void show_linear_system(std::vector<double> const & a, int const n)
  {
    std::cout << "Inverse[{";
    for (int i = 0; i < n; ++i) {
      if (i) {
        std::cout << ",";
      }
      std::cout << "{";
      for (int j = 0; j < n; ++j) {
        if (j) {
          std::cout << ",";
        }
        std::cout << a[idx(i,j,n)];
      }
      std::cout << "}";
    }
    std::cout << "}].";
    show_source_vector(a, n);
  }

  // 整形して線形方程式を表示する.
  void display_linear_system(std::vector<double> const & a, int const n)
  {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        std::cout << std::setw(10) << a[idx(i,j,n)];
      }
      std::cout << " | " << std::setw(10) << a[idx(i,n,n)] << '\n';
    }
    std::cout << '\n';
  }
}

// 拡張ガウス消去法で対角化しつつ解を計算していく.
__global__ void
  kernel_do_gaussian_elimination(
    double * const a,
    int      const n,
    int      const i)
{
  int const j = threadIdx.x + blockIdx.x * blockDim.x;
  if ((j < n) && (j != i)) {
    double const _aji = -a[idx(j,i,n)] / a[idx(i,i,n)];
    for (int k = i+1; k <= n; ++k) { // NOTE: k = i は計算不用なので省略している.
      int const jk = idx(j,k,n);
      a[jk] = normalize_number(a[jk] + a[idx(i,k,n)] * _aji);
    }
  }
}

// 行列の対角要素でベクトルをスケーリングして解を求める.
__global__ void
  kernel_diagonal_scaling(
    double * const a,
    int      const n)
{
  int const i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    a[idx(i,n,n)] /= a[idx(i,i,n)];
  }
}

// カーネルを起動する (Dynamic Parallelism を利用するともっと簡単に書ける)
__host__ void
  lunch_do_gaussian_elimination(double * const a, int const n)
{
  cudaStream_t stream;
  _check(::cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  int const num_threads = 1024;
  int const num_blocks  = (n+num_threads-1)/num_threads;

  for (int i = 0; i < n; ++i) {
    kernel_do_gaussian_elimination
      <<<num_blocks, num_threads, 0, stream>>>
      (a, n, i);
  }
  kernel_diagonal_scaling
    <<<num_blocks, num_threads, 0, stream>>>
    (a, n);

  _check(::cudaStreamSynchronize(stream));
  _check(::cudaStreamDestroy(stream));
}

// 拡張ガウス消去法の CPU 実装 (デバッグ用)
void do_gaussian_elimination(std::vector<double> & a, int const n)
{
  for (int i = 0; i < n; ++i) {
    //display_linear_system(a, n);

    double const _aii = -1.0 / a[idx(i,i,n)];
#ifndef _OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < n; ++j) {
      if (j != i) {
        double const _aji = a[idx(j,i,n)] * _aii;
        for (int k = i+1; k <= n; ++k) {
          int const jk = idx(j,k,n);
          a[jk] = normalize_number(a[jk] + a[idx(i,k,n)] * _aji);
        }
      }
    }
  }
#ifndef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n; ++i) {
    a[idx(i,n,n)] /= a[idx(i,i,n)];
  }

  //display_linear_system(a, n);
}

int main(int argc, char * argv[])
{
  std::ios::sync_with_stdio(false);

  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " device-id(cpu:-1) dim-of-linear-system\n";
    return 1;
  }

  int const d = std::atoi(argv[1]);
  int const n = std::atoi(argv[2]);

  // 線形方程式をランダムに作る.
  std::cout << "> init data.\n";
  std::vector<double> hab(n * (n+1));
  {
    std::random_device rd;
  	std::mt19937 mt(rd());
  	std::uniform_real_distribution<double> u;

    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int j = 0; j <= n; ++j) {
        double const aij = u(mt);
        hab[idx(i,j,n)] = aij;
        sum += aij;
      }
      // テスト目的なので行列を対角優位にする.
      hab[idx(i,i,n)] = -sum;
    }
  }
  //show_linear_system(hab, n);

  if (d >= 0) {
    std::cout << "> init a device.\n";
    _check(::cudaSetDevice(d));

    std::cout << "> init device data.\n";
    double * dab = NULL;
    std::size_t const size = hab.size() * sizeof(double);
    _check(::cudaMalloc(reinterpret_cast<void**>(&dab), size));
    _check(::cudaMemcpyAsync(dab, hab.data(), size, cudaMemcpyHostToDevice));
    std::cout << "> memory usage: " << (10*size/(1024*1024))/10.0 << " MiB\n";

    std::cout << "> solve.\n";

    cudaEvent_t begin; _check(::cudaEventCreate(&begin));
    cudaEvent_t end  ; _check(::cudaEventCreate(&end  ));

    _check(::cudaEventRecord(begin, 0));
    {
      lunch_do_gaussian_elimination(dab, n);
    }
    _check(::cudaEventRecord(end, 0));
    _check(::cudaEventSynchronize(end));

    float elapsed_time_ms = 0.0f;
    _check(::cudaEventElapsedTime(&elapsed_time_ms, begin, end));
    std::cout << "> done. " << elapsed_time_ms << " ms \n";

    std::cout << "> memcpy.\n";
    _check(::cudaMemcpy(hab.data(), dab, size, cudaMemcpyDeviceToHost));
    _check(::cudaFree(dab));
  } else {
    std::cout << "> solve.\n";

    auto const begin = std::chrono::steady_clock::now();
    {
      do_gaussian_elimination(hab, n);
    }
    auto const end = std::chrono::steady_clock::now();
    auto const elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "> done. " << elapsed_time_ms << " ms \n";
  }
  //show_source_vector(hab, n);

  return 0;
}
