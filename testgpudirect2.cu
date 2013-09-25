// Peer 間のメモリ転送速度を測る。
//
// nvcc -O2 -arch=sm_35 –Xcompiler "–fopenmp" testgpudirect2.cu
//
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <omp.h>

void _check(cudaError_t const e)
{
  if (e) {
    std::fprintf(stderr, "%s\n", ::cudaGetErrorString(e));
    std::exit(1);
  }
}

int main(int argc, char * argv[])
{
  if (argc < 4) {
    std::fprintf(stderr, "Usage: %s gpu_from gpu_to bytes\n", argv[0]);
    return 1;
  }

  int const gpu_from = std::atoi(argv[1]);
  int const gpu_to   = std::atoi(argv[2]);
  int const size     = std::atoi(argv[3]);

  int canAccessPeer;
  _check(::cudaDeviceCanAccessPeer(&canAccessPeer, gpu_to, gpu_from));
  if (!canAccessPeer) {
    std::fprintf(stderr, "(%d-%d) cannot access peer.\n", gpu_from, gpu_to);
    return 1;
  }
  //_check(::cudaDeviceEnablePeerAccess(gpu_to, gpu_from));

  char * data0 = NULL; // GPU 0 上のメモリ
  char * data1 = NULL; // GPU 1 上のメモリ
  char * data2 = NULL; // CPU 上のメモリ

  _check(::cudaSetDevice(gpu_from));
  _check(::cudaMalloc(&data0, size));

  _check(::cudaSetDevice(gpu_to));
  _check(::cudaMalloc(&data1, size));

  _check(::cudaHostAlloc(&data2, size, cudaHostAllocPortable));

  _check(::cudaDeviceSynchronize());

  // GPU Direct による転送
  {
    double       const wstart = ::omp_get_wtime();
    std::clock_t const cstart = std::clock();
    {
      _check(::cudaMemcpyPeer(data1, gpu_to, data0, gpu_from, size));
      _check(::cudaDeviceSynchronize());
    }
    std::clock_t const cend = std::clock();
    double       const wend = ::omp_get_wtime();

    double const wtime = (wend - wstart);
    double const ctime = (cend - cstart) / double(CLOCKS_PER_SEC);
    std::fprintf(stderr, "D wtime = %lf sec, %lf GB/sec\n", wtime, size / 1000000000.0 / wtime);
    std::fprintf(stderr, "D ctime = %lf sec, %lf GB/sec\n", ctime, size / 1000000000.0 / ctime);
  }

  // CPU を介した転送
  {
    double       const wstart = ::omp_get_wtime();
    std::clock_t const cstart = std::clock();
    {
      _check(::cudaSetDevice(gpu_from));
      _check(::cudaMemcpyAsync(data2, data0, size, cudaMemcpyDeviceToHost));
      _check(::cudaDeviceSynchronize());
      _check(::cudaSetDevice(gpu_to));
      _check(::cudaMemcpyAsync(data1, data2, size, cudaMemcpyHostToDevice));
      _check(::cudaDeviceSynchronize());
    }
    std::clock_t const cend = std::clock();
    double       const wend = ::omp_get_wtime();

    double const wtime = (wend - wstart);
    double const ctime = (cend - cstart) / double(CLOCKS_PER_SEC);
    std::fprintf(stderr, "I wtime = %lf sec, %lf GB/sec\n", wtime, size / 1000000000.0 / wtime);
    std::fprintf(stderr, "I ctime = %lf sec, %lf GB/sec\n", ctime, size / 1000000000.0 / ctime);
  }

  _check(::cudaSetDevice(gpu_from));
  _check(::cudaFree(data0));
  _check(::cudaDeviceReset());

  _check(::cudaSetDevice(gpu_to));
  _check(::cudaFree(data1));
  _check(::cudaFreeHost(data2));
  _check(::cudaDeviceReset());

  return 0;
}

