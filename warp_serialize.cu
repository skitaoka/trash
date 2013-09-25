// __ballot により warp 中のスレッドを同期化するサンプル。
#include <iostream>
#include <cuda_runtime.h>

__global__ void
  kernel(int volatile * const a)
{
  int const tid = threadIdx.x & 31;
  int const i = threadIdx.x + blockIdx.x * blockDim.x;
  for (int j = 0; j < 32; ++j) {
    int const b = __ballot(1);
    if ((b & ((1U << tid) - 1U)) == 0) {
      a[i] = b;
      break;
    }
  }
}
 
int main(int argc, char* argv[])
{
  ::cudaSetDevice(0);
  {
    static int const n = 32;
    int a[n] = {};

    int * b;
    ::cudaMalloc(&b, sizeof(int) * n);
    ::cudaMemcpyAsync(b, a, sizeof(int) * n, cudaMemcpyHostToDevice);
    kernel<<<1, n>>>(b);
    ::cudaMemcpy(a, b, sizeof(int) * n, cudaMemcpyDeviceToHost);
    ::cudaFree(b);

    for (int i = 0; i < n; ++i) {
      unsigned const v = a[i];
      for (int k = 0; k < 32; ++k) {
        std::cout << ((v >> (31-k))&1);
      }
      std::cout << std::endl;
    }
  }
  ::cudaDeviceReset();

  return 0;
}
