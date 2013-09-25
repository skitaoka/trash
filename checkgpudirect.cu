// 同一バス上にある Peer 間のメモリ転送ができるかチェックする。
#include <cstdio>
#include <cstdlib>

#define N (1000)
#define NTHREADS (32)

__global__ void
  kernel(int const n, float * const data)
{
  int const i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    data[i] = i;
  }
}

int main(int argc, char * argv[])
{
  // Peer アクセスができるかチェックする
  int deviceCount;
  {
    cudaError_t const e = ::cudaGetDeviceCount(&deviceCount);
    if (e) {
      std::fprintf(stderr, "%s\n", ::cudaGetErrorString(e));
      return 1;
    }
  }
  for (int i = 0; i < deviceCount; ++i) {
    for (int j = 0; j < deviceCount; ++j) {
      if (i != j) {
        int canAccessPeer;
        cudaError_t const e = ::cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
        if (e) {
          std::fprintf(stderr, "%s\n", ::cudaGetErrorString(e));
        } else if (canAccessPeer) {
          std::fprintf(stderr, "%d <=> %d\n", i, j);
        }
      }
    }
  }

  // cudaMemcpyPeer     (dst, dstId, src, srcId, sizeOfBytes);
  // cudaMemcpyPeerAsync(dst, dstId, src, srcId, sizeOfBytes, streamId);

#if 0
  if (argc < 3) {
    std::fprintf(stderr, "Usage: %s gpu_from gpu_to\n", argv[0]);
    return 1;
  }

  int gpu_from = std::atoi(argv[1]);
  int gpu_to   = std::atoi(argv[2]);

  int canAccessPeer;
  ::cudaDeviceCanAccessPeer(&canAccessPeer, gpu_to, gpu_from);
  if (!canAccessPeer) {
    std::fprintf(stderr, "(%d-%d) cannot access peer.\n", gpu_from, gpu_to);
    return 1;
  }

  float * dev_data = NULL; // GPU 0 上のメモリ (GPU 1 によって値が設定される)
  float * hst_data = NULL; // CPU   上のメモリ

  ::cudaSetDevice(gpu_from);
  ::cudaMalloc(&dev_data, N * sizeof(float));

  ::cudaSetDevice(gpu_to);
  ::cudaDeviceEnablePeerAccess(gpu_from, 0); // デバイス 0 へのアクセスを有効化する
  kernel<<<(N+NTHREADS-1)/NTHREADS, NTHREADS>>>(N, dev_data);

  hst_data = new float[N];
  ::cudaMemcpy(hst_data, dev_data, N * sizeof(float), cudaMemcpyDeviceToHost);

  float sum = 0.0f;
  for (int i = 0; i < N; ++i) {
    sum += hst_data[i];
  }
  std::fprintf(stderr, "(%d-%d) sum = %f\n", gpu_from, gpu_to, sum);


  ::cudaFree(dev_data);
  delete [] hst_data;
#endif

  return 0;
}

