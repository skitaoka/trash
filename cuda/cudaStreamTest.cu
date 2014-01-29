//
// nvcc -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\lib\x64" -lcudadevrt -rdc=true  -O2 -arch=sm_35 -Xcompiler "/wd4819 /MT" cudaStreamTest.cu
//
#include <vector>
#include <iostream>

#include <cuda_runtime_api.h>

void _check(cudaError_t const e)
{
  if (e) {
    std::fprintf(stderr, "%s\n", ::cudaGetErrorString(e));
    std::exit(1);
  }
}

__global__ void a_kernel(double * const a, int const n)
{
  int const i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    a[i] = i;
  }
}

__global__ void b_kernel(double * const a, int const n)
{
  int const i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    double const x = a[i];
    a[i] = x * x;
  }
}

__global__ void c_kernel(double * const a, int const n)
{
  a_kernel<<<(n+1023)/1024, 1024>>>(a, n);
  b_kernel<<<(n+1023)/1024, 1024>>>(a, n);
}

__global__ void d_kernel(double * const a, int const n, cudaStream_t const * const streams, int const num_streams)
{
  int const i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_streams) {
    int const m = n / num_streams;
    a_kernel<<<(n+1023)/1024, 1024, 0, streams[i]>>>(a+i*m, m);
    b_kernel<<<(n+1023)/1024, 1024, 0, streams[i]>>>(a+i*m, m);
  }
}

#define NUM_CHUNKS 2
#define NUM_DATA   50000000
#define N          (NUM_CHUNKS*NUM_DATA)

int main()
{
  _check(::cudaSetDevice(0));

  cudaStream_t h_streams[NUM_CHUNKS];
  for (int i = 0; i < NUM_CHUNKS; ++i) {
    _check(::cudaStreamCreate(&h_streams[i]));
  }

  double * d_a;
  _check(::cudaMalloc(&d_a, N * sizeof(double)));

  double * h_a;
  _check(::cudaMallocHost(&h_a, N * sizeof(double)));

#if 0
  // (a)
  for (int i = 0; i < NUM_CHUNKS; ++i) {
    a_kernel<<<(NUM_DATA+1023)/1024, 1024, 0, h_streams[i]>>>(d_a+i*NUM_DATA, NUM_DATA);
    b_kernel<<<(NUM_DATA+1023)/1024, 1024, 0, h_streams[i]>>>(d_a+i*NUM_DATA, NUM_DATA);
    ::cudaMemcpyAsync(h_a+i*NUM_DATA, d_a+i*NUM_DATA, NUM_DATA, cudaMemcpyDeviceToHost, h_streams[i]);
  }
#elif 0
  // (b)
  for (int i = 0; i < NUM_CHUNKS; ++i) {
    c_kernel<<<1, 1, 0, h_streams[i]>>>(d_a+i*NUM_DATA, NUM_DATA);
  }
  for (int i = 0; i < NUM_CHUNKS; ++i) {
    ::cudaMemcpyAsync(h_a+i*NUM_DATA, d_a+i*NUM_DATA, NUM_DATA, cudaMemcpyDeviceToHost, h_streams[i]);
  }
#else
  // (c)
  cudaStream_t * g_streams;
  _check(::cudaMalloc(&g_streams, NUM_CHUNKS * sizeof(cudaStream_t)));
  _check(::cudaMemcpy(g_streams, h_streams, NUM_CHUNKS * sizeof(cudaStream_t), cudaMemcpyHostToDevice));

  d_kernel<<<1, NUM_CHUNKS>>>(d_a, N, g_streams, NUM_CHUNKS);

  for (int i = 0; i < NUM_CHUNKS; ++i) {
    ::cudaMemcpyAsync(h_a+i*NUM_DATA, d_a+i*NUM_DATA, NUM_DATA, cudaMemcpyDeviceToHost, h_streams[i]);
  }

  _check(::cudaDeviceSynchronize());
  _check(::cudaFree(g_streams));
#endif
  _check(::cudaDeviceSynchronize());

  _check(::cudaFree(d_a));
  _check(::cudaFreeHost(h_a));
  for (int i = 0; i < NUM_CHUNKS; ++i) {
    _check(::cudaStreamDestroy(h_streams[i]));
  }

  return 0;
}
