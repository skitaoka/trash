//
// nvcc -O2 -arch=sm_35 -Xcompiler "/wd4819 /MT" cudaEventTest.cu
//
#include <vector>
#include <iostream>

#include <cuda_runtime_api.h>

//
// cudaStreamCreate / cudaStreamDestroy
// cudaStreamSynchronize ; wait on cpu
//
// cudaStreamWaitEvent(stream, event, 0);
//   event が有効になるまで stream の実行を待つ.
//
// cudaEventCreateWithFlags(&event, cudaEventDisableTiming) / cudaEventDestroy
// cudaEventSynchronize ; wait on cpu
//
// cudaEventRecord(event, stream);
//   stream が空になると event が有効になる.
//

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

#define NUM_CHUNKS 10
#define NUM_DATA   10000000
#define N          (NUM_CHUNKS*NUM_DATA)

int main()
{
  _check(::cudaSetDevice(0));

  cudaStream_t h_streams[NUM_CHUNKS];
  for (int i = 0; i < NUM_CHUNKS; ++i) {
    _check(::cudaStreamCreate(&h_streams[i]));
  }

  cudaEvent_t h_events[NUM_CHUNKS];
  for (int i = 0; i < NUM_CHUNKS; ++i) {
    _check(::cudaEventCreateWithFlags(&h_events[i], cudaEventDisableTiming));
  }

  double * d_a;
  _check(::cudaMalloc(&d_a, N * sizeof(double)));

  double * h_a;
  _check(::cudaMallocHost(&h_a, N * sizeof(double)));

  for (int i = 0; i < NUM_CHUNKS; ++i) {
    if (i > 0) {
      //_check(::cudaStreamWaitEvent(h_streams[i%3], h_events[i-1], 0));
      _check(::cudaStreamWaitEvent(h_streams[i%3], h_events[0], 0));
    }
    a_kernel<<<(NUM_DATA+1023)/1024, 1024, 0, h_streams[i%3]>>>(d_a+i*NUM_DATA, NUM_DATA);
    b_kernel<<<(NUM_DATA+1023)/1024, 1024, 0, h_streams[i%3]>>>(d_a+i*NUM_DATA, NUM_DATA);

    //_check(::cudaEventRecord(h_events[i], h_streams[i%3]));
    _check(::cudaEventRecord(h_events[0], h_streams[i%3]));

    ::cudaMemcpyAsync(h_a+i*NUM_DATA, d_a+i*NUM_DATA, NUM_DATA, cudaMemcpyDeviceToHost, h_streams[i%3]);
  }

  _check(::cudaDeviceSynchronize());

  _check(::cudaFree(d_a));
  _check(::cudaFreeHost(h_a));

  for (int i = 0; i < NUM_CHUNKS; ++i) {
    _check(::cudaStreamDestroy(h_streams[i]));
  }
  for (int i = 0; i < NUM_CHUNKS; ++i) {
    _check(::cudaEventDestroy(h_events[i]));
  }

  return 0;
}
