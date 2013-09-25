/*----------------------------------------------------------------------------
 * interleaved operations
 *
 * G-DEP cuda training material
 *
 * The contents of this file are provided as demonstration of a
 * programming technique. It is provided as-is for educational use.
 *
 * This kernel computes the vector 2-norm from given data. If the
 * usePrefetch template parameter is true, then prefetches the
 * next data into the 128-byte cache line using the preload() inline 
 * function.
 *
 * ./a.out
 *      CUDA synchronized results 6.93097e+12:time 404.146 [ms]
 *      CUDA synchronized results 6.93097e+12:time 534.483 [ms]
 *                   host results 6.93097e+12
 *----------------------------------------------------------------------------
 * 2011.02.28 tawara initial.
 */
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <vector>
#include <cuda_runtime_api.h>

typedef unsigned int IndexT;
typedef double       DataT;

#define ANSI_BLUE  "\033[34m"
#define ANSI_GREEN "\033[32m"
#define ANSI_RESET "\033[0m"

/*====================================================================*/
// device code
/*====================================================================*/

__device__ __inline__ void preload(const void * const addr)
{
  asm __volatile__ (
                    "prefetch.global.L1 [%0];"
                    ::
#if COMPILE_32BIT_CODE
                    "r"(addr)
#else
                    "lr"(addr)
#endif
                    );
}

/**
 * \~english
 * \brief Calculate vector norm
 *
 * Given an array of vector data, the vector length, and synchronization
 * memory, compute the vector norm.
 * syncmem should be pre-initialized to some value.
 *
 * @param sz      size of data
 * @param data    memory array of type DataT
 * @param syncmem memory array of size at least gridDim.x * 2
 *
 */
template<const bool usePrefetch, int nthreads>
__global__ __launch_bounds__(1024) void kernel_norm(std::size_t sz, DataT *data)
{
  __shared__ DataT sdata[1024+16];
  volatile   DataT *sptr = sdata;
  DataT acc = 0.0;
  const std::size_t stride = blockDim.x;

  std::size_t idx = threadIdx.x;
  for(; idx < sz - stride; idx += stride) {
    if (usePrefetch) {
      preload(&data[idx + stride]);
    }
    acc += data[idx] * data[idx];
  }
  if(idx < sz) { 
    acc += data[idx] * data[idx];
  }

  //
  // reduction
  //
  sdata[threadIdx.x] = acc;
  __syncthreads();

  if(nthreads >= 512) { 
    if (threadIdx.x < 256) { 
      sdata[threadIdx.x] = acc = acc + sdata[threadIdx.x + 256];
    }
    __syncthreads();
  }
  if(nthreads >= 256) { 
    if (threadIdx.x < 128) { 
      sdata[threadIdx.x] = acc = acc + sdata[threadIdx.x + 128];
    }
    __syncthreads(); 
  }
  if(nthreads >= 128) { 
    if (threadIdx.x < 64) { 
      sdata[threadIdx.x] = acc = acc + sdata[threadIdx.x + 64];
    }
    __syncthreads(); 
  }
  if (threadIdx.x < 32) {
    if (nthreads >= 64) { 
      sptr[threadIdx.x] = acc = acc + sptr[threadIdx.x + 32]; };
    if (nthreads >= 32) { 
      sptr[threadIdx.x] = acc = acc + sptr[threadIdx.x + 16]; };
    if (nthreads >= 16) { 
      sptr[threadIdx.x] = acc = acc + sptr[threadIdx.x + 8];  };
    if (nthreads >= 8) { 
      sptr[threadIdx.x] = acc = acc + sptr[threadIdx.x + 4];  };
    if (nthreads >= 4) { 
      sptr[threadIdx.x] = acc = acc + sptr[threadIdx.x + 2];  };
    if (nthreads >= 2) { 
      sptr[threadIdx.x] = acc = acc + sptr[threadIdx.x + 1];  };
  }

  if(threadIdx.x == 0) {
    data[blockIdx.x * blockDim.x] = sptr[0];
  }

  if (threadIdx.x == 0) {
    data[0] = sqrt(acc);
  }
}

/*====================================================================*/
// host code
/*====================================================================*/

/**
 * \~english
 * \brief Calculate vector norm
 *
 * Given an array of vector data and the vector length compute the 
 * vector norm.
 *
 * @param sz      size of data
 * @param data    memory array of type DataT
 *
 */
void hostcode(std::size_t n, const double *data)
{
  double acc;
  for(std::size_t idx = 0; idx < n; ++idx) {
    acc += data[idx] * data[idx];
  }
  printf(ANSI_BLUE "%32s %g" ANSI_RESET "\n", "host results", sqrt(acc));
}

void runCode(int blocks)
{
  cudaError_t cerr;

  //
  // initialize vector memory
  //
  const std::size_t sz = (500UL*1024UL*1024UL);
  std::vector<double> data(sz);
  double *dev_data = NULL;
  
  cerr = cudaMalloc(&dev_data, sizeof(double)*sz);
  assert(cerr == cudaSuccess);

  printf("preparing data...\n");
  for(std::size_t idx=0; idx < sz; idx++) {
    data[idx] = static_cast<double>(idx);
  }

  //
  // compute cuda values using synchronization
  //
  {
    float kerneltime;
    cudaEvent_t event[2];
    cudaEventCreateWithFlags(&event[0], cudaEventBlockingSync);
    cudaEventCreateWithFlags(&event[1], cudaEventBlockingSync);
    cudaMemcpy(dev_data, &data[0], sizeof(double)*sz, cudaMemcpyHostToDevice);
    cudaEventRecord(event[0]);
    cudaEventSynchronize(event[0]);

    kernel_norm<true, 512><<<blocks, 512>>>(sz, dev_data);

    cudaThreadSynchronize();
    cudaEventRecord(event[1]);
    cudaEventSynchronize(event[1]);
    cudaEventElapsedTime(&kerneltime, event[0], event[1]);

    double result;
    cudaMemcpy(&result, &dev_data[0], sizeof(double)*1, cudaMemcpyDeviceToHost);
    printf(ANSI_GREEN "%32s %g:time %g [ms]" ANSI_RESET "\n", "CUDA synchronized results", result, kerneltime);
  }

  //
  // compute cuda values without using synchronization
  //
  {
    float kerneltime;
    cudaEvent_t event[2];
    cudaEventCreateWithFlags(&event[0], cudaEventBlockingSync);
    cudaEventCreateWithFlags(&event[1], cudaEventBlockingSync);
    cudaMemcpy(dev_data, &data[0], sizeof(double)*sz, cudaMemcpyHostToDevice);
    cudaEventRecord(event[0]);
    cudaEventSynchronize(event[0]);

    kernel_norm<false, 512><<<blocks, 512>>>(sz, dev_data);

    cudaThreadSynchronize();
    cudaEventRecord(event[1]);
    cudaEventSynchronize(event[1]);
    cudaEventElapsedTime(&kerneltime, event[0], event[1]);

    double result;
    cudaMemcpy(&result, &dev_data[0], sizeof(double)*1, cudaMemcpyDeviceToHost);
    printf(ANSI_GREEN "%32s %g:time %g [ms]" ANSI_RESET "\n", "CUDA synchronized results", result, kerneltime);
  }

  //
  // compute host values
  //
  hostcode(sz, &data[0]);

  cudaFree(dev_data);
}

int main(int argc, char *argv[])
{
  char *devstr = getenv("SEMINAR_USE_DEVICE");
  int devno = 0;
  if(devstr) {
    devno = atoi(devstr);
  }
  cudaError_t cerr;
  cerr = cudaSetDevice(devno);
  assert(cerr == cudaSuccess);
  printf("using device #%d\n", devno);

  runCode(1);
  cudaThreadExit();
  return 0;
}
