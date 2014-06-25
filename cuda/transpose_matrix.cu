template <typename T>
__global__ void transpose(T * dst, T const * src, int const w, int const h)
{
  int const x = threadIdx.x + blockIdx.x + blockDim.x;
  int const y = threadIdx.y + blockIdx.y + blockDim.y;

  if ((x < w) && (y < h)) {
    dst[x * h + y] = src[y * w + x];
  }
}

template <int kBlockSize, typename T>
__global__ void transpose(T * dst, T const * src, int const w, int const h)
{
  __shared__ T buf[kBlockSize][kBlockSize+1]; // +1 for avoiding bank conflicts.

  int const sx = threadIdx.x + blockIdx.x + blockDim.x;
  int const sy = threadIdx.y + blockIdx.y + blockDim.y;

  if ((sx < w) && (sy < h)) {
    buf[threadIdx.y][threadIdx.x] = src[sy * w + sx];
    __syncthreads();

    int const dx = threadIdx.x + blockIdx.y + blockDim.y;
    int const dy = threadIdx.y + blockIdx.x + blockDim.x;

    dst[dy * h + dx] = buf[threadIdx.x][threadIdx.y];
  }
}

