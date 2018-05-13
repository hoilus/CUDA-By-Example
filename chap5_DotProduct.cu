#include "./common/book.h"
#include <iostream>
#include <iomanip>

using namespace std;

#define imin(a, b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1) / threadsPerBlock);

__global__ void dot(double* a, double* b, double* c) {
  __shared__ double cache[threadsPerBlock];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  double temp = 0;
  while (tid < N) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }

  // set the cache values
  cache[cacheIndex] = temp;

  // synchronize threads in this block
  __syncthreads();

  // for reductions, threadsPerBlock must be a power of 2
  // because of the following code
  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }

  if (cacheIndex == 0)
    c[blockIdx.x] = cache[0];
}

int main(void) {
  // allocate memory on the CPU side
  double *a = new double [N]; 
  double *b = new double [N]; 
  double c;
  double *partial_c = new double [blocksPerGrid];

  // allocate memory on device
  double *dev_a, *dev_b, *dev_partial_c;
  cudaMalloc(&dev_a, N*sizeof(double));
  cudaMalloc(&dev_b, N*sizeof(double));
  cudaMalloc(&dev_partial_c, blocksPerGrid*sizeof(double));

  // fill in the host memory with data
  for (int i=0; i < N; i++) {
    a[i] = i;
    b[i] = i*2;
  }
  
  // copy the array 'a' and 'b' from cpu to gpu
  cudaMemcpy(dev_a, a, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N*sizeof(double), cudaMemcpyHostToDevice);

  dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

  // copy the array 'dev_partial_c' from gpu to cpu
  cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(double), cudaMemcpyDeviceToHost);

  // finish up the cpu side
  c = 0;
  for (int i=0; i < blocksPerGrid; i++)
    c += partial_c[i];

  #define sum_squares(x) (x*(x+1)*(2*x+1)/6)
  cout << "Does GPU value " << setprecision(6) << c << " = " << setprecision(6) << 2*sum_squares((double)(N-1)) << " ?" << endl;

  // free memory on the GPU side
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_partial_c);

  // free memory on the CPU side
  delete [] a;
  delete [] b;
  delete [] partial_c;
}
