#include <iostream>
#include "cuda.h"
#include "./common/book.h"
#include "./common/cpu_bitmap.h"

#define INF 2e10f

struct Sphere {
  double r, b, g;
  double radius;
  double x, y, z;
  __device__ double hit(double ox, double oy, double *n) {
    double dx = ox - x;
    double dy = oy - y;
    if (dx*dx + dy*dy < radius*radius) {
      double dz = sqrtf(radius*radius - dx*dx - dy*dy);
      *n = dz / sqrtf(radius*radius);
      return dz + z;
    }
    return -INF;
  }
};

#define DIM 1024
#define rnd(x) (x * rand() / RAND_MAX)
#define SPHERES 50

using namespace std;

__constant__ Sphere s[SPHERES];
//Sphere *s;

__global__ void kernel(unsigned char *ptr) {
  // map from threadIdx	/blockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;
  double ox = (x - DIM/2);
  double oy = (y - DIM/2);

  double r = 0, g = 0, b = 0;
  double maxz = -INF;
  for (int i = 0; i < SPHERES; i++) {
    double n;
    double t = s[i].hit(ox, oy, &n);
    if (t > maxz) {
      double fscale = n;
      r = s[i].r * fscale;
      g = s[i].g * fscale;
      b = s[i].b * fscale;
      maxz = t;
    }
  }

  ptr[offset*4 + 0] = (int)(r * 255);
  ptr[offset*4 + 1] = (int)(g * 255);
  ptr[offset*4 + 2] = (int)(b * 255);
  ptr[offset*4 + 3] = 255;
}

int main(void) {
  // capture the start time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  CPUBitmap bitmap(DIM, DIM);
  unsigned char *dev_bitmap;

  // allocate memory on GPU
  cudaMalloc(&dev_bitmap, bitmap.image_size());
//  cudaMalloc(&s, SPHERES * sizeof(Sphere));

  // allocate memory on CPU and initialization
  Sphere *temp_s = new Sphere [SPHERES];
  for (int i = 0; i < SPHERES; i++) {
    temp_s[i].r = rnd(1.0f);
    temp_s[i].b = rnd(1.0f);
    temp_s[i].g = rnd(1.0f);
    temp_s[i].x = rnd(1000.0f) - 500;
    temp_s[i].y = rnd(1000.0f) - 500;
    temp_s[i].z = rnd(1000.0f) - 500;
    temp_s[i].radius = rnd(100.0f) + 20;
  }
//  cudaMemcpy(s, temp_s, SPHERES*sizeof(Sphere), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(s, temp_s, SPHERES*sizeof(Sphere));
  delete [] temp_s;

  // generate a bitmap from our sphere data
  dim3 grids(DIM/16, DIM/16);
  dim3 threads(16, 16);
  kernel<<<grids, threads>>>(dev_bitmap);

  // copy result from GPU to CPU
  cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

  // get stop time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cout << "Time to generate: " << elapsedTime << " ms." << endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //display
  bitmap.display_and_exit();

  // free GPU memory
  cudaFree(dev_bitmap);
//  cudaFree(s);

}
