#include <cuda_runtime.h>
#include <math.h>

__global__ void kernel(unsigned char *img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    img[idx + 0] = (unsigned char)(255.0f *x / width);
    img[idx + 1] = (unsigned char)(255.0f *y / width);
    img[idx + 2] = 128;
}

extern "C" void launch_kernel(unsigned char *img, int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    kernel<<<grid, block>>>(img, width, height);
    cudaDeviceSynchronize();
}
