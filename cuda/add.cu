#include <cuda_runtime.h>
#include "tensor_view.cuh"

#if defined(__cplusplus)
    #define EXTERN_C extern "C"
#else
    #define EXTERN_C extern
#endif

template <typename ValueT>
__global__ void __add_kernel(TensorView<ValueT, 2> a, TensorView<ValueT, 2> b, TensorView<ValueT, 2> c) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= a.shape[0] || x >= a.shape[1]) return;

    c.at(y, x) = a.at(y, x) + b.at(y, x);  // correct
}

EXTERN_C void add_f32_2d(TensorView<float, 2> a, TensorView<float, 2> b, TensorView<float, 2> c) {
    // Convert to templated struct
    dim3 block(16, 16);
    dim3 grid((a.shape[1] + block.x - 1) / block.x,
              (a.shape[0] + block.y - 1) / block.y);

    __add_kernel<float><<<grid, block>>>(a, b, c);
    cudaDeviceSynchronize();
}

EXTERN_C void add_i32_2d(TensorView<int, 2> a, TensorView<int, 2> b, TensorView<int, 2> c) {
    // Convert to templated struct
    dim3 block(16, 16);
    dim3 grid((a.shape[1] + block.x - 1) / block.x,
              (a.shape[0] + block.y - 1) / block.y);

    __add_kernel<int><<<grid, block>>>(a, b, c);
    cudaDeviceSynchronize();
}
