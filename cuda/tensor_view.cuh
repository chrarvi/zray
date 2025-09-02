#ifndef TENSOR_VIEW_CUH_
#define TENSOR_VIEW_CUH_
#include <stddef.h>

#include <utility> // for std::index_sequence

template <typename T, int Rank>
struct TensorView {
    T* data;
    size_t shape[Rank];
    size_t strides[Rank];

    template <typename... Indices>
    __device__ T& at(Indices... idxs) {
        static_assert(sizeof...(idxs) == Rank, "Index count must match Rank");

        size_t idxs_arr[Rank] = { static_cast<size_t>(idxs)... };

        size_t offset = 0;
        #pragma unroll
        for (int i = 0; i < Rank; i++)
            offset += idxs_arr[i] * strides[i];

        return data[offset];
    }
};

#endif  // TENSOR_VIEW_CUH_
