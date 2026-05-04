#include "gpu/distance_matrix.h"
#include "gpu/gpu_memory_manager.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <stdexcept>

namespace gml {






__global__ void kernel_add_norms(float* C,
                                  const float* norms,
                                  int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    float d = C[row * n + col] + norms[row] + norms[col];
    C[row * n + col] = (d > 0.0f) ? sqrtf(d) : 0.0f;
}

std::vector<float> DistanceMatrixCUDA::compute(
    const FeatureMatrix& fm,
    const std::vector<int64_t>& indices) {


    std::vector<int64_t> idx;
    if (indices.empty()) {
        idx.resize(fm.num_vertices);
        for (int64_t i = 0; i < fm.num_vertices; ++i) idx[i] = i;
    } else {
        idx = indices;
    }

    int n = static_cast<int>(idx.size());
    int F = static_cast<int>(fm.num_features);


    std::vector<float> h_A(n * F);
    for (int i = 0; i < n; ++i)
        for (int f = 0; f < F; ++f)
            h_A[i * F + f] = fm.at(idx[i], f);


    GpuBuffer<float> d_A(n * F);
    d_A.upload(h_A.data(), n * F);


    GpuBuffer<float> d_C(n * n);


    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = -2.0f, beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                n, n, F,
                &alpha,
                d_A.get(), F,
                d_A.get(), F,
                &beta,
                d_C.get(), n);
    cublasDestroy(handle);


    std::vector<float> h_norms(n, 0.0f);
    for (int i = 0; i < n; ++i)
        for (int f = 0; f < F; ++f)
            h_norms[i] += h_A[i * F + f] * h_A[i * F + f];

    GpuBuffer<float> d_norms(n);
    d_norms.upload(h_norms.data(), n);


    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);
    kernel_add_norms<<<grid, block>>>(d_C.get(), d_norms.get(), n);
    GML_CUDA_CHECK(cudaDeviceSynchronize());


    std::vector<float> result(n * n);
    d_C.download(result.data(), n * n);
    return result;
}

}
