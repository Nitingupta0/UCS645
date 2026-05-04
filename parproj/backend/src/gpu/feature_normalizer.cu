#include "gpu/feature_normalizer.h"
#include "gpu/gpu_memory_manager.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace gml {






__global__ void kernel_col_stats(const float* __restrict__ data,
                                  int64_t N, int64_t F,
                                  double* col_sum,
                                  double* col_sum2,
                                  float*  col_min,
                                  float*  col_max) {
    int64_t f = blockIdx.x;
    if (f >= F) return;

    __shared__ double s_sum[256];
    __shared__ double s_sum2[256];
    __shared__ float  s_min[256];
    __shared__ float  s_max[256];

    int tid = threadIdx.x;
    s_sum[tid]  = 0.0;
    s_sum2[tid] = 0.0;
    s_min[tid]  =  1e30f;
    s_max[tid]  = -1e30f;

    for (int64_t v = tid; v < N; v += blockDim.x) {
        float val = data[v * F + f];
        s_sum[tid]  += val;
        s_sum2[tid] += (double)val * val;
        if (val < s_min[tid]) s_min[tid] = val;
        if (val > s_max[tid]) s_max[tid] = val;
    }
    __syncthreads();


    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid]  += s_sum[tid + stride];
            s_sum2[tid] += s_sum2[tid + stride];
            if (s_min[tid + stride] < s_min[tid]) s_min[tid] = s_min[tid + stride];
            if (s_max[tid + stride] > s_max[tid]) s_max[tid] = s_max[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        col_sum[f]  = s_sum[0];
        col_sum2[f] = s_sum2[0];
        col_min[f]  = s_min[0];
        col_max[f]  = s_max[0];
    }
}




__global__ void kernel_zscore(float* data,
                               int64_t N, int64_t F,
                               const float* mean, const float* std_inv) {
    int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t f = blockIdx.y;
    if (v >= N || f >= F) return;
    data[v * F + f] = (data[v * F + f] - mean[f]) * std_inv[f];
}




__global__ void kernel_minmax(float* data,
                               int64_t N, int64_t F,
                               const float* col_min, const float* range_inv) {
    int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t f = blockIdx.y;
    if (v >= N || f >= F) return;
    data[v * F + f] = (data[v * F + f] - col_min[f]) * range_inv[f];
}




void FeatureNormalizerCUDA::process(FeatureMatrix& fm) {
    int64_t N = fm.num_vertices;
    int64_t F = fm.num_features;
    if (N == 0 || F == 0) return;

    size_t bytes = N * F * sizeof(float);


    PinnedBuffer<float> h_data(N * F);
    std::copy(fm.data.begin(), fm.data.end(), h_data.get());

    GpuBuffer<float> d_data(N * F);
    d_data.upload(h_data.get(), N * F);


    GpuBuffer<double> d_sum(F), d_sum2(F);
    GpuBuffer<float>  d_min(F), d_max(F);


    dim3 stats_grid(static_cast<unsigned>(F));
    dim3 stats_block(256);
    kernel_col_stats<<<stats_grid, stats_block>>>(
        d_data.get(), N, F,
        d_sum.get(), d_sum2.get(), d_min.get(), d_max.get());
    GML_CUDA_CHECK(cudaDeviceSynchronize());


    std::vector<double> h_sum(F), h_sum2(F);
    std::vector<float>  h_min(F), h_max(F);
    d_sum.download(h_sum.data(), F);
    d_sum2.download(h_sum2.data(), F);
    d_min.download(h_min.data(), F);
    d_max.download(h_max.data(), F);


    std::vector<float> h_param1(F), h_param2(F);
    for (int64_t f = 0; f < F; ++f) {
        if (mode_ == NormMode::Z_SCORE) {
            float mean = static_cast<float>(h_sum[f] / N);
            double var = h_sum2[f] / N - (double)mean * mean;
            float std_inv = 1.0f / (std::sqrt(static_cast<float>(var)) + eps_);
            h_param1[f] = mean;
            h_param2[f] = std_inv;
        } else {
            float range = h_max[f] - h_min[f];
            h_param1[f] = h_min[f];
            h_param2[f] = (range > eps_) ? (1.0f / range) : 0.0f;
        }
    }

    GpuBuffer<float> d_p1(F), d_p2(F);
    d_p1.upload(h_param1.data(), F);
    d_p2.upload(h_param2.data(), F);


    unsigned block_x = 256;
    unsigned grid_x  = static_cast<unsigned>((N + block_x - 1) / block_x);
    dim3 norm_grid(grid_x, static_cast<unsigned>(F));
    dim3 norm_block(block_x, 1);

    if (mode_ == NormMode::Z_SCORE)
        kernel_zscore<<<norm_grid, norm_block>>>(d_data.get(), N, F,
                                                  d_p1.get(), d_p2.get());
    else
        kernel_minmax<<<norm_grid, norm_block>>>(d_data.get(), N, F,
                                                  d_p1.get(), d_p2.get());
    GML_CUDA_CHECK(cudaDeviceSynchronize());


    d_data.download(h_data.get(), N * F);
    std::copy(h_data.get(), h_data.get() + N * F, fm.data.begin());

    std::cout << "[FeatureNormalizerCUDA] Normalised "
              << N << "×" << F << " matrix ("
              << (mode_ == NormMode::Z_SCORE ? "z-score" : "min-max") << ")\n";
}

}
