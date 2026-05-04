#include "gpu/gpu_memory_manager.h"
#include "gpu/i_gpu_processor.h"
#include <cuda_runtime.h>
#include <iostream>

namespace gml {

bool IGpuProcessor::cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

size_t IGpuProcessor::available_vram() {
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) return 0;
    return free_bytes;
}

}
