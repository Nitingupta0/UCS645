#pragma once
#include <cstddef>
#include <stdexcept>
#include <cuda_runtime.h>

namespace gml {





template<typename T>
class GpuBuffer {
public:
    GpuBuffer() = default;

    explicit GpuBuffer(size_t count) { allocate(count); }

    ~GpuBuffer() { free(); }


    GpuBuffer(const GpuBuffer&)            = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;


    GpuBuffer(GpuBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_) {
        o.ptr_ = nullptr; o.count_ = 0;
    }
    GpuBuffer& operator=(GpuBuffer&& o) noexcept {
        if (this != &o) { free(); ptr_ = o.ptr_; count_ = o.count_;
                          o.ptr_ = nullptr; o.count_ = 0; }
        return *this;
    }

    void allocate(size_t count) {
        free();
        cudaError_t err = cudaMalloc(&ptr_, count * sizeof(T));
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("cudaMalloc failed: ")
                                     + cudaGetErrorString(err));
        count_ = count;
    }

    void free() {
        if (ptr_) { cudaFree(ptr_); ptr_ = nullptr; count_ = 0; }
    }

    T*     get()   const { return ptr_; }
    size_t size()  const { return count_; }
    bool   empty() const { return count_ == 0; }


    void upload(const T* host_data, size_t count) {
        cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
    }


    void download(T* host_data, size_t count) const {
        cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
    }

private:
    T*     ptr_   = nullptr;
    size_t count_ = 0;
};





template<typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t count) { allocate(count); }
    ~PinnedBuffer() { free(); }

    PinnedBuffer(const PinnedBuffer&)            = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void allocate(size_t count) {
        free();
        cudaError_t err = cudaMallocHost(&ptr_, count * sizeof(T));
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("cudaMallocHost failed: ")
                                     + cudaGetErrorString(err));
        count_ = count;
    }

    void free() {
        if (ptr_) { cudaFreeHost(ptr_); ptr_ = nullptr; count_ = 0; }
    }

    T*     get()   const { return ptr_; }
    size_t size()  const { return count_; }
    T& operator[](size_t i) { return ptr_[i]; }

private:
    T*     ptr_   = nullptr;
    size_t count_ = 0;
};




#define GML_CUDA_CHECK(call)                                             \
    do {                                                                  \
        cudaError_t _err = (call);                                        \
        if (_err != cudaSuccess)                                          \
            throw std::runtime_error(std::string(__FILE__) + ":"          \
                + std::to_string(__LINE__) + " CUDA error: "              \
                + cudaGetErrorString(_err));                               \
    } while (0)

}
