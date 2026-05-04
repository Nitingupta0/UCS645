cmake_minimum_required(VERSION 3.18)

option(ENABLE_OPENMP "Enable OpenMP parallelism"  ON)
option(ENABLE_CUDA   "Enable CUDA GPU acceleration" ON)
option(BUILD_TESTS   "Build unit tests"           ON)

if(ENABLE_CUDA)
    project(graph_ml_pipeline LANGUAGES CXX CUDA)
else()
    project(graph_ml_pipeline LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
endif()

if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    message(STATUS "CUDA toolkit found: ${CUDAToolkit_VERSION}")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(core_lib
    src/core/graph_loader.cpp
    src/core/csr_builder.cpp
    src/core/workload_splitter.cpp
)
if(ENABLE_OPENMP)
    target_link_libraries(core_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

add_library(features_lib
    src/features/degree_centrality.cpp
    src/features/betweenness_centrality.cpp
    src/features/pagerank.cpp
    src/features/clustering_coefficient.cpp
    src/features/feature_aggregator.cpp
)
target_link_libraries(features_lib PUBLIC core_lib)
if(ENABLE_OPENMP)
    target_link_libraries(features_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

if(ENABLE_CUDA)
    add_library(gpu_lib
        src/gpu/feature_normalizer.cu
        src/gpu/distance_matrix.cu
        src/gpu/gpu_memory_manager.cu
    )
    set_target_properties(gpu_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(gpu_lib PUBLIC core_lib CUDA::cudart CUDA::cublas)
endif()

add_library(ml_lib
    src/ml/decision_tree.cpp
    src/ml/random_forest.cpp
    src/ml/evaluator.cpp
)
target_link_libraries(ml_lib PUBLIC core_lib)
if(ENABLE_OPENMP)
    target_link_libraries(ml_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(graph_ml src/main.cpp)
target_link_libraries(graph_ml PRIVATE core_lib features_lib ml_lib)
if(ENABLE_CUDA)
    target_link_libraries(graph_ml PRIVATE gpu_lib)
endif()
if(ENABLE_OPENMP)
    target_link_libraries(graph_ml PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(graph_ml PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-O3 -march=native -Wall -Wextra>
    $<$<CXX_COMPILER_ID:Clang>:-O3 -march=native -Wall -Wextra>
)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
