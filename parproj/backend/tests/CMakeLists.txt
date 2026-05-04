cmake_minimum_required(VERSION 3.18)
add_executable(run_tests
    test_csr_builder.cpp
    test_features.cpp
    test_ml.cpp
    test_main.cpp
)

target_link_libraries(run_tests PRIVATE core_lib features_lib ml_lib)

if(ENABLE_CUDA)
    target_link_libraries(run_tests PRIVATE gpu_lib)
endif()

if(ENABLE_OPENMP)
    target_link_libraries(run_tests PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME unit_tests COMMAND run_tests)
