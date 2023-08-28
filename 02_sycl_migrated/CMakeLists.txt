if(MAX_GPU)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=intel_gpu_pvc -Xsycl-target-backend '-options -ze-intel-enable-auto-large-GRF-mode' -D MAX_GPU")
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl ")
endif()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "-qmkl ${CMAKE_EXE_LINKER_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR}/02_sycl_migrated/Common/)

add_executable (02_sycl_migrated Samples/4_CUDA_Libraries/oceanFFT/oceanFFT_kernel.dp.cpp Samples/4_CUDA_Libraries/oceanFFT/oceanFFT.cpp.dp.cpp)
target_link_libraries(02_sycl_migrated sycl)


add_custom_target (run_sm cd ${CMAKE_SOURCE_DIR}/02_sycl_migrated/ && ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/02_sycl_migrated qatest)
