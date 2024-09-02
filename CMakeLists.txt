cmake_minimum_required(VERSION 3.15.0)
project(
    lxd_test
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)
include( ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cmake/modules/resolve_dirs.cmake)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(Torch_DIR /root/miniconda3/envs/trtllm/lib/python3.10/site-packages/torch/share/cmake/Torch)
find_package(Torch REQUIRED)
resolve_dirs(CUDAToolkit_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")

message(STATUS "ThomasLee: ${CUDAToolkit_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS}")
add_executable(lxd_test ${CMAKE_CURRENT_SOURCE_DIR}/lxd_test.cpp)
target_include_directories(lxd_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include ${CUDAToolkit_INCLUDE_DIRS} /usr/local/tensorrt/include ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_include_directories(lxd_test PRIVATE ${CUDAToolkit_LIBRARY_DIR})
target_link_directories(lxd_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tensorrt_llm/libs)
target_link_libraries(lxd_test PRIVATE ${TORCH_LIBRARIES} ${Python3_LIBRARIES} tensorrt_llm th_common)