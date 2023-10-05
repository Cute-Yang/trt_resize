cmake_minimum_required(VERSION 3.15)

project(
    self_plugin
    LANGUAGES CXX CUDA
    VERSION 0.0.1
)

set(TensorRT_HOME "/data/home/sunyyao/lazydog_build/TensorRT")
set(TensorRT_INCLUDE ${TensorRT_HOME}/include)
set(TensorRT_LIB_PATH ${TensorRT_HOME}/lib)
find_package(CUDA REQUIRED)

find_library(
    TRT_INFER
    nvinfer
    HINTS ${TensorRT_LIB_PATH}   
)

find_library(
    TRT_PLUGIN
    nvinfer_plugin
    HINTS ${TensorRT_LIB_PATH}
)
#这里要根据你的显卡来进行型号选取，防止产生不兼容的代码
# set(_NVCC_FLAGS)
# set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")
# set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_53,code=sm_53")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${_NVCC_FLAGS}")
file(GLOB plugin_srcs src/*.hpp src/*.cpp src/*.cu)
message("srcs:${plugin_srcs}")
set(plugin_lib_name "trt_dynamic_resize")
add_library(
    ${plugin_lib_name} SHARED
    ${plugin_srcs}
)

target_include_directories(
    ${plugin_lib_name}
    PUBLIC ${TensorRT_INCLUDE}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(
    ${plugin_lib_name}
    PRIVATE ${TRT_INFER}
    PRIVATE ${TRT_PLUGIN}
)

