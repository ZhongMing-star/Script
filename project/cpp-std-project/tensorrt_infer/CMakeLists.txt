cmake_minimum_required(VERSION 3.16)

project(tensorrt_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(TENSORRT_ROOT "/usr/local/TensorRT-10.14.1.48" CACHE PATH "TensorRT installation prefix")

find_path(TENSORRT_INCLUDE_DIR
	NAMES NvInfer.h
	PATHS
		${TENSORRT_ROOT}/include
		/usr/local/include
		/usr/include
)

find_library(NVINFER_LIB
	NAMES nvinfer
	PATHS
		${TENSORRT_ROOT}/lib
		${TENSORRT_ROOT}/lib64
		/usr/local/lib
		/usr/lib/x86_64-linux-gnu
)

find_library(NVINFER_PLUGIN_LIB
	NAMES nvinfer_plugin
	PATHS
		${TENSORRT_ROOT}/lib
		${TENSORRT_ROOT}/lib64
		/usr/local/lib
		/usr/lib/x86_64-linux-gnu
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB OR NOT NVINFER_PLUGIN_LIB)
	message(FATAL_ERROR "找不到 TensorRT 头文件或库，请检查 TENSORRT_ROOT")
endif()

add_executable(tensorrt_infer
	src/main.cpp
)

target_include_directories(tensorrt_infer PRIVATE
	${TENSORRT_INCLUDE_DIR}
	${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(tensorrt_infer PRIVATE
	${OpenCV_LIBS}
	CUDA::cudart
	${NVINFER_LIB}
	${NVINFER_PLUGIN_LIB}
)

if (MSVC)
	target_compile_options(tensorrt_infer PRIVATE /W4)
else()
	target_compile_options(tensorrt_infer PRIVATE -Wall -Wextra -Wpedantic)
endif()
