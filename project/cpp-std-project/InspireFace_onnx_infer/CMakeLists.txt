cmake_minimum_required(VERSION 3.16)
project(face_106det)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 找到 OpenCV
find_package(OpenCV REQUIRED)

# 找到 ONNX Runtime（需提前安装）
set(ONNXRUNTIME_ROOT "/mnt/d/Data/code/Script/project/cpp-std-project/InspireFace_onnx_infer/thirdparty/onnxruntime-linux-x64-1.20.1") # 改为你的路径
# set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/thirdparty/onnxruntime-linux-x64-1.20.1") # 改为你的路径
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# 库文件
add_library(insight_face_onnx_infer_lib
    include/insight_face_onnx_infer.cpp
)
target_include_directories(insight_face_onnx_infer_lib PUBLIC include)
target_link_libraries(insight_face_onnx_infer_lib
    ${OpenCV_LIBS}
    onnxruntime
)

# SCRFD 测试程序
add_executable(main src/main.cpp)
target_link_libraries(main insight_face_onnx_infer_lib)