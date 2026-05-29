cmake_minimum_required(VERSION 3.10)
project(InspireFace_demo_cpp)
set(CMAKE_CXX_STANDARD 17)


# InspireFace SDK
set(INSPIREFACE_ROOT ${CMAKE_SOURCE_DIR}/thirdparty/InspireFace)
include_directories(${INSPIREFACE_ROOT}/include)
link_directories(${INSPIREFACE_ROOT}/lib)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(InspireFace_demo_cpp 
    src/main.cpp
    # 头文件可以加进来，方便 IDE 识别
    src/face_common.h
    src/inspireface_wrapper.hpp
)

target_link_libraries(InspireFace_demo_cpp 
    InspireFace 
    ${OpenCV_LIBS}
)

