cmake_minimum_required(VERSION 3.10)
project(InspireFace_demo_cpp)
set(CMAKE_CXX_STANDARD 17)


# InspireFace SDK

find_path(INSPIREFACE_INCLUDE_DIR NAMES inspireface.h PATHS /usr/local/inspireface/include)
include_directories(${INSPIREFACE_INCLUDE_DIR})

# InspireFace SDK
find_library(INSPIREFACE_LIB NAMES InspireFace PATHS /usr/local/inspireface/lib)
if(NOT INSPIREFACE_LIB)
  message(FATAL_ERROR "Not found InspireFace in /usr/local/inspireface/lib")
endif()

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(InspireFace_demo_cpp 
    src/main.cpp
)

target_link_libraries(InspireFace_demo_cpp 
    ${OpenCV_LIBS}
    ${INSPIREFACE_LIB}
)

