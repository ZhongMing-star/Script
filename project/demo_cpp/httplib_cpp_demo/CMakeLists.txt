CMAKE_minimum_required(VERSION 3.10)
project(httplib_cpp)
SET(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/thirdparty)

# 查找 yaml-cpp 库
find_package(yaml-cpp CONFIG REQUIRED)

add_executable(httplib_cpp src/main.cpp)

# 链接 yaml-cpp 库
target_include_directories(httplib_cpp PRIVATE ${YAML_CPP_INCLUDE_DIR})
target_link_libraries(httplib_cpp ${YAML_CPP_LIBRARIES})
