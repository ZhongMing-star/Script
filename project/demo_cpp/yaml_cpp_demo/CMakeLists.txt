cmake_minimum_required(VERSION 3.10)
project(yaml_cpp_demo)
set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/include)

# 查找 yaml-cpp 库
find_package(yaml-cpp CONFIG REQUIRED)

add_executable(yaml_cpp_demo src/main.cpp)
# 链接 yaml-cpp 库
target_include_directories(yaml_cpp_demo PRIVATE ${YAML_CPP_INCLUDE_DIR})
target_link_libraries(yaml_cpp_demo ${YAML_CPP_LIBRARIES})