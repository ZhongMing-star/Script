cmake_minimum_required(VERSION 3.14)
project(CppStdProject LANGUAGES CXX)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 全局头文件路径
include_directories(${PROJECT_SOURCE_DIR}/include)

# 1. 编译业务静态库
set(SRCS
    src/math/calc.cpp
    src/utils/str.cpp
)
add_library(core_lib STATIC ${SRCS})

# 2. 引入 GTest
add_subdirectory(thirdparty/googletest)
enable_testing()

# 3. 进入测试目录
add_subdirectory(tests)