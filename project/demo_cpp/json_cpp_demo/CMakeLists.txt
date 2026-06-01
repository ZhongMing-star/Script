cmake_minimum_required(VERSION 3.10)
project(demo)

set(CMAKE_CXX_STANDARD 17)

# 包含自己的头文件
include_directories(include)

# 包含第三方库头文件 ✅
include_directories(thirdparty)

# 源文件
file(GLOB SRC src/*.cpp)

add_executable(main ${SRC})

# 线程库
if(UNIX)
    target_link_libraries(main pthread)
endif()