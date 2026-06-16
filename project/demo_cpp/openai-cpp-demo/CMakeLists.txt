cmake_minimum_required(VERSION 3.10)
project(openai_cpp_demo)
set(CMAKE_CXX_STANDARD 17)

find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(openai_cpp_demo src/main.cpp)

# 可执行文件链接阶段追加curl
target_link_libraries(openai_cpp_demo
    # 你原有其他库...
    ${CURL_LIBRARIES}
    ${OpenCV_LIBS}
)

add_executable(one_turn_chat src/one_turn_chat.cpp)
target_link_libraries(one_turn_chat
    # 你原有其他库...
    ${CURL_LIBRARIES}
    ${OpenCV_LIBS}
)

add_executable(one_turn_chat_no_think_curl src/one_turn_chat_no_think_curl.cpp)
target_link_libraries(one_turn_chat_no_think_curl
    # 你原有其他库...
    ${CURL_LIBRARIES}
    ${OpenCV_LIBS}
)