CMAKE_minimum_required(VERSION 3.10)
project(eigen_cpp_demo)
SET(CMAKE_CXX_STANDARD 17)


find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(eigen_cpp_demo src/main.cpp)
target_link_libraries(eigen_cpp_demo Eigen3::Eigen)




