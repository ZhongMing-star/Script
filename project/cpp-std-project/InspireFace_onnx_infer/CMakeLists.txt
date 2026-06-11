cmake_minimum_required(VERSION 3.10)
project(face_detection_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

# --------------------------
# ROS 2 依赖
# --------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

# --------------------------
# ONNX Runtime 配置（绝对路径，不使用 -l 方式）
# --------------------------
set(ORT_INCLUDE_DIR "/usr/local/include")
set(ORT_LIBRARY "/usr/local/lib/libonnxruntime.so")

# --------------------------
# 头文件路径（全部写在前面）
# --------------------------
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${ORT_INCLUDE_DIR}
  ${CURL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# --------------------------
# 生成可执行文件
# --------------------------
add_executable(face_detection_cpp
  src/main.cpp
  include/insight_face_onnx_feature_extract.cpp
  include/insight_face_onnx_detector.cpp
  include/base_onnx_infer.cpp
)

# --------------------------
# ROS 2 依赖链接
# --------------------------
ament_target_dependencies(face_detection_cpp
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  OpenCV
  nlohmann_json
)

# --------------------------
# 第三方库链接（关键：直接写绝对路径）
# --------------------------
target_link_libraries(face_detection_cpp
  Eigen3::Eigen
  ${CURL_LIBRARIES}
  ${ORT_LIBRARY}  # 这里只写 /usr/local/lib/libonnxruntime.so，不写任何 -lonnxruntime
)

# --------------------------
# 安装规则
# --------------------------
install(TARGETS
  face_detection_cpp
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY resources/
  DESTINATION share/${PROJECT_NAME}/resources
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()