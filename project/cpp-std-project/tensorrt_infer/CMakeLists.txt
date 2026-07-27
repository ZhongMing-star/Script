cmake_minimum_required(VERSION 3.8)
project(vision_core)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS
    /usr/include
    /usr/local/include
    /usr/local/TensorRT-10.14.1.48/include
    /usr/local/TensorRT/include
  REQUIRED)

find_library(TENSORRT_NVINFER_LIBRARY nvinfer
  PATHS
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
    /usr/local/TensorRT-10.14.1.48/lib
    /usr/local/TensorRT/lib
  REQUIRED)

find_library(TENSORRT_NVINFER_PLUGIN_LIBRARY nvinfer_plugin
  PATHS
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
    /usr/local/TensorRT-10.14.1.48/lib
    /usr/local/TensorRT/lib
  REQUIRED)

find_library(CUDART_LIBRARY cudart
  PATHS
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
    /usr/local/lib
  REQUIRED)

find_path(CUDA_INCLUDE_DIR cuda_runtime.h
  PATHS
    /usr/local/cuda/include
    /usr/include
    /usr/local/include
  REQUIRED)

add_library(vision_core SHARED
  src/base_trt_infer.cpp
  src/common.cpp
  src/yolo_trt_detector.cpp
  src/curl_llm.cpp
  src/face_trt_detector.cpp)

target_include_directories(vision_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_include_directories(vision_core
  SYSTEM PUBLIC
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})

ament_target_dependencies(vision_core
  rclcpp
  std_msgs)

target_link_libraries(vision_core
  ${OpenCV_LIBS}
  ${TENSORRT_NVINFER_LIBRARY}
  ${TENSORRT_NVINFER_PLUGIN_LIBRARY}
  ${CUDART_LIBRARY})

install(TARGETS vision_core
  EXPORT export_vision_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY include/
  DESTINATION include)

ament_export_targets(export_vision_core HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp
  std_msgs
)

if(BUILD_TESTING)
  enable_testing()

  add_executable(curl_llm_validation_test
    test/curl_llm_validation_test.cpp
    src/curl_llm.cpp)

  target_include_directories(curl_llm_validation_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)

  target_link_libraries(curl_llm_validation_test
    CURL::libcurl)

  add_test(NAME curl_llm_validation_test
    COMMAND curl_llm_validation_test)
    
  add_executable(yolo_trt_detector_smoke_test
    test/yolo_trt_detector_smoke.cpp)

    
  target_compile_definitions(yolo_trt_detector_smoke_test 
    PRIVATE VISION_CORE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  target_include_directories(yolo_trt_detector_smoke_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_include_directories(yolo_trt_detector_smoke_test
    SYSTEM PRIVATE
      ${TENSORRT_INCLUDE_DIR}
      ${CUDA_INCLUDE_DIR}
      ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(yolo_trt_detector_smoke_test
    vision_core)
  add_test(NAME yolo_trt_detector_smoke_test
    COMMAND yolo_trt_detector_smoke_test)
  
  add_executable(face_trt_detector_test
      test/face_trt_detrctor_test.cpp)
  target_compile_definitions(face_trt_detector_test 
    PRIVATE VISION_CORE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  target_include_directories(face_trt_detector_test
    SYSTEM PRIVATE
      ${TENSORRT_INCLUDE_DIR}
      ${CUDA_INCLUDE_DIR}
      ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(face_trt_detector_test
    vision_core)
  add_test(NAME face_trt_detector_test
    COMMAND face_trt_detector_test)

  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
