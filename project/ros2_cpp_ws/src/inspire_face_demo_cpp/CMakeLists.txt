# 1. 基础版本与项目定义
cmake_minimum_required(VERSION 3.10)
project(inspire_face_demo_cpp)

# C++标准配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译警告
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ========== RPATH配置：解决安装后so找不到【最优方案】 ==========
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

# ========== Step1：查找所有ROS/系统依赖 find_package ==========
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# ========== Step2：第三方闭源SDK(InspireFace) 头文件+动态库查找 ==========
# 第三方根目录
set(INSPIRE_ROOT ${CMAKE_SOURCE_DIR}/thirdparty/InspireFace)
# 头文件目录
include_directories(${INSPIRE_ROOT}/include ${CMAKE_SOURCE_DIR}/include)

# 查找libInspireFace.so
find_library(IF_LIB
  NAMES InspireFace
  PATHS ${INSPIRE_ROOT}/lib
  NO_DEFAULT_PATH
)
if(NOT IF_LIB)
  message(FATAL_ERROR "Not found libInspireFace.so in ${INSPIRE_ROOT}/lib")
endif()

# ========== Step3：构建可执行文件 ==========
add_executable(${PROJECT_NAME}
  src/main.cpp
)

# ========== Step4：链接库【统一PRIVATE关键字，规避target_link_libraries混用报错】 ==========
# 第三方库、Eigen用 target_link_libraries + PRIVATE
target_link_libraries(${PROJECT_NAME}
    ${IF_LIB}
)
# ROS依赖固定使用 ament_target_dependencies（内部自动PRIVATE，语法统一）
ament_target_dependencies(${PROJECT_NAME}
    rclcpp
    sensor_msgs
    std_msgs
    cv_bridge
    OpenCV
)

# ========== Step5：install安装（exe和so同目录，RPATH生效） ==========
# 可执行安装
install(TARGETS ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)
# 第三方.so 和exe放到同一个目录
install(FILES ${IF_LIB}
  DESTINATION lib/${PROJECT_NAME}
)
# 资源、配置放到share
install(DIRECTORY config/ resources/
  DESTINATION share/${PROJECT_NAME}
)

# ========== Step6：ROS打包收尾 ==========
ament_package()