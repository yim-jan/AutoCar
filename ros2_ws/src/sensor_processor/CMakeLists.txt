cmake_minimum_required(VERSION 3.10)
project(sensor_processor)

# 查找依赖包
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(image_processor src/image_processor_node.cpp)

# 包含头文件目录
target_include_directories(image_processor PRIVATE
    include
    ${rclcpp_INCLUDE_DIRS}
    ${cv_bridge_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# 链接库（使用 target-based 方式）
ament_target_dependencies(image_processor
    rclcpp
    std_msgs
    sensor_msgs
    cv_bridge
)
target_link_libraries(image_processor
    ${OpenCV_LIBRARIES}
)

# 安装
install(TARGETS image_processor DESTINATION lib/${PROJECT_NAME})

# 必须加这个，否则找不到依赖的包
ament_package()
