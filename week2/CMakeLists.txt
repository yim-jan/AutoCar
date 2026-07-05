cmake_minimum_required(VERSION 3.10)
project(SensorSimulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 头文件路径
include_directories(include)

# 可执行文件
add_executable(sensor_mt src/sensor_mt.cpp)

# 链接线程库
find_package(Threads REQUIRED)
target_link_libraries(sensor_mt Threads::Threads)
add_executable(sensor_mt_opt src/sensor_mt_opt.cpp)
target_link_libraries(sensor_mt_opt Threads::Threads)
