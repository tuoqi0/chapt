cmake_minimum_required(VERSION 2.8.3)
project(fast_lio_sam)

# 设置编译类型为 Release
SET(CMAKE_BUILD_TYPE "Release")  
ADD_COMPILE_OPTIONS(-std=c++14 )
set( CMAKE_CXX_FLAGS "-std=c++14 -O3" )

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# 设置C编译器标志
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")

message("Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processor number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

# 查找 OpenMP
find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

# 查找 Python
find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)


# 查找 Livox ROS Driver
set(livox_ros_driver2_DIR "/home/tuo/chatp4/chapt_ws/devel/share/livox_ros_driver2/cmake")
find_package(livox_ros_driver2 REQUIRED)

# 查找依赖的 ROS 包
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  tf2_geometry_msgs REQUIRED
  rospy
  std_msgs
  pcl_ros
  tf
  message_generation
  eigen_conversions
)

# 手动指定 GeographicLib 路径，确保它能够被正确找到
set(GeographicLib_DIR "/usr/include/GeographicLib")  # 根据你的安装路径设置
find_package(GeographicLib REQUIRED)

# 查找 GTSAM
find_package(GTSAM REQUIRED QUIET)

# 输出 Eigen 配置路径
message(Eigen: ${EIGEN3_INCLUDE_DIR})

# 包含依赖的头文件目录
include_directories(
  ${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${GeographicLib_INCLUDE_DIRS}
  include
  /home/tuo/chatp4/chapt4_ws/devel/include  # 添加 Livox 包的 include 目录
)
# 添加 ROOT_DIR 定义
add_definitions(-DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")
# 链接目标库
add_message_files(
  FILES
  Pose6D.msg
)

add_service_files(
  DIRECTORY srv
  FILES
  save_map.srv
  save_pose.srv
  AddPoint.srv
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
 std_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL GTSAM
  INCLUDE_DIRS
)

# 添加 ikd_Tree 和 preprocess 库
add_library(ikd_Tree include/ikd-Tree/ikd_Tree.cpp)
add_library(preprocess src/preprocess.cpp)

# 添加 fastlio_sam_mapping 可执行文件
add_executable(fastlio_sam_mapping src/laserMapping.cpp)
target_link_libraries(fastlio_sam_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} ${GeographicLib_LIBRARIES} ikd_Tree preprocess gtsam) 

# 包含 Python 库路径
target_include_directories(fastlio_sam_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})

