cmake_minimum_required(VERSION 3.5)
project(yolos_ros)

# 查找依赖包
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  message_generation
)

# ROS2支持
find_package(ament_cmake QUIET)
if(ament_cmake_FOUND)
  find_package(rclpy REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(cv_bridge REQUIRED)
endif()

# 添加消息文件
add_message_files(
  FILES
  Detection.msg
  DetectionArray.msg
  BoundingBox.msg
)

# 生成消息
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

# catkin包配置
if(catkin_FOUND)
  catkin_package(
    CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs cv_bridge image_transport message_runtime
  )
endif()

# 安装Python脚本
if(catkin_FOUND)
  catkin_install_python(PROGRAMS
    scripts/detection_node.py
    scripts/camera_node.py
    scripts/image_publisher.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
endif()

# ROS2安装
if(ament_cmake_FOUND)
  install(PROGRAMS
    scripts/detection_node.py
    scripts/camera_node.py
    scripts/image_publisher.py
    DESTINATION lib/${PROJECT_NAME}
  )
  
  install(DIRECTORY
    launch/
    DESTINATION share/${PROJECT_NAME}/launch
  )
  
  ament_package()
endif()