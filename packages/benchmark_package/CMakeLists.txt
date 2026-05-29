cmake_minimum_required(VERSION 3.0.2)
project(benchmark_package)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs std_msgs
)

install(DIRECTORY src/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)

install(PROGRAMS
  src/ros_benchmark.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)