cmake_minimum_required(VERSION 2.8.3)
project(rl_package)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  duckietown_msgs
)

catkin_package()
catkin_python_setup()

catkin_install_python(PROGRAMS
  src/solution.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)