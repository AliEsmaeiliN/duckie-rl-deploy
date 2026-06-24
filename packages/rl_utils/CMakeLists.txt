cmake_minimum_required(VERSION 3.0.2)
project(rl_utils)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package()

# Install the Python scripts so ROS can find them via rosrun
catkin_install_python(PROGRAMS
  src/remote_cli_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)