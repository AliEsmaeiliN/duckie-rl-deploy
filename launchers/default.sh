#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)
export DEBUG_MODE=true
rosrun rl_package rl_node.py


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
