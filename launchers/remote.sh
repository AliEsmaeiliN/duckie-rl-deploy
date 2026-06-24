#!/bin/bash
source /environment.sh
dt-launchfile-init

export DEBUG_MODE=false

# Run the CLI node
python3 /code/catkin_ws/src/duckie-rl-deploy/packages/rl_utils/src/remote_cli_node.py

dt-launchfile-join