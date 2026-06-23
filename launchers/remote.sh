#!/bin/bash
source /environment.sh
dt-launchfile-init

export DEBUG_MODE=false

# Run the CLI node
dt-exec rosrun rl_utils remote_cli_node.py

dt-launchfile-join