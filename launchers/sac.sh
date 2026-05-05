#!/bin/bash
source /environment.sh
dt-launchfile-init
export DEBUG_MODE=false
dt-exec rosrun rl_package rl_node.py --algo sac
dt-launchfile-join