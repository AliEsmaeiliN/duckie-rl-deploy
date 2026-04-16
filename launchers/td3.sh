#!/bin/bash
source /environment.sh
dt-launchfile-init
rosrun rl_package rl_node.py --algo td3
dt-launchfile-join