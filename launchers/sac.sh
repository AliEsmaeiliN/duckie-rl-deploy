#!/bin/bash
source /environment.sh
dt-launchfile-init
rosrun rl_package rl_node.py --algo sac
dt-launchfile-join