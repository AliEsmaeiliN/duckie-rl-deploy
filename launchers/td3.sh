#!/bin/bash
source /environment.sh
dt-launchfile-init
export PYTHONPATH="${PYTHONPATH}:${DT_REPO_PATH}"
rosrun rl_package solution.py --algo td3
dt-launchfile-join