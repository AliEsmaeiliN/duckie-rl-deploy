#!/bin/bash
source /environment.sh
dt-launchfile-init
export PYTHONPATH="${PYTHONPATH}:${DT_REPO_PATH}"
rosrun rl_package solution.py --algo sac
dt-launchfile-join