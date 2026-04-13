#!/bin/bash
source /environment.sh
dt-launchfile-init
export PYTHONPATH="${PYTHONPATH}:${DT_REPO_PATH}"
dt-exec python3 "${DT_REPO_PATH}/packages/solution.py" --algo sac
dt-launchfile-join