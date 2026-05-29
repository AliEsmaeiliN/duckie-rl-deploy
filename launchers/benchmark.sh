#!/bin/bash

source /environment.sh

# Initialize the Duckietown launchfile environment
dt-launchfile-init

DEVICE="cpu"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --device) DEVICE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

export VEHICLE_NAME=${VEHICLE_NAME:-"duckie1nav"}
export DT_REPO_PATH="/code/duckie-rl-deploy"

echo "--------------------------------------------------------"
echo "🎯 Launching End-to-End Sim2Real Hardware Benchmark..."
echo "🤖 Target Vehicle: $VEHICLE_NAME"
echo "--------------------------------------------------------"

dt-exec rosrun rl_package ros_benchmark.py --device "$DEVICE"

dt-launchfile-join