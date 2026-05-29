#!/bin/bash

source /environment.sh

# Initialize the Duckietown launchfile environment
dt-launchfile-init

echo "--------------------------------------------------------"
echo "🎯 Launching End-to-End Sim2Real Hardware Benchmark..."
echo "🤖 Target Vehicle: $VEHICLE_NAME"
echo "--------------------------------------------------------"

dt-exec rosrun benchmark_package ros_benchmark.py --device "cuda"

dt-launchfile-join