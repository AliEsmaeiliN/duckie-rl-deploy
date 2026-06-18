#!/bin/bash

echo "Syncing the RL models with the duckiebot98"

rsync -avz ~/workspace/rl_models/ duckie@duckiebot98.local:/data/config/rl_models/