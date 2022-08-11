#!/usr/bin/env bash

rosbag record -o bags/07-06/${RUN_NAME}.bag /spot_cams/filtered_front_depth & \
#rosbag record -o bags/spot_policy_gray_cam.bag /spot_cams/front_gray & \
python evaluate_reality.py

