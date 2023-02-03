#!/usr/bin/env bash

rosbag record -o bags/1-28/baseline_policy.bag /spot_cams/filtered_front_depth /spot_pose /spot_cams/frontleft_depth /spot_cams/frontright_depth & \
#rosbag record -o bags/spot_policy_gray_cam.bag /spot_cams/front_gray & \
python evaluate_reality.py

