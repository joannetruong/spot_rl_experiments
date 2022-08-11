#!/usr/bin/env bash

rosbag record -o bags/06-03/${RUN_NAME}.bag /spot_cams/filtered_front_depth /spot_pose & \
#rosbag record -o bags/spot_policy_gray_cam.bag /spot_cams/front_gray & \
python evaluate_reality_waypoints.py

