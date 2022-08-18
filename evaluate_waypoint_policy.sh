#!/usr/bin/env bash

rosbag record -o bags/08-12/waypoint_policy.bag /spot_cams/filtered_front_depth /spot_pose & \
#rosbag record -o bags/spot_policy_gray_cam.bag /spot_cams/front_gray & \
python evaluate_waypoint_policy.py

