#!/usr/bin/env bash

#rosbag record -o bags/spot_depth_hm3d_gibson_ckpt_16_collision_2.bag /spot_cams/filtered_front_depth & \
#rosbag record -o bags/spot_policy_gray_cam.bag /spot_cams/front_gray & \
python evaluate_reality.py

