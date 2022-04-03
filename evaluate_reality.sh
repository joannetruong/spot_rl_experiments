#!/usr/bin/env bash

GOAL=3,-3
WEIGHTS=weights/spot_depth_kinematic_hm3d_gibson_redwood_2_ckpt_12.pth

killall -9 roscore
killall -9 rosmaster
roscore & \
python spot_ros_node.py & \
#rosbag record -o bags/spot_depth_splitnet_ckpt_7.bag /spot_cams/filtered_front_depth & \
#rosbag record -o bags/spot_policy_gray_cam.bag /spot_cams/front_gray & \
python evaluate_reality.py \
       --goal=${GOAL} \
       --weights=${WEIGHTS}
