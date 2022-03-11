#!/usr/bin/env bash

GOAL=4,-5
WEIGHTS=weights/spot_cam_kinematic_hm3d_gibson_ckpt_27.pth

killall -9 roscore
killall -9 rosmaster
roscore & \
python spot_ros_node.py & \
rosbag record -o bags/spot_policy_cam.bag /spot_cams/filtered_front_depth & \
python evaluate_reality.py \
       --goal ${GOAL} \
       --weights ${WEIGHTS} \
