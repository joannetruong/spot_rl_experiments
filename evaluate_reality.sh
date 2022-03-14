#!/usr/bin/env bash

GOAL=3,-5
WEIGHTS=weights/spot_cam_kinematic_hm3d_gibson_ckpt_27.pth
#WEIGHTS=weights/spot_depth_kinematic_hm3d_gibson_pepper_ckpt_11.pth
SENSOR_TYPE="depth"
#WEIGHTS=weights/spot_gray_cam_kinematic_hm3d_gibson_ckpt_29.pth
#SENSOR_TYPE="gray"

killall -9 roscore
killall -9 rosmaster
roscore & \
python spot_ros_node.py & \
rosbag record -o bags/spot_policy_depth.bag /spot_cams/filtered_front_depth & \
#rosbag record -o bags/spot_policy_gray_cam.bag /spot_cams/front_gray & \
python evaluate_reality.py \
       --goal ${GOAL} \
       --weights ${WEIGHTS} \
       --sensor-type ${SENSOR_TYPE} \
