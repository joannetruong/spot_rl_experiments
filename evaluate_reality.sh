#!/usr/bin/env bash

GOAL=3,-5
WEIGHTS=weights/spot_cam_kinematic_hm3d_gibson_ckpt_27.pth
#WEIGHTS=weights/spot_depth_kinematic_hm3d_gibson_pepper_ckpt_11.pth
SENSOR_TYPE="rgb"
#WEIGHTS=weights/spot_gray_cam_kinematic_hm3d_gibson_ckpt_29.pth
#SENSOR_TYPE="gray"

killall -9 roscore
killall -9 rosmaster
roscore & \
python spot_ros_node.py & \
rosbag record -o bags/spot_policy_depth.bag /camera/aligned_depth_to_color/image_raw/compressed & \
#rosbag record -o bags/spot_policy_gray_cam.bag /camera/color/image_raw/compressed & \
python evaluate_reality.py \
       --goal ${GOAL} \
       --weights ${WEIGHTS} \
       --sensor-type ${SENSOR_TYPE} \
