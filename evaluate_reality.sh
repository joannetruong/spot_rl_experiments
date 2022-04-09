#!/usr/bin/env bash

GOAL=5,-2
#WEIGHTS=weights/spot_kinematic_hm3d_gibson_sd_1_ckpt_99.pth
WEIGHTS=weights/spot_kinematic_hm3d_gibson_sd_2_ckpt_99.pth
#WEIGHTS=weights/spot_kinematic_hm3d_gibson_sd_3_ckpt_99.pth
#WEIGHTS=weights/spot_dynamic_hm3d_gibson_sd_2_ckpt_44.pth
#WEIGHTS=weights/spot_dynamic_hm3d_gibson_sd_3_ckpt_44.pth

#killall -9 roscore
#killall -9 rosmaster
#roscore & \
#roslaunch realsense2_camera rs_aligned_depth.launch & \
#rosbag record -o bags/spot_depth_splitnet_ckpt_7.bag /spot_cams/filtered_front_depth & \
#rosbag record -o bags/spot_policy_gray_cam.bag /spot_cams/front_gray & \
python evaluate_reality.py \
       --goal=${GOAL} \
       --weights=${WEIGHTS} \
       -d
