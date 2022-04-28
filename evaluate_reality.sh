#!/usr/bin/env bash

#GOAL=12.38117509991416,-3.579463174584167
#GOAL=17.707214032816324,5.442589179764012
#GOAL=8.207022718096248,5.634948493067192
#GOAL=5.549698196284474,10.364897937117199
GOAL=3.334973034976443,3.530325819797179
#WEIGHTS=weights/spot_kinematic_hm3d_gibson_sd_1_ckpt_38.pth
#WEIGHTS=weights/spot_kinematic_hm3d_gibson_sd_1_ckpt_99.pth
#WEIGHTS=weights/spot_kinematic_hm3d_gibson_sd_2_ckpt_99.pth
#WEIGHTS=weights/spot_kinematic_hm3d_gibson_sd_3_ckpt_99.pth
WEIGHTS=weights/spot_dynamic_hm3d_gibson_sd_1_ckpt_10.pth
#WEIGHTS=weights/spot_dynamic_hm3d_gibson_sd_2_ckpt_37.pth
#WEIGHTS=weights/spot_dynamic_hm3d_gibson_sd_3_ckpt_44.pth

#killall -9 roscore
#killall -9 rosmaster
#roscore & \
python external_camera_ros_node.py & \
#roslaunch realsense2_camera rs_aligned_depth.launch & \
#rosbag record -o bags/spot_kinematic_hm3d_gibson_sd_1_ckpt_99_ep_1.bag /camera/aligned_depth_to_color/image_raw & \
rosbag record -o bags/spot_dynamic_hm3d_gibson_sd_1_ckpt_10_ep_5.bag /camera/aligned_depth_to_color/image_raw & \
#rosbag record -o bags/spot_policy_gray_cam.bag /spot_cams/front_gray & \
python evaluate_reality.py \
       --goal=${GOAL} \
       --weights=${WEIGHTS} \
#       -d
