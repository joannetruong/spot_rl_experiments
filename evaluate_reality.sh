#!/usr/bin/env bash

python external_camera_ros_node.py & \
#roslaunch realsense2_camera rs_aligned_depth.launch & \
#rosbag record -o bags/spot_kinematic_hm3d_gibson_sd_1_ckpt_99_ep_1.bag /camera/aligned_depth_to_color/image_raw & \
rosbag record -o bags/spot_dynamic_hm3d_gibson_sd_1_ckpt_10_ep_5.bag /camera/aligned_depth_to_color/image_raw & \
python evaluate_reality.py 
