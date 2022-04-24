#!/usr/bin/env bash

GOAL=4,0
WEIGHTS=weights/spot_depth_hm3d_gibson_no_vy_ckpt_38.pth

SENSOR_TYPE="depth"
POLICY_NAME="PointNavResNetPolicy"
#POLICY_NAME="PointNavBaselinePolicy"

#WEIGHTS=weights/spot_depth_splitnet_motion_loss_ckpt_17.pth
#WEIGHTS=weights/spot_depth_splitnet_motion_loss_finetune_ckpt_7000.pth
#SENSOR_TYPE="depth"
#POLICY_NAME="PointNavSplitNetPolicy"

#WEIGHTS=weights/spot_gray_kinematic_hm3d_gibson_ckpt_18.pth
#WEIGHTS=weights/spot_gray_kinematic_hm3d_gibson_two_cnns_ckpt_31.pth
#SENSOR_TYPE="gray"

#rosbag record -o bags/spot_depth_hm3d_gibson_ckpt_16_collision_2.bag /spot_cams/filtered_front_depth & \
#rosbag record -o bags/spot_policy_gray_cam.bag /spot_cams/front_gray & \
python evaluate_reality.py \
       --goal=${GOAL} \
       --weights=${WEIGHTS} \
       --sensor-type=${SENSOR_TYPE} \
       --policy-name=${POLICY_NAME} \
       --disable-obstacle-avoidance \
#       --no-horizontal-velocity \
#       -d


