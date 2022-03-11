#!/usr/bin/env bash

GOAL=5,-5
WEIGHTS=weights/spot_cam_kinematic_hm3d_gibson_ckpt_27.pth

killall -9 roscore
killall -9 rosmaster
roscore & \
python spot_ros_node.py & \
python evaluate_reality.py \
       --goal ${GOAL} \
       --weights ${WEIGHTS} \
