#!/home/spot/anaconda3/envs/outdoor-nav/bin/python

import argparse
import time
import os
import cv2
import hydra
import numpy as np
import torch.cuda
from nav_env import SpotNavEnv
from omegaconf import OmegaConf
from real_policy import NavPolicy
from spot_wrapper.spot import Spot, wrap_heading

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 521))

@hydra.main(config_path="config", config_name="waypoint_config")
def main(cfg):
    print("Config parameters: ")
    print(OmegaConf.to_yaml(cfg))
    spot = Spot("RealNavEnv", cfg.disable_obstacle_avoidance)
    spot.get_lease(hijack=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = NavPolicy(cfg, device)
    policy.reset()

    env = SpotNavEnv(spot, cfg)
    env.sensor_type = cfg.sensor_type
    waypoints = cfg.waypoints

    start_idx = 0 if cfg.waypoint_start == -1 else cfg.waypoint_start
    end_idx = len(waypoints)-1 if cfg.waypoint_end == -1 else cfg.waypoint_end
    assert start_idx <= end_idx
    print(f'Waypoint start idx: {start_idx}, end idx: {end_idx}')
    for idx in waypoints:
        if idx > end_idx:
            break
        if idx < start_idx:
            continue
        goal_x, goal_y = waypoints[idx].goal_x, waypoints[idx].goal_y
        cx, cy, cyaw = spot.get_xy_yaw()
        if idx == start_idx:
            observations = env.reset([goal_x, goal_y])
        elif idx > start_idx:
            policy.reset()
            # # adjust for robot stopping at success radius instead of at waypoint
            if cfg.use_local_coords:
                spot.home_robot(yaw=spot.boot_yaw)
                goal_x -= xy_diff[0]
                goal_y -= xy_diff[1]
            print(f'orig goal x: {goal_x}, orig goal y {goal_y}, xy_diff: {xy_diff}')
            env.goal_xy = np.array([goal_x, goal_y], dtype=np.float32)
            env.num_actions = 0
            env.num_collisions = 0
            env.episode_distance = 0
            env.num_steps = 0
        print(f"##### NAVIGATING TO WAYPOINT # {idx}. X: {env.goal_xy[0]}m, Y: {env.goal_xy[1]}m #####")
        if cfg.use_keyboard:
            input('press to continue')
        done = False
        stop_time = None
        if cfg.timeout != -1:
            stop_time = time.time() + cfg.timeout
        try:
            while not done:
                if cfg.debug:
                    cv2.imwrite(
                        f"img/left_depth_{env.num_actions}.png",
                        (observations["spot_left_depth"] * 255),
                    )
                    cv2.imwrite(
                        f"img/right_depth_{env.num_actions}.png",
                        (observations["spot_right_depth"] * 255),
                    )
                action = policy.act(observations)
                observations, _, done, _ = env.step(base_action=action)
                if cfg.timeout != -1 and stop_time < time.time():
                    print("############# Timeout reached. Stopping ############# ")
                    done = True
                if done:
                    print(
                        "Final Agent Episode Distance: {:.3f}".format(env.episode_distance)
                    )
                    print(
                        "Final Distance to goal: {:.3f}m".format(
                            observations["pointgoal_with_gps_compass"][0]
                        )
                    )
                    print("Final # Actions: {}".format(env.num_actions))
                    print("Final # Collisions: {}".format(env.num_collisions))
        except:
            pass
        end_x, end_y, _ = spot.get_xy_yaw()
        xy_diff = np.array([end_x-goal_x, end_y-goal_y])
        time.sleep(0.25)
    try:
        spot.dock(DOCK_ID)
        spot.home_robot()
    except:
        pass
    time.sleep(20)
    spot.power_off()


if __name__ == "__main__":
    main()