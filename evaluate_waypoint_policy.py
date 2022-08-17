#!/home/spot/anaconda3/envs/outdoor-nav/bin/python

import argparse
import time

import cv2
import hydra
import numpy as np
import torch.cuda
from nav_env import SpotNavEnv, SpotContextNavEnv
from omegaconf import OmegaConf
from real_policy import NavPolicy, ContextNavPolicy
from spot_wrapper.spot import Spot, wrap_heading

@hydra.main(config_path="config", config_name="waypoint_policy_config")
def main(cfg):
    print("Config parameters: ")
    print(OmegaConf.to_yaml(cfg))
    spot = Spot("RealNavEnv", cfg.disable_obstacle_avoidance)
    spot.get_lease(hijack=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ContextNavPolicy(cfg, device)
    policy.reset()

    env = SpotContextNavEnv(spot, cfg)
    env.sensor_type = cfg.sensor_type
    waypoints = cfg.waypoints
    global_goal_x, global_goal_y =  waypoints[999].goal_x, waypoints[999].goal_y
    env.goal_xy = np.array([global_goal_x, global_goal_y])
    env.wpt_xy = 0.0, 0.0
    time.sleep(2)

    for idx in waypoints:
        if idx == 999:
            continue
        policy.reset()
        goal_x, goal_y = waypoints[idx].goal_x, waypoints[idx].goal_y

        print(f"NAVIGATING TO WPT # {idx}. X: {goal_x}m, Y: {goal_y}m")
        cx, cy, cyaw = spot.get_xy_yaw()
        print("curr pose: ", cx, cy, np.rad2deg(cyaw))
        if idx == 0:
            observations = env.reset([global_goal_x, global_goal_y], [goal_x, goal_y])
            time.sleep(2)
            home_x, home_y, home_t = spot.get_xy_yaw()
            print('HOME POSE: ', home_x, home_y, home_t)
        else:
            if cfg.use_local_coords:
                spot.home_robot(yaw=spot.boot_yaw)
                goal_x -= xy_diff[0]
                goal_y -= xy_diff[1]
            print(f'orig goal x: {goal_x}, orig goal y {goal_y}, xy_diff: {xy_diff}')
            env.wpt_xy = np.array([goal_x, goal_y], dtype=np.float32)
            env.num_actions = 0
            env.num_collisions = 0
            env.episode_distance = 0
            env.num_steps = 0
        print('STEPPING NOW!')
        done = False
        stop_time = None
        if cfg.timeout != -1:
            stop_time = time.time() + cfg.timeout
        while not done:
            if cfg.debug:
                img = np.concatenate([observations["spot_right_depth"], observations["spot_left_depth"]], axis=1)
                cv2.imwrite(
                    f"img/depth_{env.num_actions}.png",
                    (img * 255),
                )
            action = policy.act(observations, deterministic=cfg.deterministic)
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
        end_x, end_y, _ = spot.get_xy_yaw()
        xy_diff = np.array([end_x-goal_x, end_y-goal_y])
        time.sleep(0.25)
        # input('press to continue')
    time.sleep(20)
    spot.power_off()


if __name__ == "__main__":
    main()
