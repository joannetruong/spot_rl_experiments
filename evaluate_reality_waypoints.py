#!/home/spot/anaconda3/envs/outdoor-nav/bin/python

import argparse
import time

import cv2
import hydra
import numpy as np
import torch.cuda
from nav_env import SpotNavEnv
from omegaconf import OmegaConf
from real_policy import NavPolicy
from spot_wrapper.spot import Spot

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
    for waypoint in waypoints:
        goal_x, goal_y = waypoint.goal_x, waypoint.goal_y
        print(f"NAVIGATING TO X: {goal_x}m, Y: {goal_y}m")
        print("curr pose: ", spot.get_xy_yaw())
        observations = env.reset([goal_x, goal_y])
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
        finally:
            spot.power_off()


if __name__ == "__main__":
    main()
