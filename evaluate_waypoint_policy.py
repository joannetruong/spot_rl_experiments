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
from spot_wrapper.spot import Spot, wrap_heading

@hydra.main(config_path="config", config_name="waypoint_policy_config")
def main(cfg):
    print("Config parameters: ")
    print(OmegaConf.to_yaml(cfg))
    spot = Spot("RealNavEnv", cfg.disable_obstacle_avoidance)
    spot.get_lease(hijack=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = NavPolicy(cfg, device)
    policy.reset()

    env = SpotContextNavEnv(spot, cfg)
    env.sensor_type = cfg.sensor_type
    waypoints = cfg.waypoints

    env.goal_xy = waypoints[999].goal_x, waypoints[999].goal_y,
    for idx in waypoints:
        policy.reset()
        goal_x, goal_y = waypoints[idx].goal_x, waypoints[idx].goal_y
        print(f"NAVIGATING TO X: {goal_x}m, Y: {goal_y}m")
        cx, cy, cyaw = spot.get_xy_yaw()
        print("curr pose: ", cx, cy, np.rad2deg(cyaw))
        if idx == 0:
            observations = env.reset([goal_x, goal_y])
            home_x, home_y, home_t = spot.get_xy_yaw()
            print('HOME POSE: ', home_x, home_y, home_t)
        else:
            if cfg.use_local_coords:
                spot.home_robot(yaw=spot.boot_yaw)
                cx, cy, cyaw = spot.get_xy_yaw()
                print("homed curr pose: ", cx, cy, np.rad2deg(cyaw))
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
        input('press to continue')
    time.sleep(20)
    spot.power_off()


if __name__ == "__main__":
    main()
