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
from spot_wrapper.spot import Spot


@hydra.main(config_path="config", config_name="map_policy_config")
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
    goal_x, goal_y = cfg.goal_x, cfg.goal_y
    print(f"NAVIGATING TO X: {goal_x}m, Y: {goal_y}m")
    print("curr pose: ", spot.get_xy_yaw())
    time.sleep(2)
    observations = env.reset([goal_x, goal_y])
    done = False
    time.sleep(2)
    stop_time = None
    if cfg.timeout != -1:
        stop_time = time.time() + cfg.timeout
    try:
        while not done:
            if cfg.use_keyboard:
                input('press to continue')
            if cfg.debug:
                img = np.concatenate([observations["spot_right_depth"], observations["spot_left_depth"]], axis=1)
                # cv2.imwrite(
                #     f"img/depth_{env.num_actions}.png",
                #     (img * 255),
                # )
                # cv2.imwrite(f'debug_maps/context_map_0_{env.num_actions}.png', observations["context_map"][:, :, 0]*255.0)
                # cv2.imwrite(f'debug_maps/context_map_1_{env.num_actions}.png', observations["context_map"][:, :, 1]*255.0)
                debug_map = observations["context_map"][:, :, 0]
                debug_map[observations["context_map"][:, :, 1] == 1] = 0.3
                cv2.imwrite(f'debug_maps/debug_map_{env.num_actions}.png', debug_map*255.0)


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
        time.sleep(20)
    finally:
        spot.power_off()


if __name__ == "__main__":
    main()