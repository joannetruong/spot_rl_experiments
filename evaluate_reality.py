#!/home/spot/anaconda3/envs/outdoor-nav/bin/python

import argparse
import time

import torch.cuda
from nav_env import SpotNavEnv
from real_policy import NavPolicy
from spot_wrapper.spot import Spot

NAV_WEIGHTS = "weights/spot_cam_kinematic_hm3d_gibson_ckpt_27.pth"
GOAL_XY = [1, 0]  # Local coordinates
GOAL_AS_STR = ",".join([str(i) for i in GOAL_XY])

print(GOAL_AS_STR)


def main(spot):
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--goal", default=GOAL_AS_STR)
    parser.add_argument("-w", "--weights", default=NAV_WEIGHTS)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = NavPolicy(args.weights, device=device)
    policy.reset()

    env = SpotNavEnv(spot)
    goal_x, goal_y = [float(i) for i in args.goal.split(",")]
    observations = env.reset([goal_x, goal_y])
    done = False
    time.sleep(2)
    try:
        while not done:
            action = policy.act(observations)
            observations, _, done, _ = env.step(base_action=action)
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
    spot = Spot("RealNavEnv")
    with spot.get_lease(hijack=True):
        main(spot)
