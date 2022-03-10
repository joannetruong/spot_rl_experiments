import argparse
import time

import numpy as np
import torch.cuda
from spot_wrapper.spot import Spot
from spot_wrapper.utils import say

from base_env import SpotBaseEnv
from real_policy import NavPolicy

SUCCESS_DISTANCE = 0.3
SUCCESS_ANGLE_DIST = 0.0872665  # 5 radians
NAV_WEIGHTS = "weights/two_cams_with_noise_seed4_ckpt.4.pth"
GOAL_XY = [6, 0]
GOAL_AS_STR = ",".join([str(i) for i in GOAL_XY])


def main(spot):
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--goal", default=GOAL_AS_STR)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.reset()

    env = SpotNavEnv(spot)
    goal_x, goal_y = [float(i) for i in args.goal.split(",")]
    observations = env.reset(goal_x, goal_y)
    done = False
    say("Starting episode")
    time.sleep(2)
    try:
        while not done:
            action = policy.act(observations)
            observations, _, done, _ = env.step(base_action=action)
        say("Environment is done.")
        time.sleep(20)
    finally:
        spot.power_off()


class SpotNavEnv(SpotBaseEnv):
    def __init__(self, spot: Spot):
        super().__init__(spot)
        self.goal_xy = None
        self.succ_distance = SUCCESS_DISTANCE
        self.succ_angle = SUCCESS_ANGLE_DIST

    def reset(self, goal_xy):
        self.goal_xy = np.array(goal_xy, dtype=np.float32)
        observations = super().reset()
        assert len(self.goal_xy) == 2

        return observations

    def get_success(self, observations):
        succ = self.get_nav_success(observations, self.succ_distance, self.succ_angle)
        if succ:
            self.spot.set_base_velocity(0.0, 0.0, 0.0, self.vel_time)
        return succ

    def get_observations(self):
        return self.get_nav_observation(self.goal_xy)


if __name__ == "__main__":
    spot = Spot("RealNavEnv")
    with spot.get_lease(hijack=True):
        main(spot)
