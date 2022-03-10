import time

import cv2
import gym
import numpy as np
from spot_wrapper.spot import Spot, wrap_heading

from spot_ros_node import SpotRosSubscriber

CTRL_HZ = 2
MAX_EPISODE_STEPS = 200

# Base action params
MAX_LIN_VEL = 0.5  # m/s
MAX_ANG_VEL = 0.3  # 17.19 degrees/s, in radians
VEL_TIME = 1 / CTRL_HZ

class SpotBaseEnv(SpotRosSubscriber, gym.Env):
    def __init__(self, spot: Spot):
        super().__init__("spot_reality_gym")
        self.spot = spot

        # General environment parameters
        self.ctrl_hz = CTRL_HZ
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.last_execution = time.time()
        self.should_end = False
        self.num_steps = 0
        self.reset_ran = False

        # Robot state parameters
        self.x, self.y, self.yaw = None, None, None

        # Base action parameters
        self.max_lin_vel = MAX_LIN_VEL
        self.max_ang_vel = MAX_ANG_VEL
        self.vel_time = VEL_TIME

        # Arrange Spot into initial configuration
        assert spot.spot_lease is not None, "Need motor control of Spot!"
        spot.power_on()
        spot.blocking_stand()

    def reset(self, *args, **kwargs):
        # Reset parameters
        self.num_steps = 0
        self.reset_ran = True
        self.should_end = False

        observations = self.get_observations()
        return observations

    def step(self, base_action=None):
        """Moves the arm and returns updated observations

        :param base_action: np.array of velocities (lineaer, angular)
        :return:
        """
        assert self.reset_ran, ".reset() must be called first!"
        assert base_action is not None, "Must provide action."

        if base_action is not None:
            # Command velocities using the input action
            x_vel, y_vel, ang_vel = base_action
            x_vel = np.clip(x_vel, -1, 1) * self.max_lin_vel
            y_vel = np.clip(x_vel, -1, 1) * self.max_lin_vel
            ang_vel = np.clip(ang_vel, -1, 1) * self.max_ang_vel
            # Spot-real's horizontal velocity is flipped from Habitat's convention
            self.spot.set_base_velocity(x_vel, -y_vel, ang_vel, self.vel_time)

        # Pause until enough time has passed during this step
        while time.time() < self.last_execution + 1 / self.ctrl_hz:
            pass
        env_hz = 1 / (time.time() - self.last_execution)
        self.last_execution = time.time()

        observations = self.get_observations()

        self.num_steps += 1
        timeout = self.num_steps == self.max_episode_steps
        done = timeout or self.get_success(observations) or self.should_end

        # Don't need reward or info
        reward = None
        info = {"env_hz": env_hz}

        return observations, reward, done, info

    @staticmethod
    def get_nav_success(observations, success_distance):
        # Is the agent at the goal?
        dist_to_goal, _ = observations["target_point_goal_gps_and_compass_sensor"]
        at_goal = dist_to_goal < success_distance
        return at_goal

    def print_nav_stats(self, observations):
        rho, theta = observations["target_point_goal_gps_and_compass_sensor"]
        print(
            f"Dist to goal: {rho:.2f}\t"
            f"theta: {np.rad2deg(theta):.2f}\t"
            f"x: {self.x:.2f}\t"
            f"y: {self.y:.2f}\t"
            f"yaw: {np.rad2deg(self.yaw):.2f}\t"
        )

    def get_nav_observation(self, goal_xy):
        observations = {}

        # Get visual observations
        front_depth = cv2.resize(
            self.front_depth_img, (256, 256), interpolation=cv2.INTER_AREA
        )
        front_depth = np.float32(front_depth) / 255.0
        # Add dimension for channel (unsqueeze)
        front_depth = front_depth.reshape(*front_depth.shape[:2], 1)
        observations["spot_left_depth"], observations["spot_right_depth"] = np.split(
            front_depth, 2, 1
        )

        # Get rho theta observation
        curr_xy = np.array([self.x, self.y], dtype=np.float32)
        rho = np.linalg.norm(curr_xy - goal_xy)
        theta = np.arctan2(goal_xy[1] - self.y, goal_xy[0] - self.x) - self.yaw
        rho_theta = np.array([rho, wrap_heading(theta)], dtype=np.float32)

        # Get goal heading observation
        observations["target_point_goal_gps_and_compass_sensor"] = rho_theta

        return observations

    def get_observations(self):
        raise NotImplementedError

    def get_success(self, observations):
        raise NotImplementedError

    def should_end(self):
        return False
