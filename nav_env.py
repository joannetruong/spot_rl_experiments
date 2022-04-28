import cv2
import numpy as np
from base_env import SpotBaseEnv
from spot_wrapper.spot import Spot, wrap_heading

SUCCESS_DISTANCE = 0.425


class SpotNavEnv(SpotBaseEnv):
    def __init__(self, spot: Spot):
        super().__init__(spot)
        self.goal_xy = None
        self.succ_distance = SUCCESS_DISTANCE

    def reset(self, goal_xy):
        # self.spot.home_robot()
        self.goal_xy = np.array(goal_xy, dtype=np.float32)
        self.num_actions = 0
        self.num_collisions = 0
        self.episode_distance = 0
        observations = super().reset()
        assert len(self.goal_xy) == 2

        return observations

    @staticmethod
    def get_nav_success(observations, success_distance):
        # Is the agent at the goal?
        dist_to_goal, _ = observations["pointgoal_with_gps_compass"]
        at_goal = dist_to_goal < success_distance
        return at_goal

    def print_nav_stats(self, observations):
        rho, theta = observations["pointgoal_with_gps_compass"]
        print(
            f"Dist to goal: {rho:.2f}\t"
            f"theta: {np.rad2deg(theta):.2f}\t"
            f"x: {self.x:.2f}\t"
            f"y: {self.y:.2f}\t"
            f"yaw: {np.rad2deg(self.yaw):.2f}\t"
            f"# actions: {self.num_actions}\t"
            f"# collisions: {self.num_collisions}\t"
            f"# episode_distance: {self.episode_distance}\t"
        )

    def get_nav_observation(self, goal_xy):
        observations = {}
        img_obs = self.front_depth_img

        # Get visual observations
        front_obs = np.float32(img_obs) / 255.0
        front_obs = front_obs.reshape(*front_obs.shape[:2], 1)
        observations["depth"] = front_obs

        # Get rho theta observation
        self.x, self.y, self.yaw = self.spot.get_xy_yaw()
        curr_xy = np.array([self.x, self.y], dtype=np.float32)
        print("curr_xy: ", curr_xy, self.yaw)
        rho = np.linalg.norm(curr_xy - goal_xy)
        theta = np.arctan2(goal_xy[1] - self.y, goal_xy[0] - self.x) - self.yaw
        rho_theta = np.array([rho, wrap_heading(theta)], dtype=np.float32)

        # Get goal heading observation
        observations["pointgoal_with_gps_compass"] = rho_theta

        return observations

    def get_success(self, observations):
        succ = self.get_nav_success(observations, self.succ_distance)
        if succ:
            print("SUCCESS!")
            self.spot.set_base_velocity(0.0, 0.0, 0.0, self.vel_time)
            self.print_nav_stats(observations)
        return succ

    def get_observations(self):
        return self.get_nav_observation(self.goal_xy)

    def step(self, base_action=None):
        prev_xy = np.array([self.x, self.y], dtype=np.float32)
        observations, reward, done, info = super().step(base_action)
        self.print_nav_stats(observations)
        curr_xy = np.array([self.x, self.y], dtype=np.float32)
        self.episode_distance += np.linalg.norm(curr_xy - prev_xy)
        self.num_actions += 1
        self.num_collisions += self.collided
        return observations, reward, done, info
