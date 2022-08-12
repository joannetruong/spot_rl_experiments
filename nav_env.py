import cv2
import numpy as np
from base_env import SpotBaseEnv
from spot_wrapper.spot import Spot, wrap_heading


class SpotNavEnv(SpotBaseEnv):
    def __init__(self, spot: Spot, cfg):
        super().__init__(spot, cfg)
        self.goal_xy = None
        self._log_goal = cfg.log_goal
        self.succ_distance = cfg.success_dist
        if self._log_goal:
            self.succ_distance = np.log(self.succ_distance)
        self._project_goal = cfg.project_goal
        self.use_horizontal_vel = cfg.use_horizontal_velocity

    def reset(self, goal_xy, yaw=None):
        self.spot.home_robot(yaw)
        print("Reset! Curr pose: ", self.spot.get_xy_yaw())
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
        if self._log_goal:
            rho = np.exp(rho)
        print(
            f"Dist to goal: {rho:.2f}\t"
            f"theta: {np.rad2deg(theta):.2f}\t"
            f"x: {self.x:.2f}\t"
            f"y: {self.y:.2f}\t"
            f"yaw: {np.rad2deg(self.yaw):.2f}\t"
            f"# actions: {self.num_actions}\t"
            f"# collisions: {self.num_collisions}\t"
        )

    def _compute_pointgoal(self, goal_xy):
                # Get rho theta observation
        self.x, self.y, self.yaw = self.spot.get_xy_yaw()
        curr_xy = np.array([self.x, self.y], dtype=np.float32)
        rho = np.linalg.norm(curr_xy - goal_xy)
        theta = np.arctan2(goal_xy[1] - self.y, goal_xy[0] - self.x) - self.yaw
        if self._project_goal != -1:
            try:
                slope = (goal_xy[1] - self.y) / (goal_xy[0] - self.x)
                print('self.x: ', self.x, self.y, 'goal: ', goal_xy, 'slope: ', slope)
                proj_goal_x = self._project_goal + self.x
                proj_goal_y = (self._project_goal * slope) + self.y
                proj_goal_xy = np.array([proj_goal_x, proj_goal_y])
                proj_rho = np.linalg.norm(curr_xy - proj_goal_xy)
                print("proj_rho: ", proj_rho, "proj_xy: ", proj_goal_xy, " rho: ", rho, "goal_xy: ", goal_xy)
                if proj_rho < rho:
                    goal_xy = proj_goal_xy
                    rho = proj_rho
            except:
                pass
        theta = np.arctan2(goal_xy[1] - self.y, goal_xy[0] - self.x) - self.yaw
        if self._log_goal:
            rho_theta = np.array([np.log(rho), wrap_heading(theta)], dtype=np.float32)
        else:
            rho_theta = np.array([rho, wrap_heading(theta)], dtype=np.float32)


    def get_nav_observation(self, goal_xy):
        observations = {}
        if self.sensor_type == "depth":
            img_obs = self.front_depth_img
            obs_right_key = "spot_right_depth"
            obs_left_key = "spot_left_depth"
        elif self.sensor_type == "gray":
            img_obs = self.front_gray_img
            obs_right_key = "spot_right_gray"
            obs_left_key = "spot_left_gray"
        # Get visual observations
        front_obs = np.float32(img_obs) / 255.0
        # Add dimension for channel (unsqueeze)

        front_obs = front_obs.reshape(*front_obs.shape[:2], 1)
        observations[obs_right_key], observations[obs_left_key] = np.split(
            front_obs, 2, 1
        )

        rho_theta = self._compute_pointgoal(goal_xy)

        # Get goal heading observation
        observations["pointgoal_with_gps_compass"] = rho_theta

        return observations

    def get_success(self, observations):
        succ = self.get_nav_success(observations, self.succ_distance)
        if succ:
            print("SUCCESS!")
            self.spot.set_base_velocity(0.0, 0.0, 0.0, self.vel_time)
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

class SpotContextNavEnv(SpotNavEnv):
    def __init__(self, spot: Spot, cfg):
        super().__init__(spot, cfg)
        self.wpt_xy = None

    def get_context_observations(self, waypoint_xy):
        observations = {}
        rho_theta = self._compute_pointgoal(waypoint_xy)
        observations["context_waypoint"] = rho_theta

    def get_observations(self):
        nav_obs = self.get_nav_observation(self.goal_xy)
        context_obs = self.get_context_observations(self.wpt_xy)
        return nav_obs + context_obs