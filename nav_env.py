import os

import cv2
import numpy as np
import scipy.ndimage
from base_env import SpotBaseEnv
from PIL import Image
from skimage.draw import disk
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
                print("self.x: ", self.x, self.y, "goal: ", goal_xy, "slope: ", slope)
                proj_goal_x = self._project_goal + self.x
                proj_goal_y = (self._project_goal * slope) + self.y
                proj_goal_xy = np.array([proj_goal_x, proj_goal_y])
                proj_rho = np.linalg.norm(curr_xy - proj_goal_xy)
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

        return rho_theta

    def get_nav_observation(self, goal_xy):
        observations = {}
        if self.sensor_type == "depth":
            img_obs = self.front_depth_img
            raw_img_obs = self.front_raw_depth_img
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

        # Get raw visual observations
        front_raw_obs = np.float32(raw_img_obs) / 255.0
        # Add dimension for channel (unsqueeze)

        front_raw_obs = front_obs.reshape  # Get raw visual observations
        front_raw_obs = np.float32(raw_img_obs) / 255.0
        # Add dimension for channel (unsqueeze)

        front_raw_obs = front_obs.reshape(*front_raw_obs.shape[:2], 1)
        (
            observations[obs_right_key + "_raw"],
            observations[obs_left_key + "_raw"],
        ) = np.split(front_raw_obs, 2, 1)

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
        self.context_key = f"context_{cfg.context_type}"
        if self.context_key == "context_map":
            self.map_resolution = cfg.map_resolution
            self.meters_per_pixel = cfg.meters_per_pixel
            self.use_agent_map = cfg.use_agent_map
            dim = 2 if self.use_agent_map else 1
            self.map_shape = (self.map_resolution, self.map_resolution, dim)
            # self.disk_radius = 1.5/self.meters_per_pixel
            self.disk_radius = 2
            self.map_path = os.path.join("config", cfg.map)
            self.goal_map_path = os.path.join("config", cfg.map_goal)
            assert os.path.exists(self.map_path), (
                f"{self.map_path} does not exist. "
                "Make sure the map is pushed to the robot."
            )
            assert os.path.exists(self.goal_map_path), (
                f"{self.goal_map_path} does not exist. "
                "Make sure the map is pushed to the robot."
            )

    def calculate_map_scale(self):
        goal_dist, _ = self._compute_pointgoal(self.goal_xy)
        if self._log_goal:
            goal_dist = np.exp(goal_dist)
        print("goal dist: ", self.goal_xy, goal_dist, self.meters_per_pixel)
        desired_pix = goal_dist / self.meters_per_pixel
        goal_map = cv2.imread(self.goal_map_path)
        img_hsv = cv2.cvtColor(goal_map, cv2.COLOR_BGR2HSV)
        # lower mask (0-10)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        red_mask = cv2.inRange(img_hsv, lower_red, upper_red)
        red_pixels = np.where(red_mask == 255)

        lower_green = np.array([40, 50, 50])
        upper_green = np.array([70, 255, 255])
        green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
        green_pixels = np.where(green_mask == 255)

        current_pix = np.linalg.norm(
            np.array(red_pixels)[:, 0] - np.array(green_pixels)[:, 0], axis=0
        )

        self.goal_pix_y, self.goal_pix_x = np.array(red_pixels)[:, 0]
        scale_factor = current_pix / desired_pix
        base_map_x, base_map_y = goal_map.shape[:2]
        return int(base_map_x / scale_factor), int(base_map_y / scale_factor)

    def load_context_map(self):
        base_map = cv2.imread(self.map_path)
        ## resize map to match meters per pixel
        x_scale, y_scale = self.calculate_map_scale()
        occupancy_map = cv2.resize(base_map[:, :, 0], (x_scale, y_scale))

        occupancy_map = np.expand_dims(occupancy_map, axis=2)
        # normalize image to 0 to 1
        occupancy_map = occupancy_map.astype(np.float32)
        occupancy_map /= np.max(occupancy_map)

        h, w = occupancy_map.shape[:2]
        center_coord_x, center_coord_y = (
            occupancy_map.shape[0] // 2,
            occupancy_map.shape[1] // 2,
        )

        # rotating large images takes a long time,
        # so we take a local crop of max 500x500 first
        # top = max(int(center_coord_x - 250), 0)
        # bottom = min(int(center_coord_x + 250), h)
        # left = max(int(center_coord_y - 250), 0)
        # right = min(int(center_coord_y + 250), w)

        # occupancy_map = occupancy_map[top:bottom, left:right, :]

        # pose_goal_map = np.zeros_like(occupancy_map)

        # pose_goal_map = self.get_agent_map(base_map, self.goal_pix_x, self.goal_pix_y)
        # pose_goal_map_rz = cv2.resize(pose_goal_map, (x_scale, y_scale))
        # pose_goal_map_rz = np.expand_dims(pose_goal_map_rz, axis=2)

        pose_goal_map_rz = np.zeros_like(occupancy_map)
        orig_h, orig_w = base_map.shape[:2]
        goal_pix_x_rz, goal_pix_y_rz = int(self.goal_pix_x * x_scale / orig_w), int(
            self.goal_pix_y * y_scale / orig_h
        )
        rr, cc = disk((goal_pix_x_rz, goal_pix_y_rz), self.disk_radius)
        pose_goal_map_rz[cc, rr, 0] = 1.0
        cv2.imwrite("debug.png", pose_goal_map_rz * 255)

        if self.use_agent_map:
            self.orig_map = np.concatenate([occupancy_map, pose_goal_map_rz], axis=2)
        else:
            self.orig_map = occupancy_map

        context_map = self.crop_and_fill_map(
            self.orig_map, (center_coord_x, center_coord_y)
        )

        if self.use_agent_map:
            self.context_map = self.draw_curr_goal(context_map)

    def crop_at_point(self, context_map, center_coord):
        h, w = context_map.shape[:2]
        a_x, a_y = center_coord
        top = max(int(a_x - self.map_resolution // 2), 0)
        bottom = min(int(a_x + self.map_resolution // 2), h)
        left = max(int(a_y - self.map_resolution // 2), 0)
        right = min(int(a_y + self.map_resolution // 2), w)

        if context_map.ndim == 3:
            return context_map[top:bottom, left:right, :]
        else:
            return context_map[top:bottom, left:right]

    def crop_and_fill_map(self, context_map, center_coord):
        cropped_map = self.crop_at_point(context_map, center_coord)
        lh, lw = cropped_map.shape[:2]

        pad_top = max(int(self.map_resolution // 2 - center_coord[0] - 1), 0)
        pad_left = max(int(self.map_resolution // 2 - center_coord[1] - 1), 0)
        base_map = np.zeros(self.map_shape)
        # base_map = np.ones(self.map_shape)
        # if self.use_agent_map:
        #     base_map[:, :, 1] *= 0.0

        base_map[pad_top : pad_top + lh, pad_left : pad_left + lw, :] = cropped_map[
            :, :, :
        ]
        return base_map

    def draw_curr_goal(self, context_map):
        # draw current position (center)
        center_coord = context_map.shape[0] // 2
        rr, cc = disk((center_coord, center_coord), self.disk_radius)
        context_map[rr, cc, 1] = 1.0

        # rho, theta = self._compute_pointgoal(self.goal_xy)
        # if self._log_goal:
        #     rho = np.exp(rho)
        # r_limit = (self.map_resolution // 2) * self.meters_per_pixel
        # goal_r = np.clip(rho, -r_limit, r_limit)
        #
        # x = (goal_r / self.meters_per_pixel) * np.cos(theta)
        # y = (goal_r / self.meters_per_pixel) * np.sin(theta)
        #
        # mid = self.map_resolution // 2
        # row, col = np.clip(
        #     int(mid - x),
        #     0 + self.disk_radius,
        #     self.map_resolution - (self.disk_radius + 1),
        # ), np.clip(
        #     int(mid - y),
        #     0 + self.disk_radius,
        #     self.map_resolution - (self.disk_radius + 1),
        # )
        #
        # rr, cc = disk((row, col), self.disk_radius)
        # context_map[rr, cc, 1] = 1.0

        return context_map

    def get_rotated_point(self, img, im_rot, xy, agent_rotation):
        yx = xy[::-1]
        a = -(agent_rotation - np.pi)
        org_center = (np.array(img.shape[:2][::-1]) - 1) // 2
        rot_center = (np.array(im_rot.shape[:2][::-1]) - 1) // 2
        org = yx - org_center
        new = np.array(
            [
                org[0] * np.cos(a) + org[1] * np.sin(a),
                -org[0] * np.sin(a) + org[1] * np.cos(a),
            ]
        )
        rotated_pt = new + rot_center
        return int(rotated_pt[1]), int(rotated_pt[0])

    def reset(self, goal_xy, wpt_xy=[0, 0], yaw=None):
        print("CONTEXT RESET")
        self.spot.home_robot(yaw)
        self.goal_xy = np.array(goal_xy, dtype=np.float32)
        self.wpt_xy = np.array(wpt_xy, dtype=np.float32)
        self.num_actions = 0
        self.num_collisions = 0
        self.episode_distance = 0
        if self.context_key == "context_map":
            self.load_context_map()
        observations = super().reset(goal_xy, yaw)
        assert len(self.goal_xy) == 2
        return observations

    def get_nav_success(self, observations, success_distance):
        # Is the agent at the goal?
        goal_key = (
            self.context_key
            if self.context_key == "context_waypoint"
            else "pointgoal_with_gps_compass"
        )
        dist_to_goal, _ = observations[goal_key]
        at_goal = dist_to_goal < success_distance
        return at_goal

    def print_nav_stats(self, observations):
        goal_rho, goal_theta = observations["pointgoal_with_gps_compass"]
        if self._log_goal:
            goal_rho = np.exp(goal_rho)
        if self.context_key == "context_waypoint":
            wpt_rho, wpt_theta = observations[self.context_key]
            if self._log_goal:
                wpt_rho = np.exp(wpt_rho)
            print(
                f"Dist to wpt: {wpt_rho:.2f}\t"
                f"wpt_theta: {np.rad2deg(wpt_theta):.2f}\t"
            )
        print(
            f"Dist to goal: {goal_rho:.2f}\t"
            f"goal_theta: {np.rad2deg(goal_theta):.2f}\t"
            f"x: {self.x:.2f}\t"
            f"y: {self.y:.2f}\t"
            f"yaw: {np.rad2deg(self.yaw):.2f}\t"
            f"# actions: {self.num_actions}\t"
            f"# collisions: {self.num_collisions}\t"
        )

    def pil_rotate(self, image, angle, expand=True, fillcolor="white"):
        print("image shape: ", image.shape)
        rotated_pil = Image.fromarray(image).rotate(
            angle, Image.NEAREST, expand=expand, fillcolor=fillcolor
        )
        return np.array(rotated_pil)

    def get_context_observations(self, waypoint_xy, expand=True, fillcolor="white"):
        observations = {}
        if self.context_key == "context_waypoint":
            rho_theta = self._compute_pointgoal(waypoint_xy)
            observations[self.context_key] = rho_theta
        else:
            self.x, self.y, self.yaw = self.spot.get_xy_yaw()
            x_diff = self.x / self.meters_per_pixel
            y_diff = self.y / self.meters_per_pixel
            center_coord_x, center_coord_y = (
                self.orig_map.shape[0] // 2,
                self.orig_map.shape[1] // 2,
            )

            rotated_map_0 = self.pil_rotate(
                np.array(self.orig_map[:, :, 0]),
                np.rad2deg(-self.yaw),
                expand=expand,
                fillcolor=fillcolor,
            )
            if self.use_agent_map:
                rotated_map_1 = self.pil_rotate(
                    np.array(self.orig_map[:, :, 1]),
                    np.rad2deg(-self.yaw),
                    expand=expand,
                    fillcolor=fillcolor,
                )
                rotated_map = np.stack([rotated_map_0, rotated_map_1], axis=2)
            else:
                rotated_map = np.expand_dims(rotated_map_0, axis=2)
            # center_coord_x, center_coord_y = rotated_map.shape[0] // 2, rotated_map.shape[1] // 2

            center_coord = [center_coord_x + x_diff, center_coord_y + y_diff]

            center_coord = self.get_rotated_point(
                self.orig_map, rotated_map, np.array(center_coord), self.yaw
            )

            context_map = self.crop_and_fill_map(rotated_map, center_coord)

            if self.use_agent_map:
                self.context_map = self.draw_curr_goal(context_map)
            else:
                self.context_map = context_map
            observations[self.context_key] = self.context_map.astype(np.float32)
        return observations

    def get_observations(self):
        nav_obs = self.get_nav_observation(self.goal_xy)
        context_obs = self.get_context_observations(self.wpt_xy)

        nav_obs.update(context_obs)
        return nav_obs
