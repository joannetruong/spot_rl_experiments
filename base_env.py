import time

import gym
import numpy as np
from spot_ros_node import SpotRosSubscriber
from spot_wrapper.spot import Spot, wrap_heading


def rescale_actions(actions, action_thresh=0.05, silence_only=False):
    actions = np.clip(actions, -1, 1)
    # Silence low actions
    actions[np.abs(actions) < action_thresh] = 0.0
    return actions


class SpotBaseEnv(SpotRosSubscriber, gym.Env):
    def __init__(self, spot: Spot, cfg):
        super().__init__("spot_reality_gym")
        self.spot = spot

        # General environment parameters
        self.ctrl_hz = cfg.ctrl_hz
        self.max_episode_steps = cfg.max_episode_steps
        self.should_end = False
        self.num_steps = 0
        self.reset_ran = False

        # Robot state parameters
        self.x, self.y, self.yaw = None, None, None

        # Base action parameters
        self.max_lin_dist = cfg.max_lin_dist
        self.max_ang_dist = cfg.max_ang_dist
        self.vel_time = 1 / self.ctrl_hz

        self.use_horizontal_velocity = cfg.use_horizontal_velocity

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

    def step(self, base_action):
        """Moves the arm and returns updated observations

        :param base_action: np.array of velocities (lineaer, angular)
        :return:
        """
        assert self.reset_ran, ".reset() must be called first!"
        # Command velocities using the input action
        if not self.use_horizontal_velocity:
            base_action = np.array([base_action[0], 0.0, base_action[1]])
        base_action = rescale_actions(base_action, silence_only=True)
        base_action *= [self.max_lin_dist, self.max_lin_dist, self.max_ang_dist]

        base_vel = base_action * self.ctrl_hz

        print(
            f"STEPPING! Vx: {base_vel[0]}, Vy: {base_vel[1]}, Vt: {np.rad2deg(base_vel[2])}"
        )
        start_time = time.time()
        self.spot.set_base_velocity(*base_vel, 1 / self.ctrl_hz)
        # Pause until enough time has passed during this step
        while time.time() < start_time + 1 / self.ctrl_hz:
            pass
        observations = self.get_observations()
        self.num_steps += 1
        timeout = self.num_steps == self.max_episode_steps
        done = timeout or self.get_success(observations) or self.should_end

        # Don't need reward or info
        reward = None
        info = {}

        return observations, reward, done, info

    def get_observations(self):
        raise NotImplementedError

    def get_success(self, observations):
        raise NotImplementedError

    def should_end(self):
        return False
