import time

import gym
import numpy as np
from spot_ros_node import SpotRosSubscriber
from spot_wrapper.spot import Spot, wrap_heading

CTRL_HZ = 1
MAX_EPISODE_STEPS = 200

# Base action params
MAX_LIN_VEL = 0.5  # m/s
MAX_ANG_VEL = 0.52  # 30.0 degrees/s, in radians
# MAX_ANG_VEL = 0.7  # 30.0 degrees/s, in radians
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
            x_vel, ang_vel, y_vel = base_action

            x_vel = np.clip(x_vel, -1, 1)
            ang_vel = np.clip(ang_vel, -1, 1)
            y_vel = np.clip(y_vel, -1, 1)

            x_vel = (x_vel + 1.0) / 2.0
            ang_vel = (ang_vel + 1.0) / 2.0
            y_vel = (y_vel + 1.0) / 2.0

            # Scale actions
            x_vel = -self.max_lin_vel + x_vel * 2 * self.max_lin_vel
            ang_vel = -self.max_ang_vel + ang_vel * 2 * self.max_ang_vel
            y_vel = -self.max_lin_vel + y_vel * 2 * self.max_lin_vel

            # x_vel = np.clip(x_vel, -1, 1) * self.max_lin_vel
            # y_vel = np.clip(y_vel, -1, 1) * self.max_lin_vel
            # ang_vel = np.clip(ang_vel, -1, 1) * self.max_ang_vel
            # Spot-real's horizontal velocity is flipped from Habitat's convention
            print(f"STEPPING! Vx: {x_vel}, Vy: {y_vel}, Vt: {ang_vel}")
            self.spot.set_base_velocity(x_vel, y_vel, ang_vel, self.vel_time)

            # key = input("Press key to continue\n")
            # if key == "q":
            #     return
            # else:
            #     self.spot.set_base_velocity(x_vel, -y_vel, ang_vel, self.vel_time)

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

    def get_observations(self):
        raise NotImplementedError

    def get_success(self, observations):
        raise NotImplementedError

    def should_end(self):
        return False
