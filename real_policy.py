import time

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from habitat_baselines.rl.ddppo.policy.resnet_policy import \
    PointNavResNetPolicy
from habitat_baselines.rl.ddppo.policy.splitnet_policy import \
    PointNavSplitNetPolicy
from habitat_baselines.utils.common import batch_obs


# Turn numpy observations into torch tensors for consumption by policy
def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


class RealPolicy:
    def __init__(
        self, checkpoint_path, observation_space, action_space, device, policy_name
    ):
        self.device = device
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print("using checkpoint: ", checkpoint_path)
        config = checkpoint["config"]
        if "num_cnns" not in config.RL.POLICY:
            config.RL.POLICY["num_cnns"] = 1
        """ Disable observation transforms for real world experiments """
        config.defrost()
        config.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = []
        config.freeze()

        self.policy = eval(policy_name).from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        print("Actor-critic architecture:", self.policy)
        # Move it to the device
        self.policy.to(self.device)
        # Load trained weights into the policy

        # If using Splitnet policy, filter out decoder stuff, as it's not used at test-time
        self.policy.load_state_dict(
            {k[len("actor_critic.") :]: v for k, v in checkpoint["state_dict"].items()},
            strict=False,
        )

        self.prev_actions = None
        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.config = config
        self.num_actions = action_space.shape[0]
        self.reset_ran = False

    def reset(self):
        self.reset_ran = True
        self.test_recurrent_hidden_states = torch.zeros(
            1,  # The number of environments. Just one for real world.
            self.policy.net.num_recurrent_layers,
            self.config.RL.PPO.hidden_size,
            device=self.device,
        )

        # We start an episode with 'done' being True (0 for 'not_done')
        self.not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.prev_actions = torch.zeros(1, self.num_actions, device=self.device)

    def act(self, observations):
        assert self.reset_ran, "You need to call .reset() on the policy first."
        batch = batch_obs([observations], device=self.device)
        with torch.no_grad():
            start_time = time.time()
            _, actions, _, self.test_recurrent_hidden_states = self.policy.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=True,
            )
            inf_time = time.time() - start_time
        self.prev_actions.copy_(actions)
        self.not_done_masks = torch.ones(1, 1, dtype=torch.bool, device=self.device)

        # GPU/CPU torch tensor -> numpy
        actions = actions.squeeze().cpu().numpy()

        return actions


class NavPolicy(RealPolicy):
    def __init__(self, checkpoint_path, device, sensor_type, policy_name):
        if sensor_type == "depth":
            obs_right_key = "spot_right_depth"
            obs_left_key = "spot_left_depth"
        elif sensor_type == "gray":
            obs_right_key = "spot_right_gray"
            obs_left_key = "spot_left_gray"
        observation_space = SpaceDict(
            {
                obs_left_key: spaces.Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                obs_right_key: spaces.Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "pointgoal_with_gps_compass": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        # Linear, angular, and horizontal velocity (in that order)
        action_space = spaces.Box(-1.0, 1.0, (3,))
        super().__init__(
            checkpoint_path, observation_space, action_space, device, policy_name
        )


if __name__ == "__main__":
    nav_policy = NavPolicy(
        "weights/spot_cam_kinematic_hm3d_gibson_ckpt_27.pth",
        device="cpu",
    )
    nav_policy.reset()
    observations = {
        "spot_left_depth": np.zeros([256, 128, 1], dtype=np.float32),
        "spot_right_depth": np.zeros([256, 128, 1], dtype=np.float32),
        "pointgoal_with_gps_compass": np.zeros(2, dtype=np.float32),
    }
    actions = nav_policy.act(observations)
    print("actions:", actions)
