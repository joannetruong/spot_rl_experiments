import os.path
import time

import torch
from habitat.config import Config
import sys

if __name__ == "__main__":
    weights = sys.argv[1]
    checkpoint = torch.load(weights, map_location="cpu")

    orig_ckpt = checkpoint
    orig_ckpt["config"].defrost()
    orig_ckpt["config"]["RL"]["POLICY"]["ACTION_DIST"] = Config()
    orig_ckpt["config"]["RL"]["POLICY"]["ACTION_DIST"]["use_log_std"] = False
    orig_ckpt["config"]["RL"]["POLICY"]["ACTION_DIST"]["clamp_std"] = True
    orig_ckpt["config"]["RL"]["POLICY"]["ACTION_DIST"]["min_std"] = 1e-6
    orig_ckpt["config"]["RL"]["POLICY"]["ACTION_DIST"]["max_std"] = 1
    orig_ckpt["config"]["RL"]["POLICY"]["ACTION_DIST"]["min_log_std"] = -5
    orig_ckpt["config"]["RL"]["POLICY"]["ACTION_DIST"]["max_log_std"] = 2
    orig_ckpt["config"].freeze()
    checkpoint = {
        "state_dict": orig_ckpt["state_dict"],
        "config": orig_ckpt["config"],
    }

    torch.save(
        checkpoint, cfg.weights + "_v2"
    )  