#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-05-27
################################################################

# Import required packages
import gymnasium as gym
import torch
import mani_skill.envs
from tqdm import tqdm
from mani_skill.utils.wrappers import RecordEpisode

# See section 1.5 for more details on how we create environments and save videos
env = gym.make(
    "PickClutterYCB-v1",
    num_envs=4,
    render_mode="rgb_array",
    enable_shadow=True,
)
env = RecordEpisode(
    env,
    "./videos",
    max_steps_per_video=100,
    save_trajectory=False,
)

# step through the environment with random actions
obs, _ = env.reset(seed=0)
for i in tqdm(range(100)):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(
        torch.from_numpy(action))
env.close()
