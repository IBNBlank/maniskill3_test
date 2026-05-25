#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-05-25
################################################################

# Import required packages
import gymnasium as gym
import torch
import mani_skill.envs
from tqdm import tqdm
from mani_skill.utils.wrappers import RecordEpisode
# to make it look a little more realistic, we will enable shadows which make the default lighting cast shadows
env = gym.make("PickCube-v1",
               num_envs=4,
               render_mode="rgb_array",
               enable_shadow=True)
env = RecordEpisode(
    env,
    "./videos",  # the directory to save replay videos and trajectories to
    # on GPU sim we record intervals, not by single episodes as there are multiple envs
    # each 100 steps a new video is saved
    max_steps_per_video=100)

# step through the environment with random actions
obs, _ = env.reset()
for i in tqdm(range(300)):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(
        torch.from_numpy(action))
    # env.render() # will render with a window if possible
env.close()
