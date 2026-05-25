#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-05-23
################################################################

import gymnasium as gym
# from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
import mani_skill.envs

env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    obs_mode="state", # there is also "state_dict", "rgbd", ...
    control_mode="pd_joint_delta_pos", # there is also "pd_joint_delta_pos", ...
    num_envs=16,
    parallel_in_single_scene=True,
    viewer_camera_configs=dict(shader_pack="rt-fast"),
    render_mode="human"
)
# env = CPUGymWrapper(env)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0) # reset with a seed for determinism
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"reward: {reward}, terminated: {terminated}, truncated: {truncated}")
    # done = terminated or truncated
    env.render()  # a display is required to render
env.close()