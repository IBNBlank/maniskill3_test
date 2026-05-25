#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-05-24
################################################################

import time
import cv2
import torch
import gymnasium as gym
import mani_skill.envs

num_envs = 512  # you can go up to 4096 on better GPUs
env = gym.make("PickCube-v1", num_envs=num_envs, obs_mode="rgbd")
env.unwrapped.print_sim_details()
obs, _ = env.reset(seed=0)
done = False
start_time = time.time()
total_rew = 0
while not done:
    # note that env.action_space is now a batched action space
    obs, rew, terminated, truncated, info = env.step(
        torch.from_numpy(env.action_space.sample()))
    done = (terminated | truncated).any()
N = num_envs * info["elapsed_steps"][0].item()
dt = time.time() - start_time
FPS = N / (dt)
print(f"Frames Per Second = {N} / {dt} = {FPS}")

print(obs.keys())
print(obs['sensor_data'].keys())
print(obs['sensor_data']['base_camera'].keys())
print(obs['sensor_data']['base_camera']['rgb'].shape)
cv2.imshow("RGB",
           obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()[..., ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
