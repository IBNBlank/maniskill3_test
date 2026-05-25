#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-05-24
################################################################

import time
import cv2
import gymnasium as gym
import mani_skill.envs

env = gym.make("PickCube-v1", render_mode="rgb_array")
obs, _ = env.reset(seed=0)
cv2.imshow("RGB", env.render()[0].numpy()[..., ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()

env.unwrapped.print_sim_details()
done = False
start_time = time.time()
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
N = info["elapsed_steps"].item()
dt = time.time() - start_time
FPS = N / (dt)
print(f"Frames Per Second = {N} / {dt} = {FPS}")
