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
import sapien
import mani_skill.envs
from tqdm import tqdm
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.agents.registration import REGISTERED_AGENTS

print(REGISTERED_AGENTS.keys())

env = gym.make(
    "Empty-v1",
    obs_mode="none",
    reward_mode="none",
    enable_shadow=True,
    control_mode=
    "pd_joint_pos",  # allows us to easily hold joints mostly in place
    robot_uids="anymal_c",
    render_mode="rgb_array",
)
env = RecordEpisode(env, "./videos", save_trajectory=False)
print(env.agent.keyframes.keys())

env.reset()
kf = env.agent.keyframes["standing"]
env.agent.robot.set_pose(sapien.Pose([0, 0, 1]))
env.agent.robot.set_qpos(kf.qpos)

# if running on a GPU environment you have to
# apply the changes to environment state
if env.gpu_sim_enabled:
    env.scene._gpu_apply_all()  # applies changes
    env.scene.px.gpu_update_articulation_kinematics(
    )  # updates robot link poses necessary to render correctly
    env.scene._gpu_fetch_all()  # updates GPU buffers
env.render()  # call this to update the rendering
for i in tqdm(range(100)):
    # with pd_joint_pos control we try to get the robot to maintain the same joint
    # positions as the keyframe
    obs, reward, terminated, truncated, info = env.step(kf.qpos)
env.flush_video("example")
