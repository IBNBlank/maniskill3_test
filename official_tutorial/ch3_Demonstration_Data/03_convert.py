#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-05-25
################################################################

from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset
import matplotlib.pyplot as plt

dataset = ManiSkillTrajectoryDataset(
    dataset_file=
    "demos/PegInsertionSide-v1/motionplanning/trajectory.rgbd.pd_joint_delta_pos.physx_cpu.h5"
)
data = dataset[0]
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(data["obs"]["sensor_data"]["hand_camera"]["rgb"])
axs[1].imshow(data["obs"]["sensor_data"]["hand_camera"]["depth"])
plt.savefig("03_convert.png")
plt.close()
