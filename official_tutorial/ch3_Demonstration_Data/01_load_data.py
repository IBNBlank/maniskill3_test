#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-05-25
################################################################

from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset

dataset = ManiSkillTrajectoryDataset(
    dataset_file="demos/PegInsertionSide-v1/motionplanning/trajectory.h5")
data = dataset[150]
for k, v in data.items():
    print(k, v)
