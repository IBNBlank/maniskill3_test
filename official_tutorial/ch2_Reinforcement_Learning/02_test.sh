#!/usr/bin/env bash
set -Eeuo pipefail
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-05-25
################################################################

python 02_visual_based.py --env_id="PickCube-v1" \
  --evaluate \
  --checkpoint=/home/blank/GitSource/maniskill3_test/official_tutorial/ch2_Reinforcement_Learning/runs/PickCube-v1__02_visual_based__1__1780136931/final_ckpt.pt \
  --num_eval_envs=1 \
  --num-eval-steps=1000