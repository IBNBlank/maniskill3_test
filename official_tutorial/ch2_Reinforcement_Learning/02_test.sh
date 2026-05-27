#!/usr/bin/env bash
set -Eeuo pipefail
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-05-25
################################################################

python 02_visual_based.py --env_id="PushCube-v1" \
  --evaluate --checkpoint=runs/rgb-pushcube/ckpt_41.pt \
  --num_eval_envs=1 --num-eval-steps=100