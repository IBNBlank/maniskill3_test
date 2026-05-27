#!/usr/bin/env bash
set -Eeuo pipefail
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-05-25
################################################################

python 02_visual_based.py --env_id="PushCube-v1" --exp-name="rgb-pushcube" \
  --num_envs=256 --update_epochs=8 --num_minibatches=16 \
  --total_timesteps=250_000 --eval_freq=10 --num-steps=20