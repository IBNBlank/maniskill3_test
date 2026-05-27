#!/usr/bin/env bash
set -Eeuo pipefail
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-05-27
################################################################

python -m mani_skill.examples.motionplanning.panda.run -e "PegInsertionSide-v1" \
  -n=1 --save-video --record-dir="demos" --traj-name="peginsertionside" --only-count-success