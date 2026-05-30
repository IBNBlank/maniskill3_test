#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-05-29
################################################################

import wandb
from typing import Optional
from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(
        self,
        args,
        wandb_dict: Optional[dict] = None,
    ) -> None:
        self.log_wandb = False
        if wandb_dict is not None:
            try:
                env_kwargs = wandb_dict["env_kwargs"]
                max_episode_steps = wandb_dict["max_episode_steps"]

                config = vars(args)
                config["env_cfg"] = dict(**env_kwargs,
                                         num_envs=args.num_envs,
                                         env_id=args.env_id,
                                         reward_mode="normalized_dense",
                                         env_horizon=max_episode_steps,
                                         partial_reset=args.partial_reset)
                config["eval_env_cfg"] = dict(**env_kwargs,
                                              num_envs=args.num_eval_envs,
                                              env_id=args.env_id,
                                              reward_mode="normalized_dense",
                                              env_horizon=max_episode_steps,
                                              partial_reset=args.partial_reset)
                wandb.init(project=args.wandb_project_name,
                           entity=args.wandb_entity,
                           sync_tensorboard=False,
                           config=config,
                           name=args.run_name,
                           save_code=True,
                           group=args.wandb_group,
                           tags=["ppo", "walltime_efficient"])
                self.log_wandb = True
            except:
                print("Failed to initialize wandb")

        self.writer = SummaryWriter(f"runs/{args.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join(
                [f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.close()
