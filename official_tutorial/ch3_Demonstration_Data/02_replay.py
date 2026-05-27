#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-05-25
################################################################

from mani_skill.trajectory.utils import dict_to_list_of_dicts
from mani_skill.utils.visualization.misc import images_to_video
import gymnasium as gym
import mani_skill.envs
from tqdm import tqdm

import h5py
from mani_skill.utils.io_utils import load_json

# Load the trajectory data from the .h5 file. Demonstrations are versioned and thus saved to "demos/<version>/..."
traj_path = f"demos/PegInsertionSide-v1/motionplanning/trajectory.h5"
# You can also replace the above path with the trajectory you just recorded (./tmp/trajectory.h5)
h5_file = h5py.File(traj_path, "r")

# Load associated json
json_path = traj_path.replace(".h5", ".json")
json_data = load_json(json_path)

episodes = json_data["episodes"]  # meta data of each episode
env_info = json_data["env_info"]
env_id = env_info["env_id"]
env_kwargs = env_info["env_kwargs"]

print("env_id:", env_id)
print("env_kwargs:", env_kwargs)
print("#episodes:", len(episodes))
print("Dataset source:", json_data["source_type"])
print("Dataset source description:", json_data["source_desc"])


def replay(episode_idx, h5_file, json_data, render_mode="cameras", fps=20):
    episodes = json_data["episodes"]
    ep = episodes[episode_idx]
    # episode_id should be the same as episode_idx, unless specified otherwise
    episode_id = ep["episode_id"]
    traj = h5_file[f"traj_{episode_id}"]
    env_states = dict_to_list_of_dicts(traj["env_states"])

    # Create the environment
    env_kwargs = json_data["env_info"]["env_kwargs"]
    env = gym.make(json_data["env_info"]["env_id"], **env_kwargs)
    print(env_kwargs)
    # Reset the environment
    reset_kwargs = ep["reset_kwargs"].copy()
    reset_kwargs["seed"] = ep["episode_seed"]
    env.reset(**reset_kwargs)

    frames = [env.render_rgb_array()[0].numpy()]
    for i in tqdm(range(len(traj["actions"]))):
        action = traj["actions"][i]
        obs, reward, terminated, truncated, info = env.step(action)
        env.set_state_dict(env_states[i])
        frames.append(env.render_rgb_array()[0].numpy())

    env.close()
    del env
    images_to_video(
        frames,
        output_dir=".",
        video_name="replay",
        fps=30,
    )


episode_idx = 4  #@param {type:"integer"}
replay(episode_idx, h5_file, json_data)
