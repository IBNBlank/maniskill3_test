#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-05-25
################################################################

# Import required packages
import gymnasium as gym
import numpy as np
import mani_skill.envs
import matplotlib.pyplot as plt

# env_id = "PickCube-v1"
# env_id = "PegInsertionSide-v1"
env_id = "StackCube-v1"

# obs_mode = "rgb+depth+segmentation"
obs_mode = "pointcloud"
# obs_mode = "state_dict"
# obs_mode = "state"

# control_mode = "pd_joint_pos"
# control_mode = "pd_joint_delta_pos"
# control_mode = "pd_joint_target_delta_pos"
# control_mode = "pd_ee_delta_pos"
control_mode = "pd_ee_delta_pose"
# control_mode = "pd_ee_delta_pose_align"
# control_mode = "pd_ee_target_delta_pos"
# control_mode = "pd_ee_target_delta_pose"
# control_mode = "pd_joint_pos_vel"
# control_mode = "pd_joint_delta_pos_vel"
# control_mode = "pd_joint_delta_pos_stiff_body"

reward_mode = "sparse"
# reward_mode = "dense"

robot_uids = "panda"
# robot_uids = "fetch"

env = gym.make(
    env_id,
    obs_mode=obs_mode,
    control_mode=control_mode,
    reward_mode=reward_mode,
    robot_uids=robot_uids,
    num_envs=4,
    enable_shadow=True,
)
obs, _ = env.reset()
print("Action Space:", env.action_space)

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
rgbs = env.render_rgb_array(
)  # this is a easy way to get the rgb array without having to set render_mode
for i, ax in enumerate(axs.flatten()):
    ax.imshow(rgbs[i].cpu().numpy())
    ax.axis("off")
plt.suptitle("Current States viewed from external cameras")
fig.tight_layout()
plt.savefig("03_rgb.png")
plt.close()

print(f"obs mode: {obs_mode}")
print(f"obs keys: {obs.keys()}")

if obs_mode == "rgb+depth+segmentation":

    def show_camera_view(obs_camera, title, env_id=0):
        fig = plt.figure()
        rgb, depth = obs_camera['rgb'], obs_camera['depth']
        plt.subplot(1, 3, 1)
        plt.title(f"{title} - RGB")
        plt.imshow(rgb[env_id].cpu().numpy())
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.title(f"{title} - Depth")
        plt.imshow(depth[..., 0][env_id].cpu().numpy(), cmap="gray")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.title(f"{title} - Segmentation")
        plt.imshow(obs_camera["segmentation"][..., 0][env_id].cpu().numpy())
        plt.axis("off")
        fig.tight_layout()
        plt.savefig("03_multi_images.png")
        plt.close(fig)

    show_camera_view(obs['sensor_data']['base_camera'], "Base")

elif obs_mode == "pointcloud":

    def show_pointcloud(obs, env_id=0):
        import trimesh
        v = obs['pointcloud']['xyzw'][env_id, ..., :3].cpu().numpy()
        cam2world = obs["sensor_param"]["base_camera"]["cam2world_gl"][
            env_id].cpu().numpy()
        cam2world = cam2world
        camera = trimesh.scene.Camera("camera", (1024, 1024),
                                      fov=(np.rad2deg(np.pi / 2),
                                           np.rad2deg(np.pi / 2)))
        s = trimesh.Scene([
            trimesh.points.PointCloud(
                v, obs['pointcloud']['rgb'][env_id].cpu().numpy())
        ],
                          camera=camera,
                          camera_transform=cam2world)
        return s.show()

    show_pointcloud(obs)

env.close()
