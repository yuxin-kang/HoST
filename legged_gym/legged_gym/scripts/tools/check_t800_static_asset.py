#!/usr/bin/env python3
"""Static Isaac Gym asset load check for the T800 robot."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_LEGGED_GYM_DIR = Path(__file__).resolve().parents[3]
if str(REPO_LEGGED_GYM_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_LEGGED_GYM_DIR))

import isaacgym  # noqa: F401
from isaacgym import gymapi

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.t800.t800_config_ground import (
    T800Cfg,
    T800_CONTROLLED_JOINT_NAMES,
    T800_HEAD_JOINT_NAMES,
)


def _asset_options(args: argparse.Namespace) -> gymapi.AssetOptions:
    cfg = T800Cfg.asset
    options = gymapi.AssetOptions()
    options.default_dof_drive_mode = cfg.default_dof_drive_mode
    options.collapse_fixed_joints = cfg.collapse_fixed_joints
    options.replace_cylinder_with_capsule = cfg.replace_cylinder_with_capsule
    options.flip_visual_attachments = cfg.flip_visual_attachments
    options.fix_base_link = not args.free_base
    options.density = cfg.density
    options.angular_damping = cfg.angular_damping
    options.linear_damping = cfg.linear_damping
    options.max_angular_velocity = cfg.max_angular_velocity
    options.max_linear_velocity = cfg.max_linear_velocity
    options.armature = cfg.armature
    options.thickness = cfg.thickness
    options.disable_gravity = not args.gravity
    return options


def _set_default_dof_state(gym, env, actor, dof_names) -> None:
    states = np.zeros(len(dof_names), dtype=gymapi.DofState.dtype)
    for index, name in enumerate(dof_names):
        states["pos"][index] = T800Cfg.init_state.default_joint_angles[name]
    gym.set_actor_dof_states(env, actor, states, gymapi.STATE_ALL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true", help="Open a fixed-base viewer for visual inspection.")
    parser.add_argument("--free-base", action="store_true", help="Do not fix the base link during the static check.")
    parser.add_argument("--gravity", action="store_true", help="Enable gravity during the viewer check.")
    args = parser.parse_args()

    asset_path = T800Cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    asset_root = os.path.dirname(asset_path)
    asset_file = os.path.basename(asset_path)

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = False
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 4
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation")

    asset = gym.load_asset(sim, asset_root, asset_file, _asset_options(args))
    if asset is None:
        raise RuntimeError(f"Failed to load asset: {asset_path}")

    body_names = gym.get_asset_rigid_body_names(asset)
    dof_names = gym.get_asset_dof_names(asset)
    expected_joints = T800_CONTROLLED_JOINT_NAMES + T800_HEAD_JOINT_NAMES
    missing = [name for name in expected_joints if name not in dof_names]
    extra = [name for name in dof_names if name not in expected_joints]

    print(f"asset: {asset_path}")
    print(f"bodies: {len(body_names)}")
    print(f"dofs: {len(dof_names)}")
    print(f"controlled_dofs: {len(T800_CONTROLLED_JOINT_NAMES)}")
    print(f"head_dofs: {len(T800_HEAD_JOINT_NAMES)}")
    print(f"missing_expected_dofs: {missing}")
    print(f"extra_dofs: {extra}")
    if missing:
        raise RuntimeError("T800 asset is missing expected DOFs")

    if args.viewer:
        gym.add_ground(sim, gymapi.PlaneParams())
        env = gym.create_env(sim, gymapi.Vec3(-1.5, -1.5, 0.0), gymapi.Vec3(1.5, 1.5, 1.5), 1)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, T800Cfg.init_state.pos[2])
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        actor = gym.create_actor(env, asset, pose, T800Cfg.asset.name, 0, T800Cfg.asset.self_collisions, 0)
        _set_default_dof_state(gym, env, actor, dof_names)

        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create Isaac Gym viewer")
        gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(2.0, -2.0, 1.4), gymapi.Vec3(0.0, 0.0, 0.6))
        while not gym.query_viewer_has_closed(viewer):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
        gym.destroy_viewer(viewer)

    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
