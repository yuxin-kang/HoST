import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import torch
import time

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


def log_getup_metrics(env, actions, step):
    env_id = 0
    root_z = env.root_states[env_id, 2].item()
    head_z = env.rigid_body_states[env_id, env.head_indices, 2].mean().item()
    feet_z = env.rigid_body_states[env_id, env.feet_indices, 2].mean().item()
    head_rel_feet = head_z - feet_z
    left_foot = env.rigid_body_states[env_id, env.left_foot_indices, :3].squeeze(0)
    right_foot = env.rigid_body_states[env_id, env.right_foot_indices, :3].squeeze(0)
    orientation_score = (-env.projected_gravity[env_id, 2]).item()
    phase1 = root_z > env.cfg.rewards.target_base_height_phase1
    phase2 = root_z > env.cfg.rewards.target_base_height_phase2
    phase3 = root_z > env.cfg.rewards.target_base_height_phase3
    action_abs_max = actions[env_id].abs().max().item()
    action_abs_mean = actions[env_id].abs().mean().item()
    pull_force_enabled = bool(env.cfg.curriculum.pull_force)
    cfg_force = float(env.cfg.curriculum.force)
    runtime_force = float(env.force[env_id].item()) if hasattr(env, "force") else 0.0
    force_active = int(env.real_episode_length_buf[env_id].item() > env.unactuated_time)
    print(
        "[getup] "
        f"step={step:05d} "
        f"root_z={root_z:.3f} "
        f"head_rel_feet={head_rel_feet:.3f} "
        f"orientation={orientation_score:.3f} "
        f"phase=({int(phase1)},{int(phase2)},{int(phase3)}) "
        f"pull_force=({int(pull_force_enabled)},cfg={cfg_force:.1f},runtime={runtime_force:.1f},active={force_active}) "
        f"targets=(root>{env.cfg.rewards.target_base_height_phase3:.2f}, "
        f"head>{env.cfg.rewards.target_head_height:.2f}) "
        f"left_foot=({left_foot[0].item():.2f},{left_foot[1].item():.2f},{left_foot[2].item():.2f}) "
        f"right_foot=({right_foot[0].item():.2f},{right_foot[1].item():.2f},{right_foot[2].item():.2f}) "
        f"action_abs=(mean={action_abs_mean:.3f},max={action_abs_max:.3f})",
        flush=True,
    )


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    if args.play_asset_debug:
        env_cfg.asset.fix_base_link = True
        env_cfg.asset.disable_gravity = True
        env_cfg.init_state.rot = [0.0, 0.0, 0.0, 1.0]
        for domain_rand_attr in (
            "use_random",
            "randomize_initial_joint_pos",
            "randomize_actuation_offset",
            "randomize_motor_strength",
            "randomize_kp",
            "randomize_kd",
            "randomize_payload_mass",
            "randomize_com_displacement",
            "randomize_link_mass",
            "randomize_friction",
            "randomize_restitution",
            "delay",
            "push_robots",
        ):
            if hasattr(env_cfg.domain_rand, domain_rand_attr):
                setattr(env_cfg.domain_rand, domain_rand_attr, False)
        env_cfg.curriculum.pull_force = False
        print(
            "[play_asset_debug] "
            f"asset={env_cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)}, "
            f"init_pos={env_cfg.init_state.pos}, fix_base_link={env_cfg.asset.fix_base_link}"
        )
    if args.play_action_scale is not None:
        env_cfg.control.action_scale = args.play_action_scale
    elif not args.play_keep_train_curriculum and not isinstance(env_cfg.control.action_scale, (dict, list, tuple)):
        env_cfg.control.action_scale = 0.3
    if not args.play_keep_train_curriculum:
        env_cfg.curriculum.pull_force = False
    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if args.play_asset_debug:
        env.actuation_offset.zero_()
        env.motor_strength.fill_(1.0)
        env.Kp_factors.fill_(1.0)
        env.Kd_factors.fill_(1.0)
        env.reset_idx(torch.arange(env.num_envs, device=env.device))
        env.compute_observations()
        print(
            "[play_asset_debug] "
            f"actuation_offset_max={env.actuation_offset.abs().max().item():.6f}, "
            f"motor_strength=({env.motor_strength.min().item():.3f}, {env.motor_strength.max().item():.3f}), "
            f"kp_factor=({env.Kp_factors.min().item():.3f}, {env.Kp_factors.max().item():.3f}), "
            f"kd_factor=({env.Kd_factors.min().item():.3f}, {env.Kd_factors.max().item():.3f})"
        )
    obs = env.get_observations()
    debug_initial_dof_pos = env.dof_pos.clone() if args.play_asset_debug else None
    debug_max_abs_dof_delta = 0.0
    policy_device = torch.device(env.device)

    if args.play_asset_debug:
        policy = lambda obs: torch.zeros(env.num_envs, env.num_actions, device=env.device)
    else:
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, env_cfg=env_cfg, name=args.task, args=args, train_cfg=train_cfg)
        policy_device = torch.device(args.rl_device)
        policy = ppo_runner.get_inference_policy(device=policy_device)
    
    logger = Logger(env.dt)
    num_steps = args.play_steps if args.play_steps is not None else 10 * int(env.max_episode_length)
    for i in range(num_steps):

        result = env.gym.fetch_results(env.sim, True)
        actions = policy(obs.detach().to(policy_device))
        obs, _, rews, dones, infos = env.step(actions.detach().to(env.device))
        if args.play_log_getup_metrics and i % max(args.play_log_interval, 1) == 0:
            log_getup_metrics(env, actions.detach().to(env.device), i)
        if args.play_asset_debug:
            dof_delta = (env.dof_pos - debug_initial_dof_pos).abs().max().item()
            debug_max_abs_dof_delta = max(debug_max_abs_dof_delta, dof_delta)

    if args.play_asset_debug:
        print(
            "[play_asset_debug] "
            f"max_abs_dof_delta={debug_max_abs_dof_delta:.6f}, "
            f"final_abs_dof_vel={env.dof_vel.abs().max().item():.6f}"
        )


if __name__ == '__main__':
    args = get_args()
    play(args)
