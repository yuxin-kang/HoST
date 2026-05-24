import argparse
import json
import os
from datetime import datetime
from types import SimpleNamespace

import isaacgym
from isaacgym import gymapi
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import task_registry


REGU_SCALES = {
    "action_rate": -0.01,
    "smoothness": -0.01,
    "dof_acc": -2.5e-7,
    "dof_pos_limits": -100.0,
}

DEFAULT_SPECS = [
    "t800_new:t800_host_ground:May21_20-35-04_slurm_4090_getup_4096env_30000it:17000",
    "h1_base:h1_ground:May21_09-22-26_slurm_4090_default_4096env_30000it:17000",
]

DOMAIN_RAND_FLAGS = [
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
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare regularization reward terms across checkpoints.")
    parser.add_argument("--spec", action="append", help="label:task:load_run:checkpoint. Repeatable.")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--rl-device", type=str, default="cuda:0")
    parser.add_argument("--disable-noise", action="store_true", default=False)
    parser.add_argument("--disable-domain-rand", action="store_true", default=False)
    parser.add_argument("--disable-pull-force", action="store_true", default=False)
    parser.add_argument("--policy-mode", choices=["inference", "sample"], default="inference")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def build_task_args(args, load_run, checkpoint):
    sim_device_type = "cuda" if args.sim_device.startswith("cuda") else args.sim_device
    sim_device_id = 0
    if ":" in args.sim_device:
        sim_device_type, sim_device_id_str = args.sim_device.split(":", 1)
        sim_device_id = int(sim_device_id_str)
    return SimpleNamespace(
        physics_engine=gymapi.SIM_PHYSX,
        device=sim_device_type,
        use_gpu=sim_device_type == "cuda",
        use_gpu_pipeline=sim_device_type == "cuda",
        subscenes=0,
        num_threads=0,
        headless=True,
        rl_device=args.rl_device,
        num_envs=args.num_envs,
        seed=args.seed,
        max_iterations=None,
        resume=True,
        experiment_name=None,
        run_name=None,
        load_run=load_run,
        checkpoint=checkpoint,
        checkpoint_path=None,
        sim_device_id=sim_device_id,
        sim_device_type=sim_device_type,
        sim_device=args.sim_device,
    )


def maybe_disable_domain_rand(domain_rand_cfg):
    for attr in DOMAIN_RAND_FLAGS:
        if hasattr(domain_rand_cfg, attr):
            setattr(domain_rand_cfg, attr, False)


def top_joint_entries(names, values, top_k):
    order = np.argsort(values)[::-1][:top_k]
    return [
        {
            "rank": index + 1,
            "joint": names[joint_index],
            "value": float(values[joint_index]),
            "share": float(values[joint_index] / max(values.sum(), 1e-12)),
        }
        for index, joint_index in enumerate(order)
    ]


def analyze_run(spec, cli_args):
    label, task_name, load_run, checkpoint_str = spec.split(":", 3)
    checkpoint = int(checkpoint_str)
    task_args = build_task_args(cli_args, load_run, checkpoint)
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)

    env_cfg.env.num_envs = cli_args.num_envs
    env_cfg.env.test = False
    if cli_args.disable_noise:
        env_cfg.noise.add_noise = False
    if cli_args.disable_domain_rand:
        maybe_disable_domain_rand(env_cfg.domain_rand)
    if cli_args.disable_pull_force:
        env_cfg.curriculum.pull_force = False

    env, _ = task_registry.make_env(name=task_name, args=task_args, env_cfg=env_cfg)
    obs = env.get_observations()
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    ppo_runner, _ = task_registry.make_alg_runner(
        env=env,
        env_cfg=env_cfg,
        name=task_name,
        args=task_args,
        train_cfg=train_cfg,
        log_root=log_root,
    )
    if cli_args.policy_mode == "inference":
        policy = ppo_runner.get_inference_policy(device=task_args.rl_device)
    else:
        actor_critic = ppo_runner.alg.actor_critic
        actor_critic.eval()
        actor_critic.to(task_args.rl_device)
        policy = actor_critic.act

    controlled_indices = env.controlled_dof_indices
    joint_names = getattr(env.cfg.asset, "controlled_joint_names", None)
    if joint_names is None:
        joint_names = [env.dof_names[index] for index in controlled_indices.tolist()]

    prev_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    prev_prev_actions = torch.zeros_like(prev_actions)
    prev_dof_vel = env.dof_vel[:, controlled_indices].clone()

    sums = {
        "action_rate": torch.zeros(env.num_actions, device=env.device),
        "smoothness": torch.zeros(env.num_actions, device=env.device),
        "dof_acc": torch.zeros(env.num_actions, device=env.device),
        "dof_pos_limits": torch.zeros(env.num_actions, device=env.device),
    }

    with torch.no_grad():
        for _ in range(cli_args.steps):
            policy_actions = policy(obs.detach().to(task_args.rl_device))
            obs, _, _, _, _ = env.step(policy_actions.detach().to(env.device))

            curr_actions = env.actions.clone()
            curr_dof_vel = env.dof_vel[:, controlled_indices].clone()

            action_rate = torch.square(curr_actions - prev_actions)
            smoothness = torch.square(curr_actions - 2 * prev_actions + prev_prev_actions)
            dof_acc = torch.square((prev_dof_vel - curr_dof_vel) / env.dt)

            dof_pos = env.dof_pos[:, controlled_indices]
            dof_limits = env.dof_pos_limits[controlled_indices]
            dof_pos_limits = -(dof_pos - dof_limits[:, 0]).clip(max=0.0)
            dof_pos_limits += (dof_pos - dof_limits[:, 1]).clip(min=0.0)

            sums["action_rate"] += action_rate.mean(dim=0)
            sums["smoothness"] += smoothness.mean(dim=0)
            sums["dof_acc"] += dof_acc.mean(dim=0)
            sums["dof_pos_limits"] += dof_pos_limits.mean(dim=0)

            prev_prev_actions = prev_actions
            prev_actions = curr_actions
            prev_dof_vel = curr_dof_vel

    result = {
        "label": label,
        "task": task_name,
        "load_run": load_run,
        "checkpoint": checkpoint,
        "num_envs": cli_args.num_envs,
        "steps": cli_args.steps,
        "dt": float(env.dt),
        "force": float(env.force.mean().item()) if hasattr(env, "force") else None,
        "action_rescale_mean": float(env.action_rescale.mean().item()),
        "policy_mode": cli_args.policy_mode,
        "noise_std_mean": float(ppo_runner.alg.actor_critic.std.mean().item()),
        "joint_names": joint_names,
        "metrics": {},
    }

    for metric_name, per_joint_sum in sums.items():
        per_joint_mean = (per_joint_sum / cli_args.steps).detach().cpu().numpy()
        raw_mean_sum = float(per_joint_mean.sum())
        result["metrics"][metric_name] = {
            "raw_mean_sum": raw_mean_sum,
            "predicted_logged": float(raw_mean_sum * REGU_SCALES[metric_name]),
            "raw_mean_per_joint_avg": float(raw_mean_sum / len(joint_names)),
            "top_joints": top_joint_entries(joint_names, per_joint_mean, cli_args.top_k),
        }

    if env.viewer is not None:
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)
    del env
    del ppo_runner
    torch.cuda.empty_cache()
    return result


def print_summary(results):
    for result in results:
        print(
            f"[{result['label']}] task={result['task']} checkpoint={result['checkpoint']} "
            f"policy_mode={result['policy_mode']} noise_std_mean={result['noise_std_mean']:.6f}"
        )
        for metric_name in ("action_rate", "smoothness", "dof_acc", "dof_pos_limits"):
            metric = result["metrics"][metric_name]
            print(
                f"  {metric_name}: raw_mean_sum={metric['raw_mean_sum']:.6f} "
                f"predicted_logged={metric['predicted_logged']:.6f}"
            )
            for entry in metric["top_joints"][:3]:
                print(
                    f"    top{entry['rank']} joint={entry['joint']} "
                    f"value={entry['value']:.6f} share={entry['share']:.3f}"
                )
        print()

    if len(results) == 2:
        left, right = results
        print(f"[ratio] {left['label']} / {right['label']}")
        for metric_name in ("action_rate", "smoothness", "dof_acc", "dof_pos_limits"):
            left_value = abs(left["metrics"][metric_name]["predicted_logged"])
            right_value = abs(right["metrics"][metric_name]["predicted_logged"])
            ratio = float("inf") if right_value == 0 else left_value / right_value
            print(f"  {metric_name}: {ratio:.6f}x")


def main():
    cli_args = parse_args()
    specs = cli_args.spec if cli_args.spec else DEFAULT_SPECS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = cli_args.output_dir or os.path.join(
        os.path.dirname(LEGGED_GYM_ROOT_DIR),
        "artifacts",
        f"regu_compare_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)

    results = [analyze_run(spec, cli_args) for spec in specs]
    output_path = os.path.join(output_dir, "summary.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"generated_at": timestamp, "results": results}, handle, indent=2)

    print_summary(results)
    print(f"saved_json={output_path}")


if __name__ == "__main__":
    main()
