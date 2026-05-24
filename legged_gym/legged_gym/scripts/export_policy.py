import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import isaacgym  # noqa: F401
import numpy as np
import torch

import legged_gym.envs  # noqa: F401
import rsl_rl.modules as rsl_modules
from legged_gym.utils import class_to_dict, export_policy_as_jit, get_load_path, task_registry


T800_HOST_TO_ENGINEAI_JOINT = {
    "J20_SHOULDER_PITCH_R": "J18_SHOULDER_PITCH_R",
    "J21_SHOULDER_ROLL_R": "J19_SHOULDER_ROLL_R",
    "J22_SHOULDER_YAW_R": "J20_SHOULDER_YAW_R",
    "J23_ELBOW_PITCH_R": "J21_ELBOW_PITCH_R",
    "J24_ELBOW_YAW_R": "J22_ELBOW_YAW_R",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export a trained HoST policy for deployment.")
    parser.add_argument("--task", required=True, help="Registered legged_gym task name, e.g. t800_host_ground.")
    parser.add_argument("--checkpoint_path", help="Exact model_*.pt checkpoint to export.")
    parser.add_argument("--load_run", default=-1, help="Run folder used when checkpoint_path is not set.")
    parser.add_argument("--checkpoint", type=int, default=-1, help="Checkpoint number used when checkpoint_path is not set.")
    parser.add_argument("--output_dir", required=True, help="Directory where policy artifacts are written.")
    parser.add_argument(
        "--formats",
        default="torchscript,onnx",
        help="Comma-separated export formats. Supported: torchscript, onnx.",
    )
    parser.add_argument("--action_rescale", type=float, required=True, help="Deployment action_rescale constant.")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version.")
    parser.add_argument(
        "--no_mnn_conversion",
        action="store_true",
        help="Skip ONNX to MNN conversion and write conversion.blocked.txt.",
    )
    parser.add_argument("--device", default="cpu", help="Device used to load the policy before CPU export.")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_file_iteration(path):
    match = re.search(r"model_(\d+)\.pt$", Path(path).name)
    return int(match.group(1)) if match else None


def run_git(args, cwd):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostic metadata only
        return f"unavailable: {exc}"


def git_metadata(path):
    cwd = Path(path).resolve()
    if cwd.is_file():
        cwd = cwd.parent
    return {
        "repo": run_git(["rev-parse", "--show-toplevel"], cwd),
        "commit": run_git(["rev-parse", "HEAD"], cwd),
        "status_short": run_git(["status", "--short"], cwd),
    }


def resolve_checkpoint(task, checkpoint_path, load_run, checkpoint):
    if checkpoint_path:
        return Path(checkpoint_path).resolve()

    _, train_cfg = task_registry.get_cfgs(task)
    log_root = Path("legged_gym/logs") / train_cfg.runner.experiment_name
    return Path(get_load_path(str(log_root), load_run=load_run, checkpoint=checkpoint)).resolve()


def load_actor_critic(task, checkpoint_path, device):
    env_cfg, train_cfg = task_registry.get_cfgs(task)
    train_cfg_dict = class_to_dict(train_cfg)
    policy_class = getattr(rsl_modules, train_cfg.runner.policy_class_name)
    num_critic_obs = env_cfg.env.num_privileged_obs or env_cfg.env.num_observations
    actor_critic = policy_class(
        env_cfg.env.num_observations,
        num_critic_obs,
        env_cfg.env.num_actions,
        env_cfg.rewards.num_reward_groups,
        **train_cfg_dict["policy"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor_critic.load_state_dict(checkpoint["model_state_dict"])
    actor_critic.eval()
    return actor_critic, env_cfg, train_cfg, checkpoint


def export_torchscript(actor_critic, output_dir):
    export_policy_as_jit(actor_critic, str(output_dir))
    path = output_dir / "policy_1.pt"
    if not path.exists():
        raise FileNotFoundError(f"TorchScript export did not create {path}")
    return path


def verify_torchscript(actor_critic, torchscript_path, input_dim):
    actor = actor_critic.actor.to("cpu").eval()
    scripted_actor = torch.jit.load(str(torchscript_path), map_location="cpu").eval()
    generator = torch.Generator(device="cpu").manual_seed(0)
    obs = torch.randn(8, input_dim, generator=generator)
    with torch.no_grad():
        expected = actor(obs)
        actual = scripted_actor(obs)
    return float((expected - actual).abs().max().item())


def verification_obs(input_dim):
    generator = torch.Generator(device="cpu").manual_seed(0)
    return torch.randn(1, input_dim, generator=generator, dtype=torch.float32)


def pytorch_actor_output(actor_critic, obs):
    actor = actor_critic.actor.to("cpu").eval()
    with torch.no_grad():
        return actor(obs).cpu().numpy()


def verify_onnx(actor_critic, onnx_path, input_dim, output_dim):
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnxruntime is required for ONNX parity verification") from exc

    obs = verification_obs(input_dim)
    expected = pytorch_actor_output(actor_critic, obs)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"obs": obs.numpy()})[0]
    if actual.shape != (1, output_dim):
        raise ValueError(f"ONNX output shape mismatch: {actual.shape} != {(1, output_dim)}")
    return {
        "status": "verified",
        "output_shape": list(actual.shape),
        "max_abs_error_vs_pytorch": float(np.max(np.abs(expected - actual))),
    }


def verify_mnn(actor_critic, mnn_path, input_dim, output_dim):
    try:
        import MNN
    except ModuleNotFoundError as exc:
        raise RuntimeError("MNN Python runtime is required for MNN parity verification") from exc

    obs = verification_obs(input_dim)
    expected = pytorch_actor_output(actor_critic, obs)
    obs_np = obs.numpy().astype(np.float32)

    interpreter = MNN.Interpreter(str(mnn_path))
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    if tuple(input_tensor.getShape()) != (1, input_dim):
        raise ValueError(f"MNN input shape mismatch: {input_tensor.getShape()} != {(1, input_dim)}")
    mnn_input = MNN.Tensor(obs_np.shape, MNN.Halide_Type_Float, obs_np, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(mnn_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    output_shape = tuple(output_tensor.getShape())
    if output_shape != (1, output_dim):
        raise ValueError(f"MNN output shape mismatch: {output_shape} != {(1, output_dim)}")
    actual = np.array(output_tensor.getData(), dtype=np.float32).reshape(output_shape)
    return {
        "status": "verified",
        "input_shape": [1, input_dim],
        "output_shape": list(output_shape),
        "max_abs_error_vs_pytorch": float(np.max(np.abs(expected - actual))),
    }


def export_onnx(actor_critic, output_dir, input_dim, opset):
    actor = actor_critic.actor.to("cpu").eval()
    dummy_obs = torch.zeros(1, input_dim, dtype=torch.float32)
    onnx_path = output_dir / "policy.onnx"
    with torch.no_grad():
        torch.onnx.export(
            actor,
            dummy_obs,
            str(onnx_path),
            input_names=["obs"],
            output_names=["actions"],
            opset_version=opset,
        )
    return onnx_path


def write_text(path, text):
    path.write_text(text, encoding="utf-8")


def find_mnn_converter():
    try:
        if importlib.util.find_spec("MNN.tools.mnnconvert") is not None:
            return [sys.executable, "-m", "MNN.tools.mnnconvert"], "python-module"
    except ModuleNotFoundError:
        pass
    converter = shutil.which("mnnconvert")
    if converter:
        return [converter], "binary"
    return None, None


def convert_onnx_to_mnn(onnx_path, output_dir, skip_conversion):
    mnn_path = output_dir / "policy.mnn"
    for stale_marker in (output_dir / "conversion.blocked.txt", output_dir / "conversion.failed.txt"):
        if stale_marker.exists():
            stale_marker.unlink()

    if skip_conversion:
        blocked_path = output_dir / "conversion.blocked.txt"
        write_text(blocked_path, "MNN conversion skipped by --no_mnn_conversion.\n")
        return {"status": "blocked", "blocked_file": str(blocked_path)}

    converter, converter_kind = find_mnn_converter()
    if converter is None:
        blocked_path = output_dir / "conversion.blocked.txt"
        write_text(
            blocked_path,
            "MNN converter not found. Install MNN Python tools or put mnnconvert on PATH, then run:\n"
            f"python -m MNN.tools.mnnconvert -f ONNX --modelFile {onnx_path} "
            f"--MNNModel {mnn_path} --bizCode MNN\n",
        )
        return {"status": "blocked", "blocked_file": str(blocked_path)}

    command = [
        *converter,
        "-f",
        "ONNX",
        "--modelFile",
        str(onnx_path),
        "--MNNModel",
        str(mnn_path),
        "--bizCode",
        "MNN",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        failed_path = output_dir / "conversion.failed.txt"
        write_text(
            failed_path,
            "MNN conversion command failed.\n\n"
            f"command: {' '.join(command)}\n\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}\n",
        )
        return {
            "status": "failed",
            "failed_file": str(failed_path),
            "command": command,
            "converter_kind": converter_kind,
            "returncode": result.returncode,
        }

    return {
        "status": "converted",
        "path": str(mnn_path),
        "sha256": sha256_file(mnn_path),
        "command": command,
        "converter_kind": converter_kind,
    }


def get_controlled_joint_names(env_cfg):
    if hasattr(env_cfg.asset, "controlled_joint_names"):
        return list(env_cfg.asset.controlled_joint_names)
    return list(env_cfg.init_state.default_joint_angles.keys())[: env_cfg.env.num_actions]


def get_action_scale_vector(env_cfg, joint_names):
    action_scale = env_cfg.control.action_scale
    if isinstance(action_scale, dict):
        return [float(action_scale[name]) for name in joint_names]
    return [float(action_scale)] * len(joint_names)


def get_obs_scales(env_cfg):
    scales = env_cfg.normalization.obs_scales
    return {
        "ang_vel": float(scales.ang_vel),
        "dof_pos": float(scales.dof_pos),
        "dof_vel": float(scales.dof_vel),
        "projected_gravity": 1.0,
        "previous_actions": 1.0,
        "action_rescale": 1.0,
    }


def get_history_order_description(env_cfg):
    step_dim = int(env_cfg.env.num_one_step_observations)
    history_length = int(env_cfg.env.num_actor_history)
    newest_start = step_dim * (history_length - 1)
    newest_end = step_dim * history_length
    return f"oldest_to_newest; newest block is [{newest_start}:{newest_end})"


def write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main():
    args = parse_args()
    formats = {item.strip().lower() for item in args.formats.split(",") if item.strip()}
    unsupported = formats - {"torchscript", "onnx"}
    if unsupported:
        raise ValueError(f"Unsupported export format(s): {sorted(unsupported)}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve_checkpoint(args.task, args.checkpoint_path, args.load_run, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    actor_critic, env_cfg, train_cfg, checkpoint = load_actor_critic(args.task, checkpoint_path, args.device)
    joint_names = get_controlled_joint_names(env_cfg)
    deploy_joint_names = [T800_HOST_TO_ENGINEAI_JOINT.get(name, name) for name in joint_names]
    action_scale_vector = get_action_scale_vector(env_cfg, joint_names)

    state_iteration = checkpoint.get("iter")
    file_iteration = checkpoint_file_iteration(checkpoint_path)
    manifest = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": args.task,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_iteration": file_iteration if file_iteration is not None else state_iteration,
        "checkpoint_state_iteration": state_iteration,
        "checkpoint_file_iteration": file_iteration,
        "input_dim": int(env_cfg.env.num_observations),
        "output_dim": int(env_cfg.env.num_actions),
        "history_length": int(env_cfg.env.num_actor_history),
        "one_step_observation_dim": int(env_cfg.env.num_one_step_observations),
        "history_order": get_history_order_description(env_cfg),
        "observation_order": [
            "base_ang_vel",
            "projected_gravity",
            "controlled_dof_pos",
            "controlled_dof_vel",
            "previous_raw_actions",
            "action_rescale",
        ],
        "observation_scales": get_obs_scales(env_cfg),
        "observation_clip": float(env_cfg.normalization.clip_observations),
        "action_rescale": float(args.action_rescale),
        "action_target_formula": "q_des = q_real + raw_action * action_rescale * action_scale",
        "host_joint_names": joint_names,
        "engineai_joint_names": deploy_joint_names,
        "action_scale": action_scale_vector,
        "host_git": git_metadata(Path(__file__).resolve()),
        "output_git": git_metadata(output_dir),
        "artifacts": {},
    }

    if "torchscript" in formats:
        ts_path = export_torchscript(actor_critic, output_dir)
        manifest["artifacts"]["torchscript"] = {
            "path": str(ts_path),
            "sha256": sha256_file(ts_path),
            "max_abs_error_vs_pytorch": verify_torchscript(actor_critic, ts_path, env_cfg.env.num_observations),
        }

    onnx_path = None
    if "onnx" in formats:
        onnx_path = export_onnx(actor_critic, output_dir, env_cfg.env.num_observations, args.opset)
        manifest["artifacts"]["onnx"] = {
            "path": str(onnx_path),
            "sha256": sha256_file(onnx_path),
            "opset": args.opset,
            "parity": verify_onnx(actor_critic, onnx_path, env_cfg.env.num_observations, env_cfg.env.num_actions),
        }
        manifest["artifacts"]["mnn_conversion"] = convert_onnx_to_mnn(onnx_path, output_dir, args.no_mnn_conversion)
        if manifest["artifacts"]["mnn_conversion"].get("status") == "converted":
            mnn_path = Path(manifest["artifacts"]["mnn_conversion"]["path"])
            manifest["artifacts"]["mnn_conversion"]["parity"] = verify_mnn(
                actor_critic,
                mnn_path,
                env_cfg.env.num_observations,
                env_cfg.env.num_actions,
            )

    manifest_path = output_dir / "policy_manifest.json"
    write_manifest(manifest_path, manifest)
    print(f"Wrote manifest: {manifest_path}")
    for name, artifact in manifest["artifacts"].items():
        print(f"{name}: {artifact}")

    if manifest["artifacts"].get("mnn_conversion", {}).get("status") == "failed":
        raise SystemExit("MNN conversion failed; ONNX/TorchScript artifacts were left intact.")


if __name__ == "__main__":
    main()
