#!/bin/bash
#SBATCH --job-name=host-t800-lowreg
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx4090d:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export http_proxy=http://10.12.120.125:7897
export https_proxy=http://10.12.120.125:7897
export HTTP_PROXY=http://10.12.120.125:7897
export HTTPS_PROXY=http://10.12.120.125:7897

cd /srv/shared/home/kyx/Workspace/HoST

source /home/kyx/miniconda3/etc/profile.d/conda.sh
conda activate host

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"

echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "task=t800_host_ground"
echo "experiment=lowreg"
echo "run_name=slurm_4090_lowreg_getup_4096env_30000it"
echo "num_envs=4096"
echo "max_iterations=30000"
echo "gpu_request=rtx4090d:1"
nvidia-smi
python -u -c 'import isaacgym; import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'
python -u -c 'import isaacgym; from legged_gym.envs.t800.t800_config_ground import T800Cfg; S = T800Cfg.constraints.scales; keys = ["regu_dof_acc", "regu_action_rate", "regu_smoothness", "regu_joint_power", "regu_dof_vel", "regu_dof_pos_limits", "style_feet_distance", "style_collision"]; print("constraint_scales", {key: getattr(S, key) for key in keys})'
python -u -c 'import isaacgym; from legged_gym.envs.t800.t800_config_ground import T800_TARGET_JOINT_ANGLES as T; keys = ["J00_HIP_PITCH_L", "J04_ANKLE_PITCH_L", "J06_HIP_PITCH_R", "J10_ANKLE_PITCH_R", "J13_SHOULDER_PITCH_L", "J20_SHOULDER_PITCH_R"]; print("target_pose", {key: T[key] for key in keys})'

python legged_gym/legged_gym/scripts/train.py \
  --task t800_host_ground \
  --run_name slurm_4090_lowreg_getup_4096env_30000it \
  --headless \
  --num_envs 4096 \
  --max_iterations 30000
