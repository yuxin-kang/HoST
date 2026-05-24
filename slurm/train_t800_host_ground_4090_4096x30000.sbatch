#!/bin/bash
#SBATCH --job-name=host-t800-f500-a06
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
echo "run_name=slurm_4090_f500_pd18_a06_getup_4096env_30000it"
echo "num_envs=4096"
echo "max_iterations=30000"
echo "gpu_request=rtx4090d:1"
nvidia-smi
python -u -c 'import isaacgym; import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'
python -u -c 'import isaacgym; from legged_gym.envs.t800.t800_config_ground import T800Cfg, T800_STANDING_ROOT_HEIGHT, T800_STANDING_HEAD_HEIGHT; print("t800_config", "getup_root", T800Cfg.init_state.pos[2], "standing_root", T800_STANDING_ROOT_HEIGHT, "standing_head", T800_STANDING_HEAD_HEIGHT, "force", T800Cfg.curriculum.force, "threshold_height", T800Cfg.curriculum.threshold_height, "action_rescale_min", T800Cfg.curriculum.action_rescale_min, "pull_force", T800Cfg.curriculum.pull_force, "phase", (T800Cfg.rewards.target_base_height_phase1, T800Cfg.rewards.target_base_height_phase2, T800Cfg.rewards.target_base_height_phase3), "regu", (T800Cfg.constraints.scales.regu_action_rate, T800Cfg.constraints.scales.regu_smoothness, T800Cfg.constraints.scales.regu_dof_pos_limits))'
python -u -c 'import isaacgym; from legged_gym.envs.t800.t800_config_ground import T800Cfg; print("pd_gains", "stiffness", T800Cfg.control.stiffness, "damping", T800Cfg.control.damping)'

python legged_gym/legged_gym/scripts/train.py \
  --task t800_host_ground \
  --run_name slurm_4090_f500_pd18_a06_getup_4096env_30000it \
  --headless \
  --num_envs 4096 \
  --max_iterations 30000
