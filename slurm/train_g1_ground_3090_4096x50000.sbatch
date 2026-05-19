#!/bin/bash
#SBATCH --job-name=host-g1-50k
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:rtx3090:1
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
echo "task=g1_ground"
echo "num_envs=4096"
echo "max_iterations=50000"
nvidia-smi
python -c 'import isaacgym; import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'

python legged_gym/legged_gym/scripts/train.py \
  --task g1_ground \
  --run_name slurm_3090_4096env_50000it \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000
