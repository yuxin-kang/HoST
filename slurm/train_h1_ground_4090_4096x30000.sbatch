#!/bin/bash
#SBATCH --job-name=host-h1-30k
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
echo "task=h1_ground"
echo "num_envs=4096"
echo "max_iterations=30000"
echo "gpu_request=rtx4090d:1"
nvidia-smi
python -u -c 'import isaacgym; import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'

python legged_gym/legged_gym/scripts/train.py \
  --task h1_ground \
  --run_name slurm_4090_default_4096env_30000it \
  --headless \
  --num_envs 4096 \
  --max_iterations 30000
