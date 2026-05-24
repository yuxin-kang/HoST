#!/bin/bash
#SBATCH --job-name=regu-compare
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx4090d:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /srv/shared/home/kyx/Workspace/HoST

source /home/kyx/miniconda3/etc/profile.d/conda.sh
conda activate host

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
POLICY_MODE="${POLICY_MODE:-inference}"

echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "policy_mode=${POLICY_MODE}"
nvidia-smi

python -u legged_gym/legged_gym/scripts/analysis/compare_regu_terms.py \
  --num-envs 256 \
  --steps 1024 \
  --sim-device cuda:0 \
  --rl-device cuda:0 \
  --policy-mode "${POLICY_MODE}"
