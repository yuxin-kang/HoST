# T800 Get-Up 初始化修正计划

## Summary

把 `t800_host_ground` 从“站立初始化训练”改回 HoST/G1 的 get-up 范式：低高度、翻倒姿态 reset，前 30 个 step 让机器人自然落地，再学习起身。保留已对齐的 T800 PD、action scale、25 个 DFS action joints、asset physics，不回退这些改动。

## Key Changes

- 在 `t800_config_ground.py` 增加明确旋转常量：
  - `T800_GETUP_ROOT_ROT = [0.0, -1.0, 0.0, 1.0]`
- 修改 `T800Cfg.init_state`：
  - `pos = [0.0, 0.0, 0.5]`
  - `rot = T800_GETUP_ROOT_ROT`
  - `default_joint_angles` / `target_joint_angles` 保持当前 T800 真源参数。
- 保留 standing reward 目标：
  - `base_height_target = 1.04`
  - `target_head_height = 1.55`
  - `target_base_height_phase1/2/3 = 0.85/0.90/0.98`
- 不改 `host_ground.py` 的 unactuated 逻辑、torque 逻辑、reward 逻辑；这正是 G1 get-up 训练机制的一部分。

## Training Script / Run Policy

- 更新 T800 Slurm run name，避免和坏 run 混淆：
  - 3090：`slurm_3090_getup_4096env_30000it`
  - 4090：`slurm_4090_getup_4096env_30000it`
- 新配置 smoke 通过后，当前两个坏的 T800 训练 job 可停止并重启新训练；不要停止 G1 训练。
- 旧 T800 run 目录保留作对照，但不要再当成果权重使用。

## Test Plan

- 更新 `tests/test_t800_host_config.py`：
  - 删除/替换“init height 等于 standing height”的断言。
  - 新增断言：T800 init height 为 `0.5`，init rot 为 `T800_GETUP_ROOT_ROT`。
  - 保留 PD、action scale、head joints、asset physics、25 action joints 的现有回归检查。
- 运行：
  - `python -m py_compile legged_gym/legged_gym/envs/t800/t800_config_ground.py`
  - `python -m unittest tests.test_t800_host_config`
  - `python legged_gym/legged_gym/scripts/train.py --task t800_host_ground --num_envs 64 --max_iterations 1 --headless`
- 客户端视觉验证时，用客户端 `10.12.120.125` 打开/运行本地 HoST，确认初始姿态是低高度翻倒落地，而不是站立掉倒。

## Assumptions

- T800 get-up reset height 直接使用 H1/G1 get-up 体系里的 `0.5m`，不再按 T800 standing root height 另行缩放。
- T800 的训练目标仍然是最终站起到 T800 正常站立高度，不是训练趴地或半蹲。
- 这次只修正 get-up 初始化范式，不改奖励结构、PD、action scale 或机器人资产参数。

## Local Verification Notes

- `py_compile` 已通过。
- `tests.test_t800_host_config` 已通过，8 个测试 OK。
- `git diff --check` 已通过。
- 训练 smoke 已尝试，但当前本地环境缺少可用 CUDA/Isaac Gym runtime：`libcuda.so.1` 缺失、`Device count 0`，因此无法在本机完成一轮训练初始化验证。
