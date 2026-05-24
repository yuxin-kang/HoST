# T800 参数代码修改计划

## Summary

本文记录当前工作树里的 T800 配置改动计划，不启动训练。

范围分两层：

- 当前工作树已包含的 T800 get-up / H1 baseline 变更，需要作为合并前背景一起审阅。
- 本轮参数调节只覆盖探索拉力和 PD 表：把探索拉力设为 `1000`，并把 T800 腿部/躯干 PD 增益调到高于 H1；上肢和头部不纳入“高于 H1”的目标。

保留现有 `beta/action_rescale` 从 `1.0` 动态下降到 `0.25` 的逻辑。

## Current Baseline In Working Tree

这些变更已经出现在当前 `t800_config_ground.py` 相对 HEAD 的 diff 中。它们不是本轮 `1000N + PD` 调参的新增范围，但如果按当前工作树整体合并，必须一起审阅：

- `T800Cfg` / `T800CfgPPO` 的父类从 `G1Cfg` / `G1CfgPPO` 切到 `H1Cfg` / `H1CfgPPO`。
- 新增 get-up 初始化旋转常量：
  - `T800_GETUP_ROOT_ROT = [0.0, -1.0, 0.0, 1.0]`
- `T800Cfg.init_state.pos` 明确使用 `[0.0, 0.0, 0.5]`，不再保留单独的 T800 get-up root height 常量。
- T800 站立目标常量使用当前实测值：
  - `T800_STANDING_ROOT_HEIGHT = 1.037`
  - `T800_STANDING_HEAD_HEIGHT = 1.567`
- `T800Cfg.init_state.rot` 使用 `T800_GETUP_ROOT_ROT`。
- `T800Cfg.curriculum` 在 T800 配置内显式覆盖 pull-force curriculum。
- `T800CfgPPO.runner` 显式设置：
  - `save_interval = 1000`
  - `max_iterations = 30000`
- T800 不再显式写 `control.action_rescale = 1.0`；在 `host_ground.py` 中，dict 型 `action_scale` 且未配置 `action_rescale` 时仍会初始化为 `1.0`。

## Key Changes

修改 `legged_gym/legged_gym/envs/t800/t800_config_ground.py`。

显式设置 `T800Cfg.curriculum.force = 1000`。

保持这些项不变：

- `pull_force = True`
- `threshold_height = 1.42`
- `dof_vel_limit = 300`
- `base_vel_limit = 20`
- `no_orientation = True`

将 T800 PD 改为：

```python
stiffness = {
    "HIP_PITCH": 450,
    "HIP_ROLL": 420,
    "HIP_YAW": 360,
    "KNEE_PITCH": 450,
    "ANKLE": 160,
    "TORSO_YAW": 260,
    "SHOULDER": 220,
    "ELBOW_PITCH": 180,
    "ELBOW_YAW": 140,
    "HEAD": 80,
}
damping = {
    "HIP_PITCH": 7,
    "HIP_ROLL": 6,
    "HIP_YAW": 5,
    "KNEE_PITCH": 7,
    "ANKLE": 3,
    "TORSO_YAW": 5,
    "SHOULDER": 3,
    "ELBOW_PITCH": 3,
    "ELBOW_YAW": 2,
    "HEAD": 1,
}
```

H1 对照边界：

- H1 hip/knee 为 `350/4`，T800 hip/knee 为 `360-450/5-7`，高于 H1。
- H1 ankle 为 `120/2`，T800 ankle 为 `160/3`，高于 H1。
- H1 torso 为 `200/4`，T800 torso yaw 为 `260/5`，高于 H1。
- H1 shoulder/elbow 为 `350/4`，T800 shoulder/elbow 为 `220/180/140` 和 `3/3/2`，低于 H1；这是刻意排除上肢的参数目标，不应描述为“全身 PD 高于 H1”。
- HEAD 没有直接 H1 对照项。

## Keep Unchanged

- 不改 `unactuated_timesteps`。
- 不改 `action_rescale` 逻辑；现有代码已经是从 `1.0` 成功后每次减 `0.02`，下限 `0.25`。
- 不改训练脚本，不提交训练任务。
- 不把 `tests/test_t800_host_config.py` 的旧 `500N`、源 PD、直接 `T800Cfg.control.action_rescale` 断言当作本计划的通过依据；若要跑该测试，需要先把期望值更新到当前方案。

## Verification

只做静态验证，不跑训练。

```bash
python3 -m py_compile legged_gym/legged_gym/envs/t800/t800_config_ground.py
```

静态确认 25 个 T800 DOF 都能按 `host_ground.py` 的 substring 规则绑定到非零 PD gain：

```bash
python3 - <<'PY'
import ast
from pathlib import Path

path = Path("legged_gym/legged_gym/envs/t800/t800_config_ground.py")
module = ast.parse(path.read_text())

def literal_assignment(nodes, name):
    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment: {name}")

def class_body(nodes, name):
    for node in nodes:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node.body
    raise AssertionError(f"missing class: {name}")

joints = literal_assignment(module.body, "T800_DFS_JOINT_NAMES")
t800 = class_body(module.body, "T800Cfg")
control = class_body(t800, "control")
curriculum = class_body(t800, "curriculum")
stiffness = literal_assignment(control, "stiffness")
damping = literal_assignment(control, "damping")
force = literal_assignment(curriculum, "force")

assert len(joints) == 25, len(joints)
assert force == 1000, force

missing = []
for joint in joints:
    p = next((stiffness[key] for key in stiffness if key in joint), 0)
    d = next((damping[key] for key in damping if key in joint), 0)
    if p <= 0 or d <= 0:
        missing.append(joint)

assert not missing, missing
print("all 25 T800 DOFs resolve nonzero PD gains; force=1000")
PY
```

可选 runtime import 确认配置加载。该检查需要 Isaac Gym / conda runtime；如果本机报 `libpython3.8.so.1.0`，优先使用上面的纯静态检查，不把 runtime import 失败误判为配置语法失败。

```bash
LD_LIBRARY_PATH=/srv/shared/home/kyx/miniconda3/envs/host/lib:$LD_LIBRARY_PATH \
  /srv/shared/home/kyx/miniconda3/envs/host/bin/python -c 'import isaacgym; from legged_gym.envs.t800.t800_config_ground import T800Cfg; print(T800Cfg.curriculum.force); print(T800Cfg.control.stiffness); print(T800Cfg.control.damping)'
```

## Assumptions

- 本轮新增调参只改探索拉力和 PD 表；当前工作树已存在的 get-up / H1 baseline 变更按上文单独审阅，不提交训练任务。
- 客户端通过 Mutagen 同步后会拿到同一份 `t800_config_ground.py`。
- 训练脚本继续使用现有 `t800_host_ground` 任务和 `4096 env / 30000 iters` 配置。
- 如果后续真实要求“所有 T800 joints 的 PD 都高于 H1”，当前上肢参数不满足，需要另开参数调整，而不是继续沿用本文档的上肢排除边界。
