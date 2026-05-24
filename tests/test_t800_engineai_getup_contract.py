import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from legged_gym.envs.t800.t800_config_ground import (
    T800Cfg,
    T800_ACTION_SCALE_BY_JOINT,
    T800_CONTROLLED_JOINT_NAMES,
)


ENGINEAI_GETUP_CONFIG = Path(
    "/srv/shared/home/kyx/robot/mimic/engineai_robotics/assets/config/t800/rl_getup_example/default.yaml"
)
ENGINEAI_MODE_CONFIG = Path("/srv/shared/home/kyx/robot/mimic/engineai_robotics/assets/config/t800/mode.yaml")
ENGINEAI_TASK_MOTION_CONFIG = Path(
    "/srv/shared/home/kyx/robot/mimic/engineai_robotics/assets/config/t800/task_motion/default.yaml"
)
ENGINEAI_POLICY_MANIFEST = Path(
    "/srv/shared/home/kyx/robot/mimic/engineai_robotics/assets/config/t800/rl_getup_example/policies/"
    "policy_manifest.json"
)
ENGINEAI_GETUP_INCLUDE_DIR = Path(
    "/srv/shared/home/kyx/robot/mimic/engineai_robotics/src/runner/rl_getup_example/include"
)
CPP_CONTRACT_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "engineai_rl_getup_contract_check.cc"


class T800EngineAIGetupContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ENGINEAI_GETUP_CONFIG.exists():
            raise unittest.SkipTest(f"EngineAI get-up config not found: {ENGINEAI_GETUP_CONFIG}")
        with ENGINEAI_GETUP_CONFIG.open("r", encoding="utf-8") as handle:
            cls.config = yaml.safe_load(handle)

    def test_engineai_getup_config_matches_host_observation_abi(self):
        self.assertEqual(self.config["num_one_step_observations"], T800Cfg.env.num_one_step_observations)
        self.assertEqual(self.config["num_include_obs_steps"], T800Cfg.env.num_actor_history)
        self.assertEqual(self.config["num_observations"], T800Cfg.env.num_observations)
        self.assertEqual(
            self.config["num_observations"],
            self.config["num_one_step_observations"] * self.config["num_include_obs_steps"],
        )
        self.assertEqual(self.config["first_frame_history_mode"], "zero_oldest_append_current")
        self.assertAlmostEqual(self.config["action_rescale"], 0.6)
        self.assertAlmostEqual(self.config["observation_clip"], T800Cfg.normalization.clip_observations)

    def test_engineai_getup_config_maps_all_policy_joints_once(self):
        host_joint_names = self.config["host_joint_names"]
        engineai_joint_names = self.config["joint_names"]
        self.assertEqual(host_joint_names, T800_CONTROLLED_JOINT_NAMES)
        self.assertEqual(len(engineai_joint_names), T800Cfg.env.num_actions)
        self.assertEqual(len(set(engineai_joint_names)), T800Cfg.env.num_actions)
        self.assertEqual(engineai_joint_names[:18], T800_CONTROLLED_JOINT_NAMES[:18])
        self.assertEqual(engineai_joint_names[18:], [
            "J18_SHOULDER_PITCH_R",
            "J19_SHOULDER_ROLL_R",
            "J20_SHOULDER_YAW_R",
            "J21_ELBOW_PITCH_R",
            "J22_ELBOW_YAW_R",
        ])

    def test_engineai_getup_config_uses_host_action_scale_order(self):
        expected_scale = [T800_ACTION_SCALE_BY_JOINT[name] for name in T800_CONTROLLED_JOINT_NAMES]
        self.assertEqual(self.config["action_scale"], expected_scale)
        self.assertEqual(len(self.config["joint_stiffness"]), T800Cfg.env.num_actions)
        self.assertEqual(len(self.config["joint_damping"]), T800Cfg.env.num_actions)
        self.assertEqual(len(self.config["default_joint_pos"]), T800Cfg.env.num_actions)
        self.assertTrue(self.config["clamp_joint_targets"])

    def test_engineai_getup_is_registered_for_runtime_selection(self):
        with ENGINEAI_MODE_CONFIG.open("r", encoding="utf-8") as handle:
            mode_config = yaml.safe_load(handle)
        with ENGINEAI_TASK_MOTION_CONFIG.open("r", encoding="utf-8") as handle:
            task_motion_config = yaml.safe_load(handle)

        for runtime in ("robot", "sim"):
            entries = mode_config["mode"][runtime]
            self.assertIn({"tag": "rl_getup_example", "scope": "rl_getup_example/default"}, entries)

        tasks = {task["motion"]: task for task in task_motion_config["tasks"]}
        self.assertIn("getup", tasks)
        getup_task = tasks["getup"]
        self.assertEqual(getup_task["period"], 0.01)
        self.assertEqual(getup_task["runner"][0]["name"], "rl_getup_example_runner")
        self.assertEqual(getup_task["runner"][0]["param_tag"], "rl_getup_example")
        self.assertIn("getup", tasks["passive"]["manual_transition"])
        self.assertIn("getup", tasks["pd_stand"]["manual_transition"])

    def test_engineai_cpp_contract_fixture_matches_host_layout_and_targets(self):
        compiler = shutil.which("g++") or shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("C++ compiler not available")
        self.assertTrue(CPP_CONTRACT_FIXTURE.exists(), CPP_CONTRACT_FIXTURE)
        self.assertTrue(ENGINEAI_GETUP_INCLUDE_DIR.exists(), ENGINEAI_GETUP_INCLUDE_DIR)

        with tempfile.TemporaryDirectory() as tmp_dir:
            binary_path = Path(tmp_dir) / "engineai_rl_getup_contract_check"
            compile_result = subprocess.run(
                [
                    compiler,
                    "-std=c++20",
                    "-O0",
                    "-I",
                    str(ENGINEAI_GETUP_INCLUDE_DIR),
                    str(CPP_CONTRACT_FIXTURE),
                    "-o",
                    str(binary_path),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(
                compile_result.returncode,
                0,
                f"compile stdout:\n{compile_result.stdout}\ncompile stderr:\n{compile_result.stderr}",
            )

            run_result = subprocess.run(
                [str(binary_path)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(
                run_result.returncode,
                0,
                f"run stdout:\n{run_result.stdout}\nrun stderr:\n{run_result.stderr}",
            )

    def test_export_manifest_records_verified_checkpoint_and_artifacts(self):
        if not ENGINEAI_POLICY_MANIFEST.exists():
            raise unittest.SkipTest(f"policy manifest not found: {ENGINEAI_POLICY_MANIFEST}")
        with ENGINEAI_POLICY_MANIFEST.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        checkpoint_path = Path(manifest["checkpoint_path"])
        checkpoint_match = re.match(r"model_(\d+)\.pt$", checkpoint_path.name)
        self.assertTrue(checkpoint_path.exists(), checkpoint_path)
        self.assertIsNotNone(checkpoint_match, checkpoint_path.name)
        self.assertEqual(manifest["checkpoint_file_iteration"], int(checkpoint_match.group(1)))
        self.assertEqual(manifest["input_dim"], 456)
        self.assertEqual(manifest["output_dim"], 23)
        self.assertEqual(manifest["history_order"], "oldest_to_newest; newest block is [380:456)")

        onnx = manifest["artifacts"]["onnx"]
        self.assertEqual(onnx["parity"]["status"], "verified")
        self.assertLessEqual(onnx["parity"]["max_abs_error_vs_pytorch"], 1e-4)

        mnn = manifest["artifacts"]["mnn_conversion"]
        self.assertEqual(mnn["status"], "converted")
        self.assertEqual(mnn["parity"]["status"], "verified")
        self.assertEqual(mnn["parity"]["input_shape"], [1, 456])
        self.assertEqual(mnn["parity"]["output_shape"], [1, 23])
        self.assertLessEqual(mnn["parity"]["max_abs_error_vs_pytorch"], 1e-4)
        self.assertTrue(Path(mnn["path"]).exists())
        self.assertFalse((Path(mnn["path"]).parent / "conversion.blocked.txt").exists())


if __name__ == "__main__":
    unittest.main()
