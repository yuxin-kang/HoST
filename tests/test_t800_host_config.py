import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import isaacgym  # noqa: F401

from legged_gym import LEGGED_GYM_ROOT_DIR
import legged_gym.envs  # noqa: F401
from legged_gym.envs.base import host_ground
from legged_gym.envs.h1.h1_config_ground import H1Cfg, H1CfgPPO
from legged_gym.envs.t800.t800_config_ground import (
    T800Cfg,
    T800CfgPPO,
    T800_ACTION_JOINT_NAMES,
    T800_ACTION_SCALE_BY_JOINT,
    T800_ARMATURE_BY_JOINT,
    T800_CONTROLLED_JOINT_NAMES,
    T800_DEFAULT_JOINT_ANGLES,
    T800_DFS_JOINT_NAMES,
    T800_DOF_FRICTION_BY_JOINT,
    T800_FIXED_JOINT_NAMES,
    T800_GETUP_ROOT_ROT,
    T800_HEAD_JOINT_NAMES,
    T800_STANDING_HEAD_HEIGHT,
    T800_STANDING_ROOT_HEIGHT,
    T800_TARGET_JOINT_ANGLES,
)
from legged_gym.utils.task_registry import task_registry


def _asset_path():
    return Path(T800Cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))


def _urdf_root():
    return ET.parse(_asset_path()).getroot()


def _movable_urdf_joints():
    root = _urdf_root()
    return [joint.attrib["name"] for joint in root.findall("joint") if joint.attrib.get("type") != "fixed"]


def _urdf_links():
    return [link.attrib["name"] for link in _urdf_root().findall("link")]


class T800HostConfigTest(unittest.TestCase):
    def test_t800_inherits_from_h1_and_keeps_task_registry_entry(self):
        self.assertTrue(issubclass(T800Cfg, H1Cfg))
        self.assertTrue(issubclass(T800CfgPPO, H1CfgPPO))

        env_cfg, train_cfg = task_registry.get_cfgs("t800_host_ground")
        self.assertIsInstance(env_cfg, T800Cfg)
        self.assertIsInstance(train_cfg, T800CfgPPO)
        self.assertEqual(task_registry.get_task_class("t800_host_ground").__name__, "LeggedRobot")
        self.assertEqual(train_cfg.runner.experiment_name, "t800_host_ground")

    def test_policy_interface_excludes_t800_head_joints(self):
        self.assertEqual(T800Cfg.env.num_actions, 23)
        self.assertEqual(T800Cfg.env.num_dofs, 23)
        self.assertEqual(T800Cfg.env.num_one_step_observations, 76)
        self.assertEqual(T800Cfg.env.num_observations, 456)
        self.assertEqual(T800CfgPPO.runner.experiment_name, "t800_host_ground")

    def test_t800_head_joints_stay_movable_but_uncontrolled(self):
        movable_joints = _movable_urdf_joints()
        self.assertEqual(len(movable_joints), 25)
        self.assertEqual(T800_DFS_JOINT_NAMES, movable_joints)
        self.assertEqual(T800_ACTION_JOINT_NAMES, movable_joints[:23])
        self.assertEqual(T800_FIXED_JOINT_NAMES, [])
        self.assertEqual(T800_CONTROLLED_JOINT_NAMES, movable_joints[:23])
        self.assertEqual(T800_HEAD_JOINT_NAMES, movable_joints[23:])
        self.assertEqual(T800Cfg.asset.controlled_joint_names, T800_CONTROLLED_JOINT_NAMES)
        self.assertEqual(T800Cfg.asset.fixed_joint_names, [])
        for head_joint in T800_HEAD_JOINT_NAMES:
            self.assertNotIn(head_joint, T800_ACTION_JOINT_NAMES)
            self.assertNotIn(head_joint, T800_CONTROLLED_JOINT_NAMES)
            self.assertNotIn(head_joint, T800_ACTION_SCALE_BY_JOINT)
            self.assertIn(head_joint, T800_DEFAULT_JOINT_ANGLES)
            self.assertIn(head_joint, T800_TARGET_JOINT_ANGLES)
            self.assertIn(head_joint, T800_ARMATURE_BY_JOINT)
            self.assertIn(head_joint, T800_DOF_FRICTION_BY_JOINT)

    def test_all_t800_movable_joints_have_default_and_target_angles(self):
        movable_joints = _movable_urdf_joints()
        self.assertEqual(set(T800Cfg.init_state.default_joint_angles), set(movable_joints))
        self.assertEqual(set(T800Cfg.init_state.target_joint_angles), set(movable_joints))

    def test_t800_uses_generated_stl_asset_like_engineai_gym_reference(self):
        asset_path = _asset_path()
        self.assertIn("t800_stl", str(asset_path))
        self.assertTrue(asset_path.exists())

        mesh_filenames = [mesh.attrib["filename"] for mesh in _urdf_root().findall(".//mesh")]
        self.assertTrue(mesh_filenames)
        for mesh_filename in mesh_filenames:
            self.assertTrue(mesh_filename.endswith(".stl"), mesh_filename)
            self.assertTrue((asset_path.parent / mesh_filename).resolve().exists(), mesh_filename)

    def test_t800_action_scale_matches_controlled_joint_order(self):
        self.assertEqual(set(T800_ACTION_SCALE_BY_JOINT), set(T800_CONTROLLED_JOINT_NAMES))
        self.assertIs(T800Cfg.control.action_scale, T800_ACTION_SCALE_BY_JOINT)
        self.assertFalse(hasattr(T800Cfg.control, "action_rescale"))
        ordered_scale = [T800Cfg.control.action_scale[name] for name in T800_CONTROLLED_JOINT_NAMES]
        self.assertEqual(len(ordered_scale), T800Cfg.env.num_actions)
        self.assertEqual(ordered_scale[0], 0.5)
        self.assertEqual(ordered_scale[17], 0.05)
        self.assertEqual(ordered_scale[-1], 0.05)

    def test_t800_pd_gains_match_current_getup_tuning(self):
        expected_stiffness = {
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
        expected_damping = {
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
        self.assertEqual(T800Cfg.control.stiffness, expected_stiffness)
        self.assertEqual(T800Cfg.control.damping, expected_damping)

    def test_t800_actuator_physics_matches_540_and_zhiquan_t800_source(self):
        movable_joints = set(_movable_urdf_joints())
        self.assertEqual(set(T800_ARMATURE_BY_JOINT), movable_joints)
        self.assertEqual(set(T800_DOF_FRICTION_BY_JOINT), movable_joints)
        self.assertEqual(T800Cfg.asset.dof_armature, T800_ARMATURE_BY_JOINT)
        self.assertEqual(T800Cfg.asset.dof_friction, T800_DOF_FRICTION_BY_JOINT)
        self.assertEqual(T800Cfg.asset.linear_damping, 0.0)
        self.assertEqual(T800Cfg.asset.angular_damping, 0.0)
        self.assertEqual(T800Cfg.asset.armature, 0.0)
        self.assertEqual(T800Cfg.sim.physx.num_position_iterations, 8)
        self.assertEqual(T800Cfg.sim.physx.num_velocity_iterations, 4)
        self.assertAlmostEqual(T800_ARMATURE_BY_JOINT["J00_HIP_PITCH_L"], 0.2427264)
        self.assertAlmostEqual(T800_ARMATURE_BY_JOINT["J17_ELBOW_YAW_L"], 0.00671625)
        self.assertAlmostEqual(T800_DOF_FRICTION_BY_JOINT["J04_ANKLE_PITCH_L"], 0.15)

    def test_t800_auxiliary_ankle_markers_restore_ground_parallel_sampling(self):
        link_names = _urdf_links()
        left_matches = [name for name in link_names if "LINK_ANKLE_ROLL_L" in name]
        right_matches = [name for name in link_names if "LINK_ANKLE_ROLL_R" in name]
        expected_left_markers = {
            f"auxiliary_LINK_ANKLE_ROLL_L_link{i}" for i in range(1, 5)
        }
        expected_right_markers = {
            f"auxiliary_LINK_ANKLE_ROLL_R_link{i}" for i in range(1, 5)
        }
        self.assertEqual(set(left_matches), {"LINK_ANKLE_ROLL_L"} | expected_left_markers)
        self.assertEqual(set(right_matches), {"LINK_ANKLE_ROLL_R"} | expected_right_markers)
        self.assertEqual(
            set(host_ground._select_ankle_marker_names(link_names, T800Cfg.asset.left_ankle_names)),
            expected_left_markers,
        )
        self.assertEqual(
            set(host_ground._select_ankle_marker_names(link_names, T800Cfg.asset.right_ankle_names)),
            expected_right_markers,
        )

        joints_by_child = {
            joint.find("child").attrib["link"]: joint
            for joint in _urdf_root().findall("joint")
            if joint.find("child") is not None
        }
        expected_origins = {
            "0.15 0.05 -0.06453",
            "0.15 -0.05 -0.06453",
            "-0.11 0.05 -0.06453",
            "-0.11 -0.05 -0.06453",
        }
        mixed_heights = [0.0] + [-0.06453 * 10] * 4
        mixed_mean = sum(mixed_heights) / len(mixed_heights)
        mixed_variance = sum((height - mixed_mean) ** 2 for height in mixed_heights) / (len(mixed_heights) - 1)
        self.assertGreater(mixed_variance, 0.05)

        for side in ("L", "R"):
            origins = set()
            selected_heights = []
            for index in range(1, 5):
                marker_name = f"auxiliary_LINK_ANKLE_ROLL_{side}_link{index}"
                joint = joints_by_child[marker_name]
                self.assertEqual(joint.attrib["type"], "fixed")
                self.assertEqual(joint.attrib["dont_collapse"], "true")
                self.assertEqual(joint.find("parent").attrib["link"], f"LINK_ANKLE_ROLL_{side}")
                origin_xyz = joint.find("origin").attrib["xyz"]
                origins.add(origin_xyz)
                selected_heights.append(float(origin_xyz.split()[2]) * 10)
            self.assertEqual(origins, expected_origins)
            self.assertEqual(len(set(selected_heights)), 1)

    def test_t800_reset_pose_uses_getup_initialization(self):
        self.assertEqual(T800_GETUP_ROOT_ROT, [0.0, -1.0, 0.0, 1.0])
        self.assertEqual(T800Cfg.init_state.pos, [0.0, 0.0, 0.5])
        self.assertEqual(T800Cfg.init_state.rot, T800_GETUP_ROOT_ROT)
        self.assertLess(T800Cfg.init_state.pos[2], T800_STANDING_ROOT_HEIGHT)
        self.assertGreater(T800_STANDING_ROOT_HEIGHT, 1.018)

    def test_t800_getup_curriculum_rewards_and_regularization_are_locked(self):
        self.assertAlmostEqual(T800_STANDING_ROOT_HEIGHT, 1.037)
        self.assertAlmostEqual(T800_STANDING_HEAD_HEIGHT, 1.567)

        self.assertTrue(T800Cfg.curriculum.pull_force)
        self.assertEqual(T800Cfg.curriculum.force, 1000)
        self.assertAlmostEqual(T800Cfg.curriculum.threshold_height, 1.42)
        self.assertTrue(T800Cfg.curriculum.no_orientation)
        self.assertEqual(T800Cfg.curriculum.dof_vel_limit, 300)
        self.assertEqual(T800Cfg.curriculum.base_vel_limit, 20)

        self.assertAlmostEqual(T800Cfg.rewards.base_height_target, T800_STANDING_ROOT_HEIGHT)
        self.assertAlmostEqual(T800Cfg.rewards.target_head_height, T800_STANDING_HEAD_HEIGHT)
        self.assertAlmostEqual(T800Cfg.rewards.target_head_margin, 1.0)
        self.assertAlmostEqual(T800Cfg.rewards.target_base_height_phase1, 0.65)
        self.assertAlmostEqual(T800Cfg.rewards.target_base_height_phase2, 0.65)
        self.assertAlmostEqual(T800Cfg.rewards.target_base_height_phase3, 0.9)

        self.assertAlmostEqual(T800Cfg.constraints.scales.regu_action_rate, -0.01)
        self.assertAlmostEqual(T800Cfg.constraints.scales.regu_smoothness, -0.01)
        self.assertAlmostEqual(T800Cfg.constraints.scales.regu_dof_pos_limits, -100.0)

    def test_t800_collision_penalty_is_enabled_for_non_feet_contacts(self):
        self.assertAlmostEqual(T800Cfg.constraints.scales.style_collision, -1.0)
        self.assertTrue(hasattr(host_ground.LeggedRobot, "_reward_collision"))
        self.assertEqual(
            T800Cfg.asset.penalize_contacts_on,
            ["LINK_ELBOW", "LINK_SHOULDER", "LINK_TORSO", "LINK_KNEE", "LINK_HIP"],
        )
        for contact_name in T800Cfg.asset.penalize_contacts_on:
            self.assertNotIn("FOOT", contact_name)
            self.assertNotIn("ANKLE", contact_name)


if __name__ == "__main__":
    unittest.main()
