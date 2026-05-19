import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import isaacgym  # noqa: F401

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.t800.t800_config_ground import (
    T800Cfg,
    T800CfgPPO,
    T800_ACTION_JOINT_NAMES,
    T800_ACTION_SCALE_BY_JOINT,
    T800_ARMATURE_BY_JOINT,
    T800_CONTROLLED_JOINT_NAMES,
    T800_DFS_JOINT_NAMES,
    T800_DOF_FRICTION_BY_JOINT,
    T800_FIXED_JOINT_NAMES,
    T800_HEAD_JOINT_NAMES,
    T800_STANDING_ROOT_HEIGHT,
)


def _asset_path():
    return Path(T800Cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))


def _urdf_root():
    return ET.parse(_asset_path()).getroot()


def _movable_urdf_joints():
    root = _urdf_root()
    return [joint.attrib["name"] for joint in root.findall("joint") if joint.attrib.get("type") != "fixed"]


class T800HostConfigTest(unittest.TestCase):
    def test_policy_interface_trains_all_t800_dfs_joints(self):
        self.assertEqual(T800Cfg.env.num_actions, 25)
        self.assertEqual(T800Cfg.env.num_dofs, 25)
        self.assertEqual(T800Cfg.env.num_one_step_observations, 82)
        self.assertEqual(T800Cfg.env.num_observations, 492)
        self.assertEqual(T800CfgPPO.runner.experiment_name, "t800_host_ground")

    def test_t800_head_joints_are_included_in_policy_io(self):
        movable_joints = _movable_urdf_joints()
        self.assertEqual(len(movable_joints), 25)
        self.assertEqual(T800_DFS_JOINT_NAMES, movable_joints)
        self.assertEqual(T800_ACTION_JOINT_NAMES, T800_DFS_JOINT_NAMES)
        self.assertEqual(T800_FIXED_JOINT_NAMES, [])
        self.assertEqual(T800_CONTROLLED_JOINT_NAMES, movable_joints)
        self.assertEqual(T800_HEAD_JOINT_NAMES, movable_joints[23:])
        self.assertEqual(T800Cfg.asset.controlled_joint_names, T800_CONTROLLED_JOINT_NAMES)
        self.assertEqual(T800Cfg.asset.fixed_joint_names, [])

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
        self.assertEqual(set(T800_ACTION_SCALE_BY_JOINT), set(_movable_urdf_joints()))
        self.assertEqual(T800Cfg.control.action_scale, T800_ACTION_SCALE_BY_JOINT)
        ordered_scale = [T800Cfg.control.action_scale[name] for name in T800_CONTROLLED_JOINT_NAMES]
        self.assertEqual(len(ordered_scale), T800Cfg.env.num_actions)
        self.assertEqual(ordered_scale[0], 0.5)
        self.assertEqual(ordered_scale[17], 0.05)
        self.assertEqual(ordered_scale[23], 0.2)
        self.assertEqual(ordered_scale[24], 0.2)

    def test_t800_pd_gains_match_540_and_zhiquan_t800_source(self):
        expected_stiffness = {
            "HIP_PITCH": 180.0,
            "HIP_ROLL": 100.0,
            "HIP_YAW": 100.0,
            "KNEE_PITCH": 180.0,
            "ANKLE": 40.0,
            "TORSO_YAW": 100.0,
            "SHOULDER": 40.0,
            "ELBOW_PITCH": 40.0,
            "ELBOW_YAW": 50.0,
            "HEAD": 50.0,
        }
        expected_damping = {
            "HIP_PITCH": 5.0,
            "HIP_ROLL": 3.0,
            "HIP_YAW": 3.0,
            "KNEE_PITCH": 5.0,
            "ANKLE": 0.3,
            "TORSO_YAW": 3.0,
            "SHOULDER": 0.3,
            "ELBOW_PITCH": 0.3,
            "ELBOW_YAW": 0.3,
            "HEAD": 0.3,
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

    def test_t800_standing_root_height_keeps_default_mesh_above_ground(self):
        self.assertEqual(T800Cfg.init_state.pos[2], T800_STANDING_ROOT_HEIGHT)
        self.assertGreater(T800_STANDING_ROOT_HEIGHT, 1.018)


if __name__ == "__main__":
    unittest.main()
