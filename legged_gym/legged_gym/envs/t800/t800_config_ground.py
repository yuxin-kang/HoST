from legged_gym.envs.g1.g1_config_ground import G1Cfg, G1CfgPPO


T800_SOURCE_URDF = "{LEGGED_GYM_ROOT_DIR}/resources/robots/t800/urdf/serial_t800.urdf"
T800_URDF = "{LEGGED_GYM_ROOT_DIR}/resources/robots/t800_stl/urdf/serial_t800_stl.urdf"
T800_STANDING_ROOT_HEIGHT = 1.04
T800_STANDING_HEAD_HEIGHT = 1.55

T800_DFS_JOINT_NAMES = [
    "J00_HIP_PITCH_L",
    "J01_HIP_ROLL_L",
    "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW",
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
    "J20_SHOULDER_PITCH_R",
    "J21_SHOULDER_ROLL_R",
    "J22_SHOULDER_YAW_R",
    "J23_ELBOW_PITCH_R",
    "J24_ELBOW_YAW_R",
    "J27_HEAD_PITCH",
    "J28_HEAD_YAW",
]

T800_ACTION_JOINT_NAMES = T800_DFS_JOINT_NAMES
T800_FIXED_JOINT_NAMES = []
T800_CONTROLLED_JOINT_NAMES = T800_ACTION_JOINT_NAMES
T800_HEAD_JOINT_NAMES = T800_DFS_JOINT_NAMES[23:]

T800_ACTION_SCALE_BY_JOINT = {
    "J00_HIP_PITCH_L": 0.5,
    "J01_HIP_ROLL_L": 0.2,
    "J02_HIP_YAW_L": 0.2,
    "J03_KNEE_PITCH_L": 0.5,
    "J04_ANKLE_PITCH_L": 0.5,
    "J05_ANKLE_ROLL_L": 0.2,
    "J06_HIP_PITCH_R": 0.5,
    "J07_HIP_ROLL_R": 0.2,
    "J08_HIP_YAW_R": 0.2,
    "J09_KNEE_PITCH_R": 0.5,
    "J10_ANKLE_PITCH_R": 0.5,
    "J11_ANKLE_ROLL_R": 0.2,
    "J12_TORSO_YAW": 0.2,
    "J13_SHOULDER_PITCH_L": 0.2,
    "J14_SHOULDER_ROLL_L": 0.2,
    "J15_SHOULDER_YAW_L": 0.05,
    "J16_ELBOW_PITCH_L": 0.2,
    "J17_ELBOW_YAW_L": 0.05,
    "J20_SHOULDER_PITCH_R": 0.2,
    "J21_SHOULDER_ROLL_R": 0.2,
    "J22_SHOULDER_YAW_R": 0.05,
    "J23_ELBOW_PITCH_R": 0.2,
    "J24_ELBOW_YAW_R": 0.05,
    "J27_HEAD_PITCH": 0.2,
    "J28_HEAD_YAW": 0.2,
}

T800_ARMATURE_BY_JOINT = {
    "J00_HIP_PITCH_L": 0.2427264,
    "J01_HIP_ROLL_L": 0.14110848,
    "J02_HIP_YAW_L": 0.0448737,
    "J03_KNEE_PITCH_L": 0.2427264,
    "J04_ANKLE_PITCH_L": 0.0354625,
    "J05_ANKLE_ROLL_L": 0.0354625,
    "J06_HIP_PITCH_R": 0.2427264,
    "J07_HIP_ROLL_R": 0.14110848,
    "J08_HIP_YAW_R": 0.0448737,
    "J09_KNEE_PITCH_R": 0.2427264,
    "J10_ANKLE_PITCH_R": 0.0354625,
    "J11_ANKLE_ROLL_R": 0.0354625,
    "J12_TORSO_YAW": 0.0448737,
    "J13_SHOULDER_PITCH_L": 0.0354625,
    "J14_SHOULDER_ROLL_L": 0.0354625,
    "J15_SHOULDER_YAW_L": 0.0354625,
    "J16_ELBOW_PITCH_L": 0.0354625,
    "J17_ELBOW_YAW_L": 0.00671625,
    "J20_SHOULDER_PITCH_R": 0.0354625,
    "J21_SHOULDER_ROLL_R": 0.0354625,
    "J22_SHOULDER_YAW_R": 0.0354625,
    "J23_ELBOW_PITCH_R": 0.0354625,
    "J24_ELBOW_YAW_R": 0.00671625,
    "J27_HEAD_PITCH": 0.00671625,
    "J28_HEAD_YAW": 0.00671625,
}

T800_DOF_FRICTION_BY_JOINT = {
    "J00_HIP_PITCH_L": 0.1,
    "J01_HIP_ROLL_L": 0.1,
    "J02_HIP_YAW_L": 0.05,
    "J03_KNEE_PITCH_L": 0.1,
    "J04_ANKLE_PITCH_L": 0.15,
    "J05_ANKLE_ROLL_L": 0.12,
    "J06_HIP_PITCH_R": 0.1,
    "J07_HIP_ROLL_R": 0.1,
    "J08_HIP_YAW_R": 0.05,
    "J09_KNEE_PITCH_R": 0.1,
    "J10_ANKLE_PITCH_R": 0.15,
    "J11_ANKLE_ROLL_R": 0.12,
    "J12_TORSO_YAW": 0.08,
    "J13_SHOULDER_PITCH_L": 0.12,
    "J14_SHOULDER_ROLL_L": 0.1,
    "J15_SHOULDER_YAW_L": 0.08,
    "J16_ELBOW_PITCH_L": 0.08,
    "J17_ELBOW_YAW_L": 0.06,
    "J20_SHOULDER_PITCH_R": 0.12,
    "J21_SHOULDER_ROLL_R": 0.1,
    "J22_SHOULDER_YAW_R": 0.08,
    "J23_ELBOW_PITCH_R": 0.08,
    "J24_ELBOW_YAW_R": 0.06,
    "J27_HEAD_PITCH": 0.05,
    "J28_HEAD_YAW": 0.05,
}

T800_DEFAULT_JOINT_ANGLES = {
    "J00_HIP_PITCH_L": -0.06,
    "J01_HIP_ROLL_L": 0.0,
    "J02_HIP_YAW_L": 0.0,
    "J03_KNEE_PITCH_L": 0.12,
    "J04_ANKLE_PITCH_L": -0.06,
    "J05_ANKLE_ROLL_L": 0.0,
    "J06_HIP_PITCH_R": -0.06,
    "J07_HIP_ROLL_R": 0.0,
    "J08_HIP_YAW_R": 0.0,
    "J09_KNEE_PITCH_R": 0.12,
    "J10_ANKLE_PITCH_R": -0.06,
    "J11_ANKLE_ROLL_R": 0.0,
    "J12_TORSO_YAW": 0.0,
    "J13_SHOULDER_PITCH_L": 0.0,
    "J14_SHOULDER_ROLL_L": 0.15,
    "J15_SHOULDER_YAW_L": 0.0,
    "J16_ELBOW_PITCH_L": -0.25,
    "J17_ELBOW_YAW_L": 0.0,
    "J20_SHOULDER_PITCH_R": 0.0,
    "J21_SHOULDER_ROLL_R": -0.15,
    "J22_SHOULDER_YAW_R": 0.0,
    "J23_ELBOW_PITCH_R": -0.25,
    "J24_ELBOW_YAW_R": 0.0,
    "J27_HEAD_PITCH": 0.0,
    "J28_HEAD_YAW": 0.0,
}

T800_TARGET_JOINT_ANGLES = {
    **T800_DEFAULT_JOINT_ANGLES,
    "J14_SHOULDER_ROLL_L": 0.3,
    "J16_ELBOW_PITCH_L": 0.0,
    "J21_SHOULDER_ROLL_R": -0.3,
    "J23_ELBOW_PITCH_R": 0.0,
}


class T800Cfg(G1Cfg):
    class init_state(G1Cfg.init_state):
        pos = [0.0, 0.0, T800_STANDING_ROOT_HEIGHT]
        rot = [0.0, 0.0, 0.0, 1.0]
        target_joint_angles = T800_TARGET_JOINT_ANGLES
        default_joint_angles = T800_DEFAULT_JOINT_ANGLES

    class env(G1Cfg.env):
        num_one_step_observations = 82
        num_actions = 25
        num_dofs = 25
        num_actor_history = 6
        num_observations = num_actor_history * num_one_step_observations

    class control(G1Cfg.control):
        action_scale = T800_ACTION_SCALE_BY_JOINT
        action_rescale = 1.0
        stiffness = {
            "HIP_PITCH": 180,
            "HIP_ROLL": 100,
            "HIP_YAW": 100,
            "KNEE_PITCH": 180,
            "ANKLE": 40,
            "TORSO_YAW": 100,
            "SHOULDER": 40,
            "ELBOW_PITCH": 40,
            "ELBOW_YAW": 50,
            "HEAD": 50,
        }
        damping = {
            "HIP_PITCH": 5,
            "HIP_ROLL": 3,
            "HIP_YAW": 3,
            "KNEE_PITCH": 5,
            "ANKLE": 0.3,
            "TORSO_YAW": 3,
            "SHOULDER": 0.3,
            "ELBOW_PITCH": 0.3,
            "ELBOW_YAW": 0.3,
            "HEAD": 0.3,
        }

    class rewards(G1Cfg.rewards):
        base_height_target = T800_STANDING_ROOT_HEIGHT
        target_head_height = T800_STANDING_HEAD_HEIGHT
        target_base_height_phase1 = 0.85
        target_base_height_phase2 = 0.90
        target_base_height_phase3 = 0.98

    class asset(G1Cfg.asset):
        file = T800_URDF
        name = "t800"
        left_foot_name = "LINK_FOOT_L"
        right_foot_name = "LINK_FOOT_R"
        left_knee_name = "LINK_KNEE_PITCH_L"
        right_knee_name = "LINK_KNEE_PITCH_R"
        foot_name = "LINK_FOOT"
        penalize_contacts_on = ["LINK_ELBOW", "LINK_SHOULDER", "LINK_TORSO", "LINK_KNEE", "LINK_HIP"]
        terminate_after_contacts_on = []

        left_shoulder_name = "LINK_SHOULDER_PITCH_L"
        right_shoulder_name = "LINK_SHOULDER_PITCH_R"

        controlled_joint_names = T800_CONTROLLED_JOINT_NAMES
        fixed_joint_names = T800_FIXED_JOINT_NAMES

        left_leg_joints = [
            "J00_HIP_PITCH_L",
            "J01_HIP_ROLL_L",
            "J02_HIP_YAW_L",
            "J03_KNEE_PITCH_L",
            "J04_ANKLE_PITCH_L",
            "J05_ANKLE_ROLL_L",
        ]
        right_leg_joints = [
            "J06_HIP_PITCH_R",
            "J07_HIP_ROLL_R",
            "J08_HIP_YAW_R",
            "J09_KNEE_PITCH_R",
            "J10_ANKLE_PITCH_R",
            "J11_ANKLE_ROLL_R",
        ]
        left_hip_joints = ["J02_HIP_YAW_L"]
        right_hip_joints = ["J08_HIP_YAW_R"]
        left_hip_roll_joints = ["J01_HIP_ROLL_L"]
        right_hip_roll_joints = ["J07_HIP_ROLL_R"]
        left_hip_pitch_joints = ["J00_HIP_PITCH_L"]
        right_hip_pitch_joints = ["J06_HIP_PITCH_R"]

        left_shoulder_roll_joints = ["J14_SHOULDER_ROLL_L"]
        right_shoulder_roll_joints = ["J21_SHOULDER_ROLL_R"]

        left_knee_joints = ["J03_KNEE_PITCH_L"]
        right_knee_joints = ["J09_KNEE_PITCH_R"]

        left_arm_joints = [
            "J13_SHOULDER_PITCH_L",
            "J14_SHOULDER_ROLL_L",
            "J15_SHOULDER_YAW_L",
            "J16_ELBOW_PITCH_L",
            "J17_ELBOW_YAW_L",
        ]
        right_arm_joints = [
            "J20_SHOULDER_PITCH_R",
            "J21_SHOULDER_ROLL_R",
            "J22_SHOULDER_YAW_R",
            "J23_ELBOW_PITCH_R",
            "J24_ELBOW_YAW_R",
        ]
        waist_joints = ["J12_TORSO_YAW"]
        knee_joints = ["J03_KNEE_PITCH_L", "J09_KNEE_PITCH_R"]
        ankle_joints = ["J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R"]

        keyframe_name = "keyframe"
        head_name = "LINK_HEAD_YAW"

        trunk_names = ["LINK_BASE", "LINK_TORSO_YAW"]
        base_name = "LINK_TORSO_YAW"
        mass_link_name = "LINK_TORSO_YAW"

        left_upper_body_names = ["LINK_SHOULDER_PITCH_L", "LINK_ELBOW_PITCH_L"]
        right_upper_body_names = ["LINK_SHOULDER_PITCH_R", "LINK_ELBOW_PITCH_R"]
        left_lower_body_names = ["LINK_HIP_PITCH_L", "LINK_ANKLE_ROLL_L", "LINK_KNEE_PITCH_L"]
        right_lower_body_names = ["LINK_HIP_PITCH_R", "LINK_ANKLE_ROLL_R", "LINK_KNEE_PITCH_R"]

        left_ankle_names = ["LINK_ANKLE_ROLL_L"]
        right_ankle_names = ["LINK_ANKLE_ROLL_R"]

        collapse_fixed_joints = False
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False
        linear_damping = 0.0
        angular_damping = 0.0
        armature = 0.0
        dof_armature = T800_ARMATURE_BY_JOINT
        dof_friction = T800_DOF_FRICTION_BY_JOINT

    class sim(G1Cfg.sim):
        class physx(G1Cfg.sim.physx):
            num_velocity_iterations = 4


class T800CfgPPO(G1CfgPPO):
    class runner(G1CfgPPO.runner):
        run_name = ""
        experiment_name = "t800_host_ground"
