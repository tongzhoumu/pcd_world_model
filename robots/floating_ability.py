from copy import deepcopy
from typing import List

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import vectorize_pose


@register_agent()
class FloatingAbilityHandRight(BaseAgent):
    uid = "floating_ability_hand_right"
    urdf_path = f"./assets/robot_hand_urdf/ability_hand_urdf/ability_hand_right_floating.urdf"
    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
        ),
        link={
            # fingertips
            "thumb_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "index_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "middle_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "ring_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            # optionally include pinky tip if needed
            # "pinky_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        },
    )
    disable_self_collisions = True

    root_joint_names = [
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_axis_joint",
        "root_x_rot_joint",
        "root_y_rot_joint",
        "root_z_rot_joint",
    ]

    keyframes = dict(
        fingers_down=Keyframe(
            qpos=np.concatenate([
                np.array([0, 0, 0.3, 0, np.pi, 0]), np.zeros(10)
            ]),
            pose=sapien.Pose([0, 0, 0], q=[1, 0, 0, 0]),
        ),
    )

    def __init__(self, *args, **kwargs):
        # Ability hand DOFs: 10 finger joints (2 per finger across 5 fingers with some mimics)
        self.joint_names = [
            "thumb_q1",
            "thumb_q2",
            "index_q1",
            "index_q2",
            "middle_q1",
            "middle_q2",
            "ring_q1",
            "ring_q2",
            "pinky_q1",
            "pinky_q2",
        ]

        self.joint_force_limit = 5e1
        self.finger_joint_stiffness = self.float_joint_stiffness = 1e4
        self.finger_joint_damping = self.float_joint_damping = 1e3
        self.finger_joint_force_limit = self.float_joint_force_limit = 1e2

        # Order: thumb, index, middle, ring
        self.tip_link_names = [
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            # optionally include pinky tip if desired
            # "pinky_tip",
        ]

        # thumb_base corresponds to palm mesh in the URDF
        self.palm_link_name = "thumb_base"
        super().__init__(*args, **kwargs)

    def _after_init(self):
        self.tip_links: List[sapien.Entity] = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.tip_link_names
        )
        self.palm_link: sapien.Entity = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.palm_link_name
        )

    @property
    def _controller_configs(self):
        float_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=self.root_joint_names,
            lower=None,
            upper=None,
            stiffness=self.float_joint_stiffness,
            damping=self.float_joint_damping,
            force_limit=self.float_joint_force_limit,
            normalize_action=False,
        )
        joint_pos = PDJointPosControllerConfig(
            self.joint_names,
            None,
            None,
            self.finger_joint_stiffness,
            self.finger_joint_damping,
            self.finger_joint_force_limit,
            normalize_action=False,
        )
        joint_delta_pos = PDJointPosControllerConfig(
            self.joint_names,
            -0.1,
            0.1,
            self.finger_joint_stiffness,
            self.finger_joint_damping,
            self.finger_joint_force_limit,
            use_delta=True,
        )
        joint_target_delta_pos = deepcopy(joint_delta_pos)
        joint_target_delta_pos.use_target = True

        float_pd_joint_delta_pos = deepcopy(float_pd_joint_pos)
        float_pd_joint_delta_pos.use_delta = True
        float_pd_joint_delta_pos.normalize_action = True
        float_pd_joint_delta_pos.lower = -0.1
        float_pd_joint_delta_pos.upper = 0.1

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                root=float_pd_joint_delta_pos,
                fingers=joint_delta_pos,
            ),
            pd_joint_pos=dict(
                root=float_pd_joint_pos,
                fingers=joint_pos,
            ),
        )

        return deepcopy_dict(controller_configs)

    def get_proprioception(self):
        obs = super().get_proprioception()
        obs.update(
            {
                "palm_pose": self.palm_pose,
                "tip_poses": self.tip_poses.reshape(-1, len(self.tip_links) * 7),
            }
        )
        return obs

    @property
    def tip_poses(self):
        tip_poses = [
            vectorize_pose(link.pose, device=self.device) for link in self.tip_links
        ]
        return torch.stack(tip_poses, dim=-2)

    @property
    def palm_pose(self):
        return vectorize_pose(self.palm_link.pose, device=self.device)


