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
class FloatingAllegroHandRight(BaseAgent):
    uid = "floating_allegro_hand_right"
    urdf_path = f"./assets/robot_hand_urdf/allegro_hand_urdf/allegro_hand_right_floating.urdf"
    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
        ),
        link={
            "link_3_0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_7_0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_11_0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_15_0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
        },
    )
    disable_self_collisions = True
    # you could model all of the fingers and disable certain impossible self collisions that occur
    # but it is simpler and faster to just disable all self collisions. It is highly unlikely the hand self-collides to begin with
    # due to the design of the hand

    root_joint_names = [
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_axis_joint",
        "root_x_rot_joint",
        "root_y_rot_joint",
        "root_z_rot_joint",
    ]

    keyframes = dict(
        palm_side=Keyframe(
            qpos=np.zeros(6 + 16),
            pose=sapien.Pose([0, 0, 0.5], q=[1, 0, 0, 0]),
        ),
        palm_up=Keyframe(
            qpos=np.zeros(6 + 16),
            pose=sapien.Pose([0, 0, 0.5], q=[-0.707, 0, 0.707, 0]),
        ),
        palm_down=Keyframe(
            qpos=np.zeros(6 + 16),
            pose=sapien.Pose([0, 0, 0.5], q=[0.707, 0, 0.707, 0]),
        ),
    )

    def __init__(self, *args, **kwargs):
        self.joint_names = [
            "joint_0_0",
            "joint_1_0",
            "joint_2_0",
            "joint_3_0",
            "joint_4_0",
            "joint_5_0",
            "joint_6_0",
            "joint_7_0",
            "joint_8_0",
            "joint_9_0",
            "joint_10_0",
            "joint_11_0",
            "joint_12_0",
            "joint_13_0",
            "joint_14_0",
            "joint_15_0",
        ]

        # self.joint_stiffness = 4e2
        # self.joint_damping = 1e1
        self.joint_force_limit = 5e1
        self.finger_joint_stiffness = self.float_joint_stiffness = 1e4
        self.finger_joint_damping = self.float_joint_damping = 1e3
        self.finger_joint_force_limit = self.float_joint_force_limit = 1e2

        # Order: thumb finger, index finger, middle finger, ring finger
        self.tip_link_names = [
            "link_15_0_tip",
            "link_3_0_tip",
            "link_7_0_tip",
            "link_11_0_tip",
        ]

        self.palm_link_name = "palm"
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
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
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

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent.
        """
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
        """
        Get the tip pose for each of the finger, four fingers in total
        """
        tip_poses = [
            vectorize_pose(link.pose, device=self.device) for link in self.tip_links
        ]
        return torch.stack(tip_poses, dim=-2)

    @property
    def palm_pose(self):
        """
        Get the palm pose for allegro hand
        """
        return vectorize_pose(self.palm_link.pose, device=self.device)