from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import vectorize_pose


@dataclass
class HandSpec:
    uid: str
    urdf_path: str
    finger_joint_names: List[str]
    tip_link_names: List[str]
    palm_link_name: str
    urdf_config: Dict
    base_qpos_default: np.ndarray  # length 6: [x, y, z, rx, ry, rz]


class FloatingDexHandBase(BaseAgent):
    disable_self_collisions = True

    # 6-DoF floating base joints shared by all dex hands
    root_joint_names = [
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_axis_joint",
        "root_x_rot_joint",
        "root_y_rot_joint",
        "root_z_rot_joint",
    ]

    # default controller parameters; subclasses can override before calling super().__init__
    finger_joint_stiffness = 1e4
    finger_joint_damping = 1e3
    finger_joint_force_limit = 1e2
    float_joint_stiffness = 1e4
    float_joint_damping = 1e3
    float_joint_force_limit = 1e2

    HAND_SPEC: HandSpec = None

    def __init__(self, *args, **kwargs):
        assert self.HAND_SPEC is not None, "HAND_SPEC must be set in subclass"

        # wire spec
        self.uid = self.HAND_SPEC.uid
        self.urdf_path = self.HAND_SPEC.urdf_path
        self.urdf_config = self.HAND_SPEC.urdf_config

        # finger/joint naming
        self.joint_names = list(self.HAND_SPEC.finger_joint_names)
        self.tip_link_names = list(self.HAND_SPEC.tip_link_names)
        self.palm_link_name = self.HAND_SPEC.palm_link_name

        super().__init__(*args, **kwargs)

        # keyframes constructed from spec
        base_qpos = np.array(self.HAND_SPEC.base_qpos_default, dtype=float)
        assert base_qpos.shape == (6,)
        fingers_zero = np.zeros(len(self.joint_names), dtype=float)
        fingers_down_qpos = np.concatenate([base_qpos, fingers_zero])

        self.keyframes = dict(
            fingers_down=Keyframe(
                qpos=fingers_down_qpos,
                pose=sapien.Pose([0, 0, 0], q=[1, 0, 0, 0]),
            ),
        )

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


