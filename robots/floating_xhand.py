import numpy as np
from mani_skill.agents.registration import register_agent
from robots.floating_dex_hand_base import FloatingDexHandBase, HandSpec


XHAND_TIP_URDF_CONFIG = dict(
    _materials=dict(
        tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
    ),
    link={
        "right_hand_thumb_rota_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "right_hand_index_rota_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "right_hand_mid_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "right_hand_ring_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "right_hand_pinky_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
    },
)


XHAND_SPEC = HandSpec(
    uid="floating_xhand_right",
    urdf_path="./assets/robot_hand_urdf/xhand_urdf/xhand_right_floating.urdf",
    finger_joint_names=[
        # thumb
        "right_hand_thumb_bend_joint",
        "right_hand_thumb_rota_joint1",
        "right_hand_thumb_rota_joint2",
        # index
        "right_hand_index_bend_joint",
        "right_hand_index_joint1",
        "right_hand_index_joint2",
        # middle
        "right_hand_mid_joint1",
        "right_hand_mid_joint2",
        # ring
        "right_hand_ring_joint1",
        "right_hand_ring_joint2",
        # pinky
        "right_hand_pinky_joint1",
        "right_hand_pinky_joint2",
    ],
    tip_link_names=[
        "right_hand_thumb_rota_tip",
        "right_hand_index_rota_tip",
        "right_hand_mid_tip",
        "right_hand_ring_tip",
        "right_hand_pinky_tip",
    ],
    palm_link_name="right_hand_link",
    urdf_config=XHAND_TIP_URDF_CONFIG,
    base_qpos_default=np.array([0.0, 0.0, 0.3, 0.0, np.pi, 0.0], dtype=float),
)


@register_agent()
class FloatingXHandRight(FloatingDexHandBase):
    uid = XHAND_SPEC.uid
    HAND_SPEC = XHAND_SPEC


