import numpy as np
from mani_skill.agents.registration import register_agent
from robots.floating_dex_hand_base import FloatingDexHandBase, HandSpec


ABILITY_TIP_URDF_CONFIG = dict(
    _materials=dict(
        tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
    ),
    link={
        "thumb_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "index_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "middle_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "ring_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        # "pinky_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
    },
)


ABILITY_SPEC = HandSpec(
    uid="floating_ability_hand_right",
    urdf_path="./assets/robot_hand_urdf/ability_hand_urdf/ability_hand_right_floating.urdf",
    finger_joint_names=[
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
    ],
    tip_link_names=[
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
    ],
    palm_link_name="thumb_base",
    urdf_config=ABILITY_TIP_URDF_CONFIG,
    base_qpos_default=np.array([0.0, 0.0, 0.3, 0.0, np.pi, 0.0], dtype=float),
)


@register_agent()
class FloatingAbilityHandRight(FloatingDexHandBase):
    uid = ABILITY_SPEC.uid
    HAND_SPEC = ABILITY_SPEC


