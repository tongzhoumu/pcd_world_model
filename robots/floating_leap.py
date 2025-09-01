import numpy as np
from mani_skill.agents.registration import register_agent
from robots.floating_dex_hand_base import FloatingDexHandBase, HandSpec


LEAP_TIP_URDF_CONFIG = dict(
    _materials=dict(
        tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
    ),
    link={
        "thumb_tip_head": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "index_tip_head": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "middle_tip_head": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "ring_tip_head": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
    },
)


LEAP_SPEC = HandSpec(
    uid="floating_leap_hand_right",
    urdf_path="./assets/robot_hand_urdf/leap_hand/leap_hand_right_floating.urdf",
    # 16 finger joints including thumb (named numerically 0-15 in the URDF)
    finger_joint_names=[str(i) for i in range(16)],
    tip_link_names=[
        "thumb_tip_head",
        "index_tip_head",
        "middle_tip_head",
        "ring_tip_head",
    ],
    palm_link_name="palm_lower",
    urdf_config=LEAP_TIP_URDF_CONFIG,
    base_qpos_default=np.array([0.0, 0.0, 0.3, 0.0, np.pi, 0.0], dtype=float),
)


@register_agent()
class FloatingLeapHandRight(FloatingDexHandBase):
    uid = LEAP_SPEC.uid
    HAND_SPEC = LEAP_SPEC


