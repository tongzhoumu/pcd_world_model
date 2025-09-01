import numpy as np
from mani_skill.agents.registration import register_agent
from robots.floating_dex_hand_base import FloatingDexHandBase, HandSpec


SHADOW_TIP_URDF_CONFIG = dict(
    _materials=dict(
        tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
    ),
    link={
        "fftip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "mftip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "rftip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "lftip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "thtip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
    },
)


SHADOW_SPEC = HandSpec(
    uid="floating_shadow_hand_right",
    urdf_path="./assets/robot_hand_urdf/shadow_wo_forearm_urdf/shadow_hand_right_floating.urdf",
    # include wrist joints and all finger joints (order chosen wrist -> fingers)
    finger_joint_names=[
        # wrist
        "WRJ2", "WRJ1",
        # first finger (index)
        "FFJ4", "FFJ3", "FFJ2", "FFJ1",
        # middle
        "MFJ4", "MFJ3", "MFJ2", "MFJ1",
        # ring
        "RFJ4", "RFJ3", "RFJ2", "RFJ1",
        # little
        "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",
        # thumb
        "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",
    ],
    tip_link_names=[
        "fftip",
        "mftip",
        "rftip",
        "lftip",
        "thtip",
    ],
    palm_link_name="palm",
    urdf_config=SHADOW_TIP_URDF_CONFIG,
    base_qpos_default=np.array([0.0, 0.0, 0.3, 0.0, np.pi, 0.0], dtype=float),
)


@register_agent()
class FloatingShadowHandRight(FloatingDexHandBase):
    uid = SHADOW_SPEC.uid
    HAND_SPEC = SHADOW_SPEC


