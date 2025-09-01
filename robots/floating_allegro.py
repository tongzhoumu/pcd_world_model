import numpy as np
from mani_skill.agents.registration import register_agent
from robots.floating_dex_hand_base import FloatingDexHandBase, HandSpec


ALLEGRO_TIP_URDF_CONFIG = dict(
    _materials=dict(
        tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
    ),
    link={
        "link_3_0_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "link_7_0_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "link_11_0_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        "link_15_0_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
    },
)


ALLEGRO_SPEC = HandSpec(
    uid="floating_allegro_hand_right",
    urdf_path="./assets/robot_hand_urdf/allegro_hand_urdf/allegro_hand_right_floating.urdf",
    finger_joint_names=[
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
    ],
    tip_link_names=[
        "link_15_0_tip",
        "link_3_0_tip",
        "link_7_0_tip",
        "link_11_0_tip",
    ],
    palm_link_name="palm",
    urdf_config=ALLEGRO_TIP_URDF_CONFIG,
    base_qpos_default=np.array([0.0, 0.0, 0.3, 0.0, np.pi, 0.0], dtype=float),
)


@register_agent()
class FloatingAllegroHandRight(FloatingDexHandBase):
    uid = ALLEGRO_SPEC.uid
    HAND_SPEC = ALLEGRO_SPEC