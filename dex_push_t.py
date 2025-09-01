from typing import Dict

import numpy as np
import sapien
import torch
import trimesh

from mani_skill.envs.tasks.tabletop.push_t import PushTEnv, WhiteTableSceneBuilder
from mani_skill.utils.registration import register_env
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.geometry.trimesh_utils import merge_meshes

from robots.floating_allegro import FloatingAllegroHandRight
from robots.floating_ability import FloatingAbilityHandRight

def transform_points_np(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    assert H.shape == (4, 4), H.shape
    assert pts.ndim == 2 and pts.shape[1] == 3, pts.shape
    return pts @ H[:3, :3].T + H[:3, 3]

@register_env("DexPushT-v1", max_episode_steps=None)
class DexPushTEnv(PushTEnv):


    SUPPORTED_ROBOTS = ["floating_allegro_hand_right", "floating_ability_hand_right"]
    agent: FloatingAllegroHandRight | FloatingAbilityHandRight

    def __init__(
        self, *args, robot_uids="floating_allegro_hand_right", **kwargs
    ):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        BaseEnv._load_agent(self, options, sapien.Pose(p=[0, 0, 0])) # override the default pose

    def _get_obs_extra(self, info: Dict):
        # ee position is super useful for pandastick robot
        obs = dict()
        if self.obs_mode_struct.use_state:
            # state based gets info on goal position and t full pose - necessary to learn task
            obs.update(
                goal_pos=self.goal_tee.pose.p,
                obj_pose=self.tee.pose.raw_pose,
            )

        # Generate point cloud
        transformed_robot_pcds = []
        for link_pcd, link in zip(self.robot_link_pcds, self.agent.robot.links):
            if link_pcd is not None:
                T = link.pose.to_transformation_matrix()[0].numpy()
                link_pcd = transform_points_np(T, np.array(link_pcd))
                transformed_robot_pcds.append(link_pcd)
        merged_robot_pcd = np.concatenate(transformed_robot_pcds, axis=0)
        # # visualize the point cloud
        # trimesh.PointCloud(merged_robot_pcd).show()
        # breakpoint()

        tee_pcd = transform_points_np(self.tee.pose.to_transformation_matrix()[0].numpy(), np.array(self.tee_pcd))

        obs.update(
            robot_pcd=merged_robot_pcd,
            object_pcd=tee_pcd,
        )

        return obs

    def compute_dense_reward(self, *args, **kwargs):
        return 0.0

    def _initialize_episode(self, *args, **kwargs):
        super()._initialize_episode(*args, **kwargs)

        # prepare point cloud
        if not hasattr(self, "robot_link_meshes"):
            self.robot_link_meshes = []
            for link in self.agent.robot.links:
                merged_meshes = link.generate_mesh(filter=lambda link, render_shape: True, mesh_name="merged")
                if len(merged_meshes) == 0:
                    self.robot_link_meshes.append(None)
                elif len(merged_meshes) == 1:
                    merged_mesh = merged_meshes[0]
                    # merged_mesh.apply_transform(link.pose.to_transformation_matrix()[0].numpy())
                    self.robot_link_meshes.append(merged_mesh)
                else:
                    raise ValueError(f"Expected 1 mesh, got {len(merged_meshes)}")

        if not hasattr(self, "tee_mesh"):
            meshes = self.tee.get_collision_meshes(to_world_frame=False)
            assert len(meshes) == 1
            self.tee_mesh = meshes[0]

        # resample point cloud every episode
        self.robot_link_pcds = []
        for link_mesh in self.robot_link_meshes:
            if link_mesh is None:
                self.robot_link_pcds.append(None)
            else:
                pcd = trimesh.sample.sample_surface(
                    link_mesh, 32,
                )[0]
                self.robot_link_pcds.append(pcd)
        
        # # Visualization
        # all_pcds = []
        # for pcd, link in zip(self.robot_link_pcds, self.agent.robot.links):
        #     if pcd is not None:
        #         T = link.pose.to_transformation_matrix()[0].numpy()
        #         pcd = transform_points_np(T, np.array(pcd))
        #         all_pcds.append(pcd)
        # merged_pcd = np.concatenate(all_pcds, axis=0)
        # trimesh.PointCloud(merged_pcd).show()
        # breakpoint()

        # resample tee point cloud every episode
        self.tee_pcd = trimesh.sample.sample_surface(
            self.tee_mesh, 300,
        )[0]
        # # visualize the point cloud
        # trimesh.PointCloud(self.tee_pcd).show()
        # breakpoint()

        # Reset robot initial position
        qpos = self.agent.keyframes["fingers_down"].qpos
        qpos[:2] = torch.rand(2) * 0.8 - 0.4 # xy in [-0.4, 0.4]
        self.agent.reset(qpos)
    
        min_z = np.inf
        for pcd, link in zip(self.robot_link_pcds, self.agent.robot.links):
            if pcd is not None:
                T = link.pose.to_transformation_matrix()[0].numpy()
                pcd = transform_points_np(T, np.array(pcd))
                min_z = min(min_z, pcd[:, 2].min())
        qpos[2] = qpos[2] - min_z + 0.001
        self.agent.reset(qpos)

    def _load_scene(self, options: dict):
        # have to put these parmaeters to device - defined before we had access to device
        # load scene is a convienent place for this one time operation
        self.ee_starting_pos2D = self.ee_starting_pos2D.to(self.device)
        self.ee_starting_pos3D = self.ee_starting_pos3D.to(self.device)

        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = WhiteTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # returns 3d cad of create_tee - center of mass at (0,0,0)
        # cad Tee is upside down (both 3D tee and target)
        TARGET_RED = (
            np.array([194, 19, 22, 255]) / 255
        )  # same as mani_skill.utils.building.actors.common - goal target

        def create_tee(name="tee", target=False, base_color=TARGET_RED):
            # dimensions of boxes that make tee
            # box2 is same as box1, except (3/4) the lenght, and rotated 90 degrees
            # these dimensions are an exact replica of the 3D tee model given by diffusion policy: https://cad.onshape.com/documents/f1140134e38f6ed6902648d5/w/a78cf81827600e4ff4058d03/e/f35f57fb7589f72e05c76caf
            box1_half_w = 0.2 / 2
            box1_half_h = 0.05 / 2
            half_thickness = 0.04 / 2 if not target else 1e-4

            # we have to center tee at its com so rotations are applied to com
            # vertical block is (3/4) size of horizontal block, so
            # center of mass is (1*com_horiz + (3/4)*com_vert) / (1+(3/4))
            # # center of mass is (1*(0,0)) + (3/4)*(0,(.025+.15)/2)) / (1+(3/4)) = (0,0.0375)
            com_y = 0.0375

            builder = self.scene.create_actor_builder()
            first_block_pose = sapien.Pose([0.0, 0.0 - com_y, 0.0])
            first_block_size = [box1_half_w, box1_half_h, half_thickness]
            if not target:
                builder._mass = self.T_mass
                tee_material = sapien.pysapien.physx.PhysxMaterial(
                    static_friction=self.T_dynamic_friction,
                    dynamic_friction=self.T_static_friction,
                    restitution=0,
                )
                builder.add_box_collision(
                    pose=first_block_pose,
                    half_size=first_block_size,
                    material=tee_material,
                    density=100.0, # NOTE(tongzhou): make it very light
                )
                # builder.add_box_collision(pose=first_block_pose, half_size=first_block_size)
            builder.add_box_visual(
                pose=first_block_pose,
                half_size=first_block_size,
                material=sapien.render.RenderMaterial(
                    base_color=base_color,
                ),
            )

            # for the second block (vertical part), we translate y by 4*(box1_half_h)-com_y to align flush with horizontal block
            # note that the cad model tee made here is upside down
            second_block_pose = sapien.Pose([0.0, 4 * (box1_half_h) - com_y, 0.0])
            second_block_size = [box1_half_h, (3 / 4) * (box1_half_w), half_thickness]
            if not target:
                builder.add_box_collision(
                    pose=second_block_pose,
                    half_size=second_block_size,
                    material=tee_material,
                    density=100.0, # NOTE(tongzhou): make it very light
                )
                # builder.add_box_collision(pose=second_block_pose, half_size=second_block_size)
            builder.add_box_visual(
                pose=second_block_pose,
                half_size=second_block_size,
                material=sapien.render.RenderMaterial(
                    base_color=base_color,
                ),
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
            if not target:
                return builder.build(name=name)
            else:
                return builder.build_kinematic(name=name)

        self.tee = create_tee(name="Tee", target=False)
        self.goal_tee = create_tee(
            name="goal_Tee",
            target=True,
            base_color=np.array([128, 128, 128, 255]) / 255,
        )

        # adding end-effector end-episode goal position
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_visual(
            radius=0.02,
            half_length=1e-4,
            material=sapien.render.RenderMaterial(
                base_color=np.array([128, 128, 128, 255]) / 255
            ),
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.ee_goal_pos = builder.build_kinematic(name="goal_ee")

        # Rest of function is setting up for Custom 2D "Pseudo-Rendering" function below
        res = 64
        uv_half_width = 0.15
        self.uv_half_width = uv_half_width
        self.res = res
        oned_grid = torch.arange(res, dtype=torch.float32).view(1, res).repeat(
            res, 1
        ) - (res / 2)
        self.uv_grid = (
            torch.cat([oned_grid.unsqueeze(0), (-1 * oned_grid.T).unsqueeze(0)], dim=0)
            + 0.5
        ) / ((res / 2) / uv_half_width)
        self.uv_grid = self.uv_grid.to(self.device)
        self.homo_uv = torch.cat(
            [self.uv_grid, torch.ones_like(self.uv_grid[0]).unsqueeze(0)], dim=0
        )

        # tee render
        # tee is made of two different boxes, and then translated by center of mass
        self.center_of_mass = (
            0,
            0.0375,
        )  # in frame of upside tee with center of horizontal box (add cetner of mass to get to real tee frame)
        box1 = torch.tensor(
            [[-0.1, 0.025], [0.1, 0.025], [-0.1, -0.025], [0.1, -0.025]]
        )
        box2 = torch.tensor(
            [[-0.025, 0.175], [0.025, 0.175], [-0.025, 0.025], [0.025, 0.025]]
        )
        box1[:, 1] -= self.center_of_mass[1]
        box2[:, 1] -= self.center_of_mass[1]

        # convert tee boxes to indices
        box1 *= (res / 2) / uv_half_width
        box1 += res / 2

        box2 *= (res / 2) / uv_half_width
        box2 += res / 2

        box1 = box1.long()
        box2 = box2.long()

        self.tee_render = torch.zeros(res, res)
        # image map has flipped x and y, set values in transpose to undo
        self.tee_render.T[box1[0, 0] : box1[1, 0], box1[2, 1] : box1[0, 1]] = 1
        self.tee_render.T[box2[0, 0] : box2[1, 0], box2[2, 1] : box2[0, 1]] = 1
        # image map y is flipped of xy plane, flip to unflip
        self.tee_render = self.tee_render.flip(0).to(self.device)

        goal_fake_quat = torch.tensor(
            [(torch.tensor([self.goal_z_rot]) / 2).cos(), 0, 0, 0.0]
        ).unsqueeze(0)
        zrot = self.quat_to_zrot(goal_fake_quat).squeeze(
            0
        )  # 3x3 rot matrix for goal to world transform
        goal_trans = torch.eye(3)
        goal_trans[:2, :2] = zrot[:2, :2]
        goal_trans[0:2, 2] = self.goal_offset
        self.world_to_goal_trans = torch.linalg.inv(goal_trans).to(
            self.device
        )  # this is just a 3x3 matrix (2d homogenious transform)