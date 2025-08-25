import gymnasium as gym
from collections import defaultdict
import os
import pickle

class PointCloudRecorder(gym.Wrapper):
    def __init__(self, env, output_dir):
        super().__init__(env)
        assert self.env.unwrapped.obs_mode == "state_dict", "PointCloudRecorder only supports state_dict observation mode"
        self.output_dir = output_dir
        self._episode_idx = 0

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)

        qpos = obs["agent"]["qpos"]
        self._ep_data["joint_q"].append(qpos[6:])
        self._ep_data["body_q"].append(qpos[:6])
        self._ep_data["object_pos"].append(obs["extra"]["obj_pose"])

        self._ep_data["object_sampled_particles"].append(obs["extra"]["object_pcd"])
        self._ep_data["hand_sampled_particles"].append(obs["extra"]["robot_pcd"])

        self._step_idx += 1
        return obs, rew, terminated, truncated, info
    
    def reset(self, *args, **kwargs):
        self._flush()
        self._episode_idx += 1
        self._step_idx = 0
        self._ep_data = defaultdict(list)
        return self.env.reset(*args, **kwargs)

    def _flush(self):
        if hasattr(self, "_ep_data") and "joint_q" in self._ep_data:
            with open(os.path.join(self.output_dir, f"trajectory_{self._episode_idx:04d}.pkl"), "wb") as f:
                pickle.dump(self._ep_data, f)

    def close(self):
        self._flush()
        super().close()
    