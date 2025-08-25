from datetime import datetime
import os
import shutil

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - needed to register environments
from dex_push_t import DexPushTEnv
from mani_skill.utils.wrappers import RecordEpisode, CPUGymWrapper
from pcd_recorder import PointCloudRecorder


def random_action(env):
    return env.action_space.sample()

def cpu_make_env(env_id, seed, video_dir=None, env_kwargs = dict(), wrappers: list[gym.Wrapper] = []):
    def thunk():
        env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        if video_dir:
            env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False)
            env = PointCloudRecorder(env, output_dir=video_dir)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":

    num_envs = 1
    # env_id = "PushT-v1"
    env_id = "DexPushT-v1"
    headless = True
    # headless = False
    env_kwargs = dict(
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        obs_mode="state_dict",
        render_mode="rgb_array" if headless else "human",
    )
    video_dir = f"videos/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


    env = cpu_make_env(env_id, 0, video_dir if headless else None, env_kwargs)()

    ep_count = 2

    for ep_idx in range(ep_count):
        obs, info = env.reset()

        for step_idx in range(100):
            action = random_action(env)
            # action[:3] *= 0.1
            # action[3:6] = 0
            action[:6] = 0
            obs, rew, terminated, truncated, info = env.step(action)
            if not headless:
                env.render()
            print(f"{step_idx}: {rew:.2f}")
            if truncated or terminated:
                break

    env.close()
    # check video_dir, if it's empty, then delete it
    if os.path.exists(video_dir) and not os.listdir(video_dir):
        shutil.rmtree(video_dir)
    print("Done.")

