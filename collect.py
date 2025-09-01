from datetime import datetime
import os
import shutil
import sys
import termios
import tty
import select
import threading
import time

import gymnasium as gym
import numpy as np

import mani_skill.envs  # noqa: F401 - needed to register environments
from mani_skill.utils.wrappers import RecordEpisode, CPUGymWrapper

from dex_push_t import DexPushTEnv
from pcd_recorder import PointCloudRecorder


class TerminalKeyboardHandler:
    def __init__(self):
        self.pressed_keys = set()
        self.reset_requested = False
        self.quit_requested = False
        self.running = True
        self.old_settings = None
        
    def start(self):
        """Start keyboard listener in a separate thread"""
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        self.thread = threading.Thread(target=self._listen_for_keys, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop keyboard listener and restore terminal settings"""
        self.running = False
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            
    def _listen_for_keys(self):
        """Listen for key presses in a separate thread"""
        while self.running:
            if select.select([sys.stdin], [], [], 0.01)[0]:
                key = sys.stdin.read(1).lower()
                
                if key == '\x1b':  # ESC key
                    self.quit_requested = True
                elif key == 'r':
                    self.reset_requested = True
                elif key in 'wasdqe':
                    self.pressed_keys.add(key)
                    
            # Clear keys after a short time to simulate key release
            time.sleep(0.01)
            # Keys are held for multiple iterations to allow smooth movement
            
    def get_current_keys(self):
        """Get currently pressed keys and clear one-time events"""
        keys = self.pressed_keys.copy()
        self.pressed_keys.clear()  # Clear after reading
        return keys
        
    def check_reset(self):
        """Check if reset was requested and clear the flag"""
        if self.reset_requested:
            self.reset_requested = False
            return True
        return False
        
    def check_quit(self):
        """Check if quit was requested"""
        return self.quit_requested


def keyboard_action(env, keyboard_handler):
    """Generate action based on keyboard input. WASD controls x,y movement."""
    # Initialize action array with zeros
    action = np.zeros(env.action_space.shape)
    
    # Get pressed keys
    keys = keyboard_handler.get_current_keys()
    
    # Movement speed
    move_speed = 1.0
    
    # Map WASD to x,y movement (action[0:2])
    if 'w' in keys:
        action[0] -= move_speed
    if 's' in keys:
        action[0] += move_speed
    if 'a' in keys:
        action[1] -= move_speed
    if 'd' in keys:
        action[1] += move_speed

    # Map QE to z movement
    if 'q' in keys:
        action[2] += move_speed
    if 'e' in keys:
        action[2] -= move_speed
    
    # All other action dimensions remain 0
    return action


def random_action(env):
    return env.action_space.sample()

def cpu_make_env(env_id, seed, video_dir=None, env_kwargs = dict(), wrappers: list[gym.Wrapper] = []):
    def thunk():
        env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        if video_dir:
            env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=False)
            env = PointCloudRecorder(env, output_dir=video_dir)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    robot_uids = "floating_allegro_hand_right"
    # robot_uids = "floating_ability_hand_right"
    
    print("Keyboard Controls: WASD to move, QE to move up and down, R to reset, ESC to quit")
    print("Press keys directly in the terminal (no need to press Enter)")
    
    # Initialize terminal keyboard handler
    keyboard_handler = TerminalKeyboardHandler()
    keyboard_handler.start()

    env_id = "DexPushT-v1"
    env_kwargs = dict(
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        obs_mode="state_dict",
        render_mode="human",
        robot_uids=robot_uids,
    )
    video_dir = f"videos/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


    env = cpu_make_env(env_id, 0, video_dir, env_kwargs)()
    env.unwrapped.render_mode = "rgb_array"

    try:
        while True:
            obs, info = env.reset()

            step_idx = 0
            while True:
                time.sleep(0.1)
                # Check for quit request
                if keyboard_handler.check_quit():
                    print("\nExiting...")
                    break
                
                # Check for reset request
                if keyboard_handler.check_reset():
                    print("Resetting environment...")
                    break
                
                # Get keyboard-controlled action
                action = keyboard_action(env, keyboard_handler)
                
                obs, rew, terminated, truncated, info = env.step(action)
                env.render_human()
                print(f"\rStep {step_idx}: {rew:.2f}", end='', flush=True)
                step_idx += 1
                if truncated or terminated:
                    print()  # New line after episode ends
                    break
                    
            # Break out of outer loop if quit was requested
            if keyboard_handler.check_quit():
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        keyboard_handler.stop()
        env.close()
    # check video_dir, if it's empty, then delete it
    if os.path.exists(video_dir) and not os.listdir(video_dir):
        shutil.rmtree(video_dir)
        print(f"Deleted empty video directory: {video_dir}")
    print("Done.")

