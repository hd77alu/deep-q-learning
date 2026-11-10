import ale_py
import gymnasium as gym
from gymnasium.wrappers import (AtariPreprocessing, FrameStackObservation)

gym.register_envs(ale_py)


def make_atari_env(env_id="ALE/Pacman-v5", render_mode=None):
    # Initialize base Atari environment
    env = gym.make(env_id, render_mode=render_mode, frameskip=1)
    
    # Preprocess frames: 84x84 grayscale, frame skipping (4), and max pooling
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False  # Keep as uint8 [0, 255] to conserve memory in replay buffer
    )
    
    # Stack the last 4 frames into shape (4, 84, 84)
    env = FrameStackObservation(env, stack_size=4)
    
    return env