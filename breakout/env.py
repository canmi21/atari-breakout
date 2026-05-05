"""Atari Breakout environment factories.

`make_train_env` yields single (84, 84) uint8 frames; the replay buffer
stacks them on sample. `make_eval_env` wraps with FrameStackObservation
so the agent can act directly on (4, 84, 84) without a Python-side stack.
"""

from __future__ import annotations

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)

ENV_ID = "ALE/Breakout-v5"
FRAME_H = 84
FRAME_W = 84
FRAME_STACK = 4


def _base_env(seed: int, render_mode: str | None) -> gym.Env:
	env = gym.make(
		ENV_ID,
		frameskip=1,
		repeat_action_probability=0.0,
		render_mode=render_mode,
	)
	env = AtariPreprocessing(
		env,
		noop_max=30,
		frame_skip=4,
		screen_size=FRAME_H,
		terminal_on_life_loss=False,
		grayscale_obs=True,
		scale_obs=False,
	)
	env.reset(seed=seed)
	return env


def make_train_env(seed: int = 42, render_mode: str | None = None) -> gym.Env:
	"""Single-frame env. Obs shape: (84, 84) uint8."""
	return _base_env(seed=seed, render_mode=render_mode)


def make_eval_env(seed: int = 42, render_mode: str | None = None) -> gym.Env:
	"""Frame-stacked env. Obs shape: (4, 84, 84) uint8."""
	env = _base_env(seed=seed, render_mode=render_mode)
	return FrameStackObservation(env, stack_size=FRAME_STACK)
