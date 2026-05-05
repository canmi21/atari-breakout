"""Hyperparameters and CLI wiring for train / play."""

from __future__ import annotations

import argparse
import typing
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class TrainConfig:
	# Env
	env_id: str = "ALE/Breakout-v5"
	seed: int = 42

	# Schedule
	total_steps: int = 1_500_000
	learning_starts: int = 50_000
	train_freq: int = 4
	target_sync_freq: int = 8_000

	# Optimisation
	batch_size: int = 32
	lr: float = 1e-4
	gamma: float = 0.99
	grad_clip: float = 10.0

	# Replay buffer
	buffer_size: int = 100_000

	# Exploration (linear epsilon schedule)
	eps_start: float = 1.0
	eps_end: float = 0.05
	eps_decay_steps: int = 250_000

	# Eval
	eval_freq: int = 50_000
	eval_episodes: int = 5
	eval_epsilon: float = 0.01

	# Checkpointing
	ckpt_freq: int = 50_000
	keep_ckpts: int = 3
	ckpt_dir: Path = Path("checkpoints")
	log_path: Path = Path("logs/train.csv")

	# System
	device: str = "auto"
	frame_stack: int = 4
	frame_h: int = 84
	frame_w: int = 84


@dataclass
class PlayConfig:
	checkpoint: Path = Path("checkpoints/best.pt")
	seed: int = 42
	epsilon: float = 0.01
	episodes: int = 5
	device: str = "auto"


def _add_dataclass_args(parser: argparse.ArgumentParser, cls: type) -> None:
	hints = typing.get_type_hints(cls)
	for f in fields(cls):
		flag = "--" + f.name.replace("_", "-")
		annotation = hints[f.name]
		caster = Path if annotation is Path else annotation
		parser.add_argument(flag, type=caster, default=f.default, dest=f.name)


def parse_train_args(argv: list[str] | None = None) -> TrainConfig:
	parser = argparse.ArgumentParser(prog="breakout.train")
	_add_dataclass_args(parser, TrainConfig)
	ns = parser.parse_args(argv)
	return TrainConfig(**vars(ns))


def parse_play_args(argv: list[str] | None = None) -> PlayConfig:
	parser = argparse.ArgumentParser(prog="breakout.play")
	_add_dataclass_args(parser, PlayConfig)
	ns = parser.parse_args(argv)
	return PlayConfig(**vars(ns))


def resolve_device(name: str) -> str:
	"""Resolve `auto` to mps / cuda / cpu in that order of preference."""
	if name != "auto":
		return name
	import torch

	if torch.backends.mps.is_available():
		return "mps"
	if torch.cuda.is_available():
		return "cuda"
	return "cpu"
