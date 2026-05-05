"""Nature CNN Q-network for Atari."""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
	"""(B, frame_stack, 84, 84) uint8 or float -> (B, n_actions) Q values."""

	def __init__(self, n_actions: int, frame_stack: int = 4):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.ReLU(inplace=True),
		)
		self.head = nn.Sequential(
			nn.Linear(64 * 7 * 7, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, n_actions),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Buffer returns uint8; normalising lives in the model so callers
		# don't have to remember whether they already scaled.
		if x.dtype == torch.uint8:
			x = x.float() / 255.0
		feats = self.conv(x).flatten(1)
		return self.head(feats)
