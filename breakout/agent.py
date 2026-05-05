"""Double DQN agent: epsilon-greedy actions, online/target nets, Adam."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from breakout.model import QNetwork


class DQNAgent:
	def __init__(
		self,
		n_actions: int,
		device: torch.device,
		lr: float,
		gamma: float,
		grad_clip: float,
		frame_stack: int = 4,
		seed: int = 42,
	):
		self.n_actions = n_actions
		self.device = device
		self.gamma = gamma
		self.grad_clip = grad_clip

		self.online = QNetwork(n_actions, frame_stack=frame_stack).to(device)
		self.target = QNetwork(n_actions, frame_stack=frame_stack).to(device)
		self.target.load_state_dict(self.online.state_dict())
		for p in self.target.parameters():
			p.requires_grad_(False)

		self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
		self.rng = random.Random(seed)

	def select_action(self, state: np.ndarray | torch.Tensor, epsilon: float) -> int:
		if self.rng.random() < epsilon:
			return self.rng.randrange(self.n_actions)
		state_t = state if isinstance(state, torch.Tensor) else torch.from_numpy(np.asarray(state))
		state_t = state_t.to(self.device)
		if state_t.ndim == 3:
			state_t = state_t.unsqueeze(0)
		with torch.no_grad():
			q = self.online(state_t)
		return int(q.argmax(1).item())

	def learn(
		self,
		states: np.ndarray,
		actions: np.ndarray,
		rewards: np.ndarray,
		next_states: np.ndarray,
		dones: np.ndarray,
	) -> float:
		states_t = torch.from_numpy(states).to(self.device)
		next_states_t = torch.from_numpy(next_states).to(self.device)
		actions_t = torch.from_numpy(actions).to(self.device)
		rewards_t = torch.from_numpy(rewards).to(self.device)
		dones_t = torch.from_numpy(dones).to(self.device).float()

		with torch.no_grad():
			# Double DQN: action chosen by online, value taken from target.
			next_actions = self.online(next_states_t).argmax(1, keepdim=True)
			next_q = self.target(next_states_t).gather(1, next_actions).squeeze(1)
			td_target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

		q = self.online(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
		loss = F.smooth_l1_loss(q, td_target)

		self.optim.zero_grad(set_to_none=True)
		loss.backward()
		nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
		self.optim.step()

		return float(loss.detach().item())

	def sync_target(self) -> None:
		self.target.load_state_dict(self.online.state_dict())

	def save(self, path: Path) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(
			{
				"online": self.online.state_dict(),
				"target": self.target.state_dict(),
				"optim": self.optim.state_dict(),
			},
			path,
		)

	def load(self, path: Path) -> None:
		ckpt = torch.load(path, map_location=self.device, weights_only=True)
		self.online.load_state_dict(ckpt["online"])
		self.target.load_state_dict(ckpt["target"])
		if "optim" in ckpt:
			self.optim.load_state_dict(ckpt["optim"])
