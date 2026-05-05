"""Load a checkpoint and run the agent with render_mode='human'.

ALE opens an SDL window directly -- the agent's play is the demo. Run
via `just play -- --checkpoint checkpoints/best.pt`.
"""

from __future__ import annotations

import numpy as np
import torch

from breakout.agent import DQNAgent
from breakout.config import parse_play_args, resolve_device
from breakout.env import FRAME_STACK, make_eval_env


def main() -> None:
	cfg = parse_play_args()
	device = torch.device(resolve_device(cfg.device))
	print(f"device: {device}, checkpoint: {cfg.checkpoint}")

	env = make_eval_env(seed=cfg.seed, render_mode="human")
	n_actions = int(env.action_space.n)  # type: ignore[attr-defined]

	# lr / gamma / grad_clip aren't used at inference but the constructor
	# wants them; values are inert here.
	agent = DQNAgent(
		n_actions=n_actions,
		device=device,
		lr=1e-4,
		gamma=0.99,
		grad_clip=10.0,
		frame_stack=FRAME_STACK,
		seed=cfg.seed,
	)
	agent.load(cfg.checkpoint)
	agent.online.train(mode=False)

	for ep in range(cfg.episodes):
		obs, _ = env.reset(seed=cfg.seed + ep)
		ep_return = 0.0
		steps = 0
		done = False
		while not done:
			action = agent.select_action(np.asarray(obs), cfg.epsilon)
			obs, reward, term, trunc, _ = env.step(action)
			done = bool(term or trunc)
			ep_return += float(reward)
			steps += 1
		print(f"episode {ep + 1}/{cfg.episodes}: return {ep_return:.0f}, steps {steps}")

	env.close()


if __name__ == "__main__":
	main()
