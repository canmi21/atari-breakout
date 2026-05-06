"""Main training loop for double DQN on Atari Breakout.

Run via `just train` or `uv run python -m breakout.train --help`.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from breakout._native import ReplayBuffer
from breakout.agent import DQNAgent
from breakout.config import TrainConfig, parse_train_args, resolve_device
from breakout.env import make_eval_env, make_train_env
from breakout.logger import CSVLogger


def linear_eps(step: int, cfg: TrainConfig) -> float:
	if step >= cfg.eps_decay_steps:
		return cfg.eps_end
	frac = step / cfg.eps_decay_steps
	return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def evaluate(agent: DQNAgent, env, n_episodes: int, epsilon: float, seed: int) -> float:
	returns: list[float] = []
	for ep in range(n_episodes):
		obs, _ = env.reset(seed=seed + ep)
		done = False
		ep_return = 0.0
		while not done:
			action = agent.select_action(np.asarray(obs), epsilon)
			obs, reward, term, trunc, _ = env.step(action)
			done = bool(term or trunc)
			ep_return += float(reward)
		returns.append(ep_return)
	return sum(returns) / len(returns)


def manage_checkpoints(ckpt_dir: Path, keep: int) -> None:
	"""Keep at most `keep` step_*.pt files; best.pt is never touched."""
	step_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
	while len(step_ckpts) > keep:
		step_ckpts.pop(0).unlink()


def run(cfg: TrainConfig) -> None:
	device = torch.device(resolve_device(cfg.device))
	print(f"device: {device}")

	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)

	train_env = make_train_env(seed=cfg.seed)
	eval_env = make_eval_env(seed=cfg.seed + 1000)
	n_actions = int(train_env.action_space.n)  # type: ignore[attr-defined]
	print(f"action space: {n_actions}")

	agent = DQNAgent(
		n_actions=n_actions,
		device=device,
		lr=cfg.lr,
		gamma=cfg.gamma,
		grad_clip=cfg.grad_clip,
		frame_stack=cfg.frame_stack,
		seed=cfg.seed,
	)
	buf = ReplayBuffer(
		capacity=cfg.buffer_size,
		frame_h=cfg.frame_h,
		frame_w=cfg.frame_w,
		frame_stack=cfg.frame_stack,
		seed=cfg.seed,
	)
	cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

	stack: deque[np.ndarray] = deque(maxlen=cfg.frame_stack)
	obs, _ = train_env.reset(seed=cfg.seed)
	for _ in range(cfg.frame_stack):
		stack.append(obs)

	episode_return = 0.0
	episode_returns: deque[float] = deque(maxlen=100)
	best_eval_return = float("-inf")
	last_eval_time = time.perf_counter()

	with CSVLogger(cfg.log_path) as logger:
		pbar = tqdm(range(1, cfg.total_steps + 1), desc="train", unit="step", smoothing=0.05)
		for step in pbar:
			eps = linear_eps(step, cfg)
			state = np.stack(stack)
			action = agent.select_action(state, eps)

			next_obs, reward, term, trunc, _ = train_env.step(action)
			done = bool(term or trunc)

			buf.push(next_obs, action, float(reward), done)
			stack.append(next_obs)
			episode_return += float(reward)

			if done:
				episode_returns.append(episode_return)
				episode_return = 0.0
				obs, _ = train_env.reset()
				stack.clear()
				for _ in range(cfg.frame_stack):
					stack.append(obs)

			loss = float("nan")
			if step >= cfg.learning_starts and step % cfg.train_freq == 0:
				batch = buf.sample(cfg.batch_size)
				loss = agent.learn(*batch)

			if step % cfg.target_sync_freq == 0:
				agent.sync_target()

			if step % 500 == 0:
				rolling_now = (
					sum(episode_returns) / len(episode_returns) if episode_returns else 0.0
				)
				pbar.set_postfix(
					eps=f"{eps:.3f}",
					loss=("--" if loss != loss else f"{loss:.3f}"),
					ret=f"{rolling_now:.1f}",
					ep=len(episode_returns),
				)

			if step % cfg.eval_freq == 0:
				eval_ret = evaluate(
					agent, eval_env, cfg.eval_episodes, cfg.eval_epsilon, cfg.seed + 1000
				)
				rolling = (
					sum(episode_returns) / len(episode_returns) if episode_returns else float("nan")
				)
				now = time.perf_counter()
				fps = cfg.eval_freq / max(now - last_eval_time, 1e-6)
				last_eval_time = now
				logger.log(
					step=step,
					epsilon=eps,
					loss=loss,
					rolling_return=rolling,
					eval_return=eval_ret,
					fps=fps,
				)
				pbar.write(
					f"step {step:>8} | eps {eps:.3f} | loss {loss:.4f} "
					f"| rolling {rolling:7.2f} | eval {eval_ret:6.2f} | fps {fps:.0f}"
				)
				if eval_ret > best_eval_return:
					best_eval_return = eval_ret
					agent.save(cfg.ckpt_dir / "best.pt")

			if step % cfg.ckpt_freq == 0:
				agent.save(cfg.ckpt_dir / f"step_{step:08d}.pt")
				manage_checkpoints(cfg.ckpt_dir, cfg.keep_ckpts)

	train_env.close()
	eval_env.close()


def main(argv: list[str] | None = None) -> None:
	# Under a jupyter kernel (nbconvert --execute, Run All, etc.) sys.argv
	# carries the kernel's connection-file flags; default to [] so we
	# fall back to TrainConfig defaults instead of choking in argparse.
	if argv is None and "ipykernel" in sys.modules:
		argv = []
	cfg = parse_train_args(argv)
	run(cfg)


if __name__ == "__main__":
	main()
