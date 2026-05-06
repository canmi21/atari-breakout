# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Atari Breakout DQN — Training Report
#
# Run `just train` first to produce `logs/train.csv` and
# `checkpoints/best.pt`, then `just report` to (re)build this notebook.

# %%
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _find_repo_root() -> Path:
	p = Path.cwd().resolve()
	while p != p.parent:
		if (p / "pyproject.toml").exists():
			return p
		p = p.parent
	return Path.cwd()


ROOT = _find_repo_root()
LOG_PATH = ROOT / "logs/train.csv"
CKPT_PATH = ROOT / "checkpoints/best.pt"
CURVES_PNG = ROOT / "logs/curves.png"
FRAMES_PNG = ROOT / "logs/frames.png"

# %% [markdown]
# ## Training summary

# %%
def load_csv(path):
	with open(path, newline="") as fh:
		rows = list(csv.DictReader(fh))
	cols = {k: [] for k in rows[0]}
	for row in rows:
		for k, v in row.items():
			try:
				cols[k].append(float(v))
			except ValueError:
				cols[k].append(float("nan"))
	return cols


cols = load_csv(LOG_PATH)
final_step = int(cols["step"][-1])
final_eval = cols["eval_return"][-1]
best_eval = max(cols["eval_return"])
final_rolling = cols["rolling_return"][-1]

print(f"final step:                  {final_step:,}")
print(f"final eval return:           {final_eval:.1f}")
print(f"best eval return:            {best_eval:.1f}")
print(f"final rolling 100-ep return: {final_rolling:.1f}")

# %% [markdown]
# ## Training curves

# %%
def _drop_nans(xs, ys):
	pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
	return ([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else ([], [])


steps = cols["step"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

x, y = _drop_nans(steps, cols["rolling_return"])
axes[0, 0].plot(x, y)
axes[0, 0].set_title("rolling 100-episode return")
axes[0, 0].set_xlabel("step")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(steps, cols["eval_return"], color="tab:orange")
axes[0, 1].set_title("eval return")
axes[0, 1].set_xlabel("step")
axes[0, 1].grid(True, alpha=0.3)

x, y = _drop_nans(steps, cols["loss"])
axes[1, 0].plot(x, y, color="tab:red")
axes[1, 0].set_yscale("log")
axes[1, 0].set_title("huber loss (log scale)")
axes[1, 0].set_xlabel("step")
axes[1, 0].grid(True, alpha=0.3, which="both")

axes[1, 1].plot(steps, cols["epsilon"], color="tab:green")
axes[1, 1].set_title("epsilon schedule")
axes[1, 1].set_xlabel("step")
axes[1, 1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(CURVES_PNG, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"saved {CURVES_PNG.relative_to(ROOT)}")

# %% [markdown]
# ![training curves](../../logs/curves.png)

# %% [markdown]
# ## Q-network architecture

# %%
from breakout.model import QNetwork

net = QNetwork(n_actions=4)
print(net)
print(f"\ntotal trainable params: {sum(p.numel() for p in net.parameters()):,}")

# %% [markdown]
# ## Sample play — frames from one greedy episode
#
# Loads `best.pt`, plays one episode with `epsilon=0.05`, captures every
# rendered frame, and saves a 4×4 grid of evenly-spaced screenshots.

# %%
from breakout.agent import DQNAgent
from breakout.config import resolve_device
from breakout.env import make_eval_env

device = torch.device(resolve_device("auto"))
env = make_eval_env(seed=42, render_mode="rgb_array")
n_actions = int(env.action_space.n)

agent = DQNAgent(
	n_actions=n_actions, device=device, lr=1e-4, gamma=0.99, grad_clip=10.0
)
agent.load(CKPT_PATH)
agent.online.train(mode=False)

frames = []
ep_return = 0.0
obs, _ = env.reset(seed=42)
# Press FIRE once on reset so the game actually starts.
obs, _, _, _, _ = env.step(1)
frames.append(env.render())

done = False
max_frames = 1500
while not done and len(frames) < max_frames:
	action = agent.select_action(np.asarray(obs), epsilon=0.05)
	obs, reward, term, trunc, _ = env.step(action)
	done = bool(term or trunc)
	ep_return += float(reward)
	frames.append(env.render())

env.close()
print(f"captured {len(frames)} frames, episode return: {ep_return:.0f}")

# %%
n_show = 16
indices = np.linspace(0, len(frames) - 1, n_show, dtype=int)
fig, axes = plt.subplots(4, 4, figsize=(10, 13))
for ax, idx in zip(axes.flat, indices):
	ax.imshow(frames[idx])
	ax.set_title(f"frame {idx}", fontsize=9)
	ax.axis("off")
fig.tight_layout()
fig.savefig(FRAMES_PNG, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"saved {FRAMES_PNG.relative_to(ROOT)}")

# %% [markdown]
# ![sample episode frames](../../logs/frames.png)
