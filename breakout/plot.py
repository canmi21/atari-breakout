"""Render training curves from the CSV log into a 2x2 PNG."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path: Path) -> dict[str, list[float]]:
	with path.open(newline="") as fh:
		rows = list(csv.DictReader(fh))
	if not rows:
		return {}
	cols: dict[str, list[float]] = {k: [] for k in rows[0]}
	for row in rows:
		for k, v in row.items():
			try:
				cols[k].append(float(v))
			except ValueError:
				cols[k].append(float("nan"))
	return cols


def _filter_nans(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
	pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if not math.isnan(y)]
	if not pairs:
		return [], []
	x_clean, y_clean = zip(*pairs, strict=True)
	return list(x_clean), list(y_clean)


def main() -> None:
	parser = argparse.ArgumentParser(prog="breakout.plot")
	parser.add_argument("--log", type=Path, default=Path("logs/train.csv"))
	parser.add_argument("--out", type=Path, default=Path("logs/curves.png"))
	args = parser.parse_args()

	cols = load_csv(args.log)
	if "step" not in cols:
		raise SystemExit(f"no `step` column found in {args.log}")
	steps = cols["step"]

	fig, axes = plt.subplots(2, 2, figsize=(12, 8))

	if "rolling_return" in cols:
		x, y = _filter_nans(steps, cols["rolling_return"])
		axes[0, 0].plot(x, y)
		axes[0, 0].set_title("rolling 100-episode return")
		axes[0, 0].set_xlabel("step")
		axes[0, 0].grid(True, alpha=0.3)

	if "eval_return" in cols:
		axes[0, 1].plot(steps, cols["eval_return"], color="tab:orange")
		axes[0, 1].set_title("eval return")
		axes[0, 1].set_xlabel("step")
		axes[0, 1].grid(True, alpha=0.3)

	if "loss" in cols:
		x, y = _filter_nans(steps, cols["loss"])
		axes[1, 0].plot(x, y, color="tab:red")
		axes[1, 0].set_title("huber loss")
		axes[1, 0].set_xlabel("step")
		axes[1, 0].set_yscale("log")
		axes[1, 0].grid(True, alpha=0.3, which="both")

	if "epsilon" in cols:
		axes[1, 1].plot(steps, cols["epsilon"], color="tab:green")
		axes[1, 1].set_title("epsilon")
		axes[1, 1].set_xlabel("step")
		axes[1, 1].grid(True, alpha=0.3)

	fig.tight_layout()
	args.out.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(args.out, dpi=120)
	print(f"saved {args.out}")


if __name__ == "__main__":
	main()
