# Atari Breakout

Double DQN, PyTorch trainer, Rust replay buffer via PyO3.

## Features

- **Double DQN** with **target-net hard sync**, linear epsilon decay, Adam + Huber loss
- **Replay buffer in Rust**: stores single `uint8` frames and reconstructs the 4-frame stack at sample time, ~4x memory saving over a pre-stacked layout
- **Auto device pick** (mps / cuda / cpu), tqdm progress bar, CSV log + matplotlib post-mortem curves, live SDL demo via `render_mode="human"`

## Usage

Enter the pinned dev shell:

```sh
nix develop
```

Install deps and build the Rust extension:

```sh
uv sync && uv run maturin develop --release
```

Train (≈5–7 h on M-series mps for 1.5M steps, override with `--total-steps`):

```sh
just train
```

Run the trained agent live in an SDL window:

```sh
just play --checkpoint checkpoints/best.pt
```

Without uv, a frozen `requirements.txt` is checked in:

```sh
pip install -r requirements.txt
```

## License

Released under the MIT License © 2026 [Canmi](https://canmi.net)
