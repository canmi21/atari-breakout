# CLAUDE.md

## Stack

Atari Breakout DQN agent — homework project, Python训练 + 一点 Rust。

- **Python** (uv-managed venv): PyTorch CNN + DQN, Gymnasium `ALE/Breakout-v5`, Matplotlib for training curves, Pygame via Gymnasium `render_mode="human"` for the demo window.
- **Rust** (cargo crate compiled to `breakout._native` via maturin / PyO3): replay buffer only. The rest stays in Python.
- **Layout**: maturin mixed layout — Python package in `breakout/`, Rust crate sources in `src/`, single workspace at the repo root.
- **Toolchain** pinned by `flake.nix`: rust toolchain via rust-overlay, plus `uv`, `bun`, `just`, `maturin` (maturin comes through `pyproject.toml`'s build-system, not nix). JS-ecosystem dev tools (`commitlint`, `lefthook`, `dprint`) installed via `bun`.

## Quality gates

Mechanical formatting and lint run on commit via lefthook. Don't re-check by hand — trust the gate.

- `cargo fmt --all -- --check` — formatting (rustfmt config in `rustfmt.toml`)
- `cargo clippy --all-targets -- -D warnings` — lint
- `dprint fmt` — md/json/toml/yaml formatting
- `commitlint` — conventional commit format, 72-char header, lower-case subject

When the gate flags formatting or lint issues, **let the toolchain fix what it can mechanically before touching anything by hand**:

- `cargo fmt --all` writes rustfmt's output in place.
- `cargo clippy --all-targets --fix --allow-dirty --allow-staged` applies clippy's machine-applicable suggestions in place.

Hand-edit only what remains after the auto-pass.

## Conventions

- **Chat:** Simplified Chinese. **Code / commits:** English.
- This is a homework project — keep code clean but **don't over-engineer**. No elaborate test pyramid. `cargo check` + a manual `just play` smoke run are the baseline. Add a unit test only when a piece of logic is non-trivial enough that eyeballing it isn't enough — otherwise skip.

## Git

Conventional Commits (see `commitlint.config.js`). Subject ≤ 72 chars, lower-case. Prefer the un-scoped `type: subject` form; use a `(scope)` only when the change touches a non-obvious area and the reader genuinely needs the disambiguation. The diff already shows the scope — most subjects should be scope-less.

When the user says "commit this" without a message, write one that passes commitlint.
