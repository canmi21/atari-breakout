# `just` lists recipes; full names are canonical, short aliases work everywhere.

default:
	@just --list --unsorted

# ─── aliases ────────────────────────────────────────────────────────
alias c := check
alias b := build
alias l := lint
alias f := fmt
alias d := develop

# cargo check
check:
	cargo check --all-targets

# cargo build
build:
	cargo build --all-targets

# Rebuild the PyO3 module into the active uv venv.
develop:
	uv run maturin develop --release

# Format: rustfmt for .rs, dprint for md/json/toml/yaml (writes changes).
fmt:
	cargo fmt --all
	dprint fmt

# Lint: clippy + rustfmt check + dprint check.
lint: lint-clippy lint-fmt lint-prose

lint-clippy:
	cargo clippy --all-targets -- -D warnings

lint-fmt:
	cargo fmt --all -- --check

lint-prose:
	dprint check

# Train the DQN agent.
train *args:
	uv run python -m breakout.train {{args}}

# Load a checkpoint and run the agent in a render window.
play *args:
	uv run python -m breakout.play {{args}}

# Render training curves from logs/train.csv.
plot *args:
	uv run python -m breakout.plot {{args}}

# Clean cargo build artifacts.
clean:
	cargo clean
