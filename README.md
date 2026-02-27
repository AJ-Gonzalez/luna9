# Luna9

Luna9 is a CLI-first agentic coding harness focused on persistent project
continuity, context-aware memory, and safe terminal workflows.

## Setup (UV)

### Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

### Install dependencies

```bash
uv sync
```

### Run the CLI

```bash
uv run luna9
```

### One-shot mode

```bash
uv run luna9 -m "Summarize current state"
```

### Development checks

```bash
uv run pytest
uv run flake8 src tests
uv run pylint src tests
```

## Repository Layout

```text
.
├── docs/                 # Specifications and architecture notes
├── migrations/           # SQLite schema migrations
├── src/luna9/            # CLI and core package code
├── tests/                # Test suite
├── roadmap.md            # Living project roadmap
└── README.md             # Setup and usage
```
