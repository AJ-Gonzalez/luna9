# Luna9

Project goals: CLI agentic coding harness that learns and grows.

## Tech stack

(May be subject to change later)

For v1: Python, packaged with UV. Must run on CPU only machines.

Inference providers (Remote): Openrouter, Venice.ai

Local inference: Loading models with pytorch/gguf

## Agent directives

Ask any and all relevant clarifying questions. If something is unclear, ask.

Always ask for confirmation before committing to git. Always run git status.

When something seems logically unsound, call it out.

When answering a user question, always explain your rationale.

## Coding style and quality

- No emojis in code comments ever.
- Tests are run with pytest.
- Python files are linted with pylint and flake8.
- Type hints are required throughout the codebase.
- Use Google-style docstrings for public modules, classes, and functions.
- Maximum line length is 88 characters.
- Import sorting is not enforced for now.
