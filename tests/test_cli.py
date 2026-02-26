"""CLI smoke tests for Luna9."""

import json

from typer.testing import CliRunner

from luna9.cli import app


def test_version_command_prints_version() -> None:
    """Verify the version command prints a Luna9 version string."""
    runner = CliRunner()
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.startswith("luna9 ")


def test_repl_mode_starts_and_exits() -> None:
    """Verify forced REPL mode starts and exits on /exit."""
    runner = CliRunner()
    result = runner.invoke(app, ["--repl"], input="/exit\n")

    assert result.exit_code == 0
    assert "Luna9 REPL. Type /exit to quit." in result.stdout
    assert "Exiting Luna9." in result.stdout


def test_message_mode_uses_attached_sessions() -> None:
    """Verify one-shot mode echoes message and mocked session attachment."""
    runner = CliRunner()
    result = runner.invoke(app, ["--message", "Draft a plan"])

    assert result.exit_code == 0
    assert "[mock] attached sessions:" in result.stdout
    assert "universal=universal" in result.stdout
    assert "message: Draft a plan" in result.stdout


def test_message_mode_accepts_piped_input() -> None:
    """Verify one-shot mode accepts piped stdin content alongside message."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["-m", "Review this input"],
        input="print('hello from pipe')\n",
    )

    assert result.exit_code == 0
    assert "message: Review this input" in result.stdout
    assert "piped_input:" in result.stdout
    assert "print('hello from pipe')" in result.stdout


def test_message_mode_supports_json_output() -> None:
    """Verify one-shot mode can return structured JSON output."""
    runner = CliRunner()
    result = runner.invoke(app, ["-m", "Run nightly sync", "--output", "json"])

    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert payload["mode"] == "one_shot"
    assert payload["session"]["universal_id"] == "universal"
    assert payload["input"]["message"] == "Run nightly sync"
    assert payload["result"]["content"] == "mock_response"


def test_message_mode_json_output_includes_piped_input() -> None:
    """Verify JSON output includes piped stdin content in one-shot mode."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--message", "Index this", "--output", "json"],
        input="file line 1\nfile line 2\n",
    )

    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert payload["input"]["message"] == "Index this"
    assert payload["input"]["piped_input"] == "file line 1\nfile line 2\n"
