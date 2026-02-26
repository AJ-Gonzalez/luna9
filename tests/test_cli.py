"""CLI smoke tests for Luna9."""

from typer.testing import CliRunner

from luna9.cli import app


def test_version_command_prints_version() -> None:
    """Verify the version command prints a Luna9 version string."""
    runner = CliRunner()
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.startswith("luna9 ")


def test_hello_command_uses_name_option() -> None:
    """Verify the hello command accepts the --name option."""
    runner = CliRunner()
    result = runner.invoke(app, ["hello", "--name", "Alicia"])

    assert result.exit_code == 0
    assert "Hello, Alicia." in result.stdout
