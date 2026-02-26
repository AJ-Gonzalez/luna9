"""Command-line interface for Luna9."""

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Optional

import typer

from luna9 import __version__

EXIT_TOKENS = {"exit", "quit", "/exit", "/quit"}
UNIVERSAL_SESSION_ID = "universal"

app = typer.Typer(
    invoke_without_command=True,
    no_args_is_help=False,
    help="Luna9 CLI harness.",
)


@dataclass
class SessionContext:
    """Mocked session context for current invocation.

    Attributes:
        universal_session_id: Identifier for the universal session scope.
        project_session_key: Directory-sensitive key for project session scope.
    """

    universal_session_id: str
    project_session_key: str


def _build_session_context() -> SessionContext:
    """Build a mocked session context from current working directory."""
    cwd = Path.cwd().resolve()
    return SessionContext(
        universal_session_id=UNIVERSAL_SESSION_ID,
        project_session_key=str(cwd),
    )


def _read_piped_input() -> Optional[str]:
    """Read piped input from stdin when available."""
    if sys.stdin.isatty():
        return None

    data = sys.stdin.read()
    if data.strip() == "":
        return None
    return data


def _format_mock_response(
    message: str,
    piped_input: Optional[str],
    session: SessionContext,
) -> str:
    """Format a mocked response that reflects session attachment behavior."""
    lines = [
        (
            "[mock] attached sessions: "
            f"universal={session.universal_session_id} "
            f"project={session.project_session_key}"
        ),
        f"message: {message}",
    ]

    if piped_input is not None:
        lines.append("piped_input:")
        lines.append(piped_input.rstrip())

    return "\n".join(lines)


def _build_one_shot_payload(
    message: str,
    piped_input: Optional[str],
    session: SessionContext,
) -> dict[str, object]:
    """Build a structured one-shot response payload."""
    return {
        "mode": "one_shot",
        "session": {
            "universal_id": session.universal_session_id,
            "project_key": session.project_session_key,
        },
        "input": {
            "message": message,
            "piped_input": piped_input,
        },
        "result": {
            "content": "mock_response",
        },
        "meta": {
            "version": __version__,
        },
    }


def _run_repl(session: SessionContext) -> None:
    """Run interactive REPL mode."""
    typer.echo("Luna9 REPL. Type /exit to quit.")

    while True:
        try:
            user_input = typer.prompt("luna9")
        except (EOFError, typer.Abort):
            break

        message = user_input.strip()
        if message.lower() in EXIT_TOKENS:
            break

        response = _format_mock_response(message, None, session)
        typer.echo(response)

    typer.echo("Exiting Luna9.")


@app.callback()
def cli(
    ctx: typer.Context,
    message: Optional[str] = typer.Option(
        None,
        "--message",
        "-m",
        help="Run one-shot mode with the provided message.",
    ),
    repl: bool = typer.Option(
        False,
        "--repl",
        help="Force interactive REPL mode.",
    ),
    output: str = typer.Option(
        "text",
        "--output",
        help="One-shot output format: text or json.",
    ),
) -> None:
    """Run Luna9 in REPL mode or one-shot message mode.

    REPL mode is the default in interactive terminals. One-shot mode is activated
    when a message is provided or when stdin is piped.
    """
    if ctx.invoked_subcommand is not None:
        return

    if output not in {"text", "json"}:
        raise typer.BadParameter("--output must be one of: text, json")

    session = _build_session_context()
    piped_input = _read_piped_input()

    if repl:
        _run_repl(session)
        return

    if message is not None or piped_input is not None:
        one_shot_message = message or ""

        payload = _build_one_shot_payload(one_shot_message, piped_input, session)
        if output == "json":
            typer.echo(json.dumps(payload))
        else:
            response = _format_mock_response(one_shot_message, piped_input, session)
            typer.echo(response)
        return

    _run_repl(session)


@app.command()
def version() -> None:
    """Show the installed Luna9 version."""
    typer.echo(f"luna9 {__version__}")


def main() -> None:
    """Run the Luna9 CLI application."""
    app()


if __name__ == "__main__":
    main()
