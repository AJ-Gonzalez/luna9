"""Command-line interface for Luna9."""

import typer

from luna9 import __version__

app = typer.Typer(
    no_args_is_help=True,
    help="Luna9 CLI harness.",
)


@app.command()
def version() -> None:
    """Show the installed Luna9 version."""
    typer.echo(f"luna9 {__version__}")


@app.command()
def hello(
    name: str = typer.Option("world", "--name", "-n", help="Name to greet."),
) -> None:
    """Print a simple greeting."""
    typer.echo(f"Hello, {name}.")


def main() -> None:
    """Run the Luna9 CLI application."""
    app()


if __name__ == "__main__":
    main()
