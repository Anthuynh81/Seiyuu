"""Seiyuu CLI. Runnable as `seiyuu` or `python -m seiyuu.cli`."""

import click

from seiyuu import __version__


@click.group()
@click.version_option(__version__, prog_name="seiyuu")
def main() -> None:
    """Seiyuu — multi-voice audiobook creator."""


if __name__ == "__main__":
    main()
