"""Seiyuu CLI. Runnable as `seiyuu` or `python -m seiyuu.cli`."""

import click

from seiyuu import __version__


@click.group()
@click.version_option(__version__, prog_name="seiyuu")
def main() -> None:
    """Seiyuu — multi-voice audiobook creator."""


# Command modules attach themselves to `main` via @main.command() at import time, so
# these imports MUST stay below the group definition; they run purely for registration.
from seiyuu.cli import (  # noqa: E402, F401
    assemble,
    attribute,
    casting,
    ingest,
    lexicon,
    render,
    voices,
)

# Re-exported: tests import `_pass_cost_gate` from `seiyuu.cli`.
from seiyuu.cli.common import _pass_cost_gate  # noqa: E402, F401
