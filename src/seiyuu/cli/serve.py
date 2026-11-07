"""`seiyuu serve`: run the API and built UI on this PC, for local use or the Cloudflare Tunnel."""

import click

from seiyuu.cli import main

# Never a public interface: the tunnel connects to localhost, and anything else would expose
# an API that spends money to the network. Deliberately not a CLI option.
BIND_HOST = "127.0.0.1"


@main.command()
@click.option("--port", type=click.IntRange(1, 65535), default=None, help="Default: API_PORT.")
def serve(port: int | None) -> None:
    """Serve Seiyuu on 127.0.0.1 with exactly one worker and no reload."""
    import uvicorn

    from seiyuu.api.frontend import BUILD_COMMAND
    from seiyuu.settings import get_settings

    cfg = get_settings()
    if not (cfg.frontend_dist / "index.html").is_file():
        click.echo(f"The UI isn't built, so only the API will answer. Run: {BUILD_COMMAND}")
    # One worker, no reload: the job runner, GPU manager and heavy-work gate are process-local.
    uvicorn.run("seiyuu.api.main:app", host=BIND_HOST, port=port or cfg.api_port, workers=1)
