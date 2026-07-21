"""Voice library commands: list, add-preset, blend, audition, add-cloud, delete, clone."""

from pathlib import Path

import click

from seiyuu.cli import main
from seiyuu.cli.common import _voices_dir_option


@main.group()
def voice() -> None:
    """Manage the voice library (voices/{voice_id}/)."""


@voice.command("list")
@_voices_dir_option
def voice_list(voices_dir: Path | None) -> None:
    """List voices in the library."""
    from seiyuu.settings import get_settings
    from seiyuu.voices import VoiceLibrary

    lib = VoiceLibrary(voices_dir or get_settings().voices_dir)
    metas = lib.list_voices()
    if not metas:
        click.echo("no voices yet — try `seiyuu voice add-preset` or `seiyuu voice blend`")
        return
    for m in metas:
        if m.blend:
            detail = " + ".join(f"{c.preset_id}:{c.weight:g}" for c in m.blend)
        else:
            detail = m.preset_id or m.reference_audio or "—"
        click.echo(f"  {m.voice_id}  [{m.kind}/{m.engine}]  {m.name}  ({detail})  seed={m.seed}")


@voice.command("add-preset")
@click.argument("name")
@click.argument("preset_id")
@click.option("--engine", default="kokoro", show_default=True, help="TTS engine.")
@click.option("--seed", default=41172, show_default=True, help="Pinned synthesis seed.")
@click.option("--voice-id", default=None, help="Explicit voice_id (default: slug + random).")
@_voices_dir_option
def voice_add_preset(
    name: str, preset_id: str, engine: str, seed: int, voice_id: str | None, voices_dir: Path | None
) -> None:
    """Create a single-preset voice, e.g. `voice add-preset Narrator af_heart`."""
    from seiyuu.settings import get_settings
    from seiyuu.voices import VoiceKind, VoiceLibrary, VoiceLibraryError, VoiceMeta

    lib = VoiceLibrary(voices_dir or get_settings().voices_dir)
    vid = voice_id or lib.new_voice_id(name)
    try:
        lib.save(
            VoiceMeta(
                voice_id=vid,
                name=name,
                kind=VoiceKind.PRESET,
                engine=engine,
                preset_id=preset_id,
                seed=seed,
                source="preset",
            )
        )
    except (VoiceLibraryError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"added preset voice {vid} -> {preset_id}")


@voice.command("blend")
@click.argument("name")
@click.option(
    "--component",
    "components",
    multiple=True,
    help="preset_id:weight (repeatable; >=2 components). Omit to auto-draft from --gender.",
)
@click.option(
    "--gender", default=None, help="Auto-draft: same-family 2-preset blend for this gender."
)
@click.option(
    "--accent",
    default="a",
    show_default=True,
    help="Auto-draft accent: a (American) / b (British).",
)
@click.option("--seed", default=41172, show_default=True, help="Pinned synthesis seed.")
@click.option("--voice-id", default=None, help="Explicit voice_id (default: slug + random).")
@_voices_dir_option
def voice_blend(
    name: str,
    components: tuple[str, ...],
    gender: str | None,
    accent: str,
    seed: int,
    voice_id: str | None,
    voices_dir: Path | None,
) -> None:
    """Create a Kokoro blend voice from explicit components or an auto-draft."""
    from seiyuu.settings import get_settings
    from seiyuu.voices import (
        BlendComponent,
        VoiceKind,
        VoiceLibrary,
        VoiceLibraryError,
        VoiceMeta,
        auto_blend_recipe,
        canonical_recipe,
    )

    lib = VoiceLibrary(voices_dir or get_settings().voices_dir)
    if components:
        parsed = []
        for spec in components:
            preset, _, weight = spec.partition(":")
            try:
                parsed.append((preset, float(weight)))
            except ValueError as exc:
                raise click.ClickException(
                    f"bad --component {spec!r}; expected preset_id:weight"
                ) from exc
        recipe = canonical_recipe(parsed)
        source = "manual_blend"
    else:
        recipe = auto_blend_recipe(name, gender, accent=accent)
        source = "auto_blend"

    blend = [BlendComponent(preset_id=p, weight=w) for p, w in recipe]
    vid = voice_id or lib.new_voice_id(name)
    try:
        lib.save(
            VoiceMeta(
                voice_id=vid,
                name=name,
                kind=VoiceKind.BLEND,
                engine="kokoro",
                blend=blend,
                seed=seed,
                source=source,
            )
        )
    except (VoiceLibraryError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"added blend voice {vid}: {', '.join(f'{p}:{w:g}' for p, w in recipe)}")


@voice.command("audition")
@click.argument("voice_id")
@click.option(
    "--text",
    default='The quick brown fox jumps over the lazy dog. "Well," she said, "how about that?"',
    help="Sample line to synthesize.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output WAV (default: voices/{voice_id}/audition.wav).",
)
@_voices_dir_option
def voice_audition(voice_id: str, text: str, out: Path | None, voices_dir: Path | None) -> None:
    """Synthesize a sample line with a voice so you can hear it (loads a TTS engine)."""
    from contextlib import nullcontext

    from seiyuu.engines import get_engine, voices_dir_kwargs
    from seiyuu.engines.base import SynthesisError
    from seiyuu.gpu import GpuBusyError, get_gpu_manager
    from seiyuu.normalize import normalize_text, profile_for
    from seiyuu.settings import get_settings
    from seiyuu.voices import (
        VoiceKind,
        VoiceLibrary,
        VoiceLibraryError,
        ensure_cloud_voice,
        render_voice_args,
    )

    cfg = get_settings()
    lib = VoiceLibrary(voices_dir or cfg.voices_dir)
    try:
        meta = lib.load(voice_id)
    except VoiceLibraryError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        lib.verify_consent(meta)  # cloned: attested + reference hash-matches the record
    except VoiceLibraryError as exc:
        raise click.ClickException(str(exc)) from exc

    engine = get_engine(meta.engine, **voices_dir_kwargs(meta.engine, lib.voices_dir))
    engine_voice, settings = render_voice_args(meta)
    norm = normalize_text(text, profile=profile_for(meta.engine))

    # cloud voices are paid: confirm the (small) cost and resolve the cloud handle first
    cost = engine.cost_estimate(norm)
    if cost > 0:
        click.confirm(
            f"Auditioning {voice_id} is a paid call (~${cost:.4f}). Continue?", abort=True
        )
        if meta.engine == "elevenlabs" and meta.kind is VoiceKind.CLONED:
            from seiyuu.repository import RepositoryError
            from seiyuu.voices import CloudVoiceError

            try:
                engine_voice = ensure_cloud_voice(
                    meta, engine.client, lib, max_slots=cfg.elevenlabs_max_voice_slots
                )
            except (SynthesisError, CloudVoiceError, RepositoryError) as exc:
                raise click.ClickException(str(exc)) from exc

    gpu = get_gpu_manager()
    ctx = gpu.acquire(engine, engine.engine_id) if engine.uses_gpu else nullcontext()
    try:
        with ctx:
            audio = engine.synthesize(norm, engine_voice, {**settings, "seed": meta.seed})
    except (GpuBusyError, SynthesisError) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        if engine.uses_gpu:
            gpu.free_all()  # a refused acquire never claimed the card, so this stays a no-op
    out_path = Path(out) if out else lib.dir_for(voice_id) / "audition.wav"
    audio.save(out_path)
    click.echo(f"wrote {out_path} ({audio.duration_seconds:.1f}s)")


@voice.command("add-cloud")
@click.argument("name")
@click.argument("remote_voice_id")
@click.option("--engine", default="elevenlabs", show_default=True, help="Cloud engine.")
@click.option("--seed", default=41172, show_default=True, help="Pinned synthesis seed.")
@click.option("--voice-id", default=None, help="Explicit voice_id (default: slug + random).")
@_voices_dir_option
def voice_add_cloud(
    name: str, remote_voice_id: str, engine: str, seed: int, voice_id: str | None,
    voices_dir: Path | None,
) -> None:  # fmt: skip
    """Add a stock cloud voice, e.g. `voice add-cloud Rachel EXAVITQu` (a preset on the cloud)."""
    from seiyuu.settings import get_settings
    from seiyuu.voices import VoiceKind, VoiceLibrary, VoiceLibraryError, VoiceMeta

    lib = VoiceLibrary(voices_dir or get_settings().voices_dir)
    vid = voice_id or lib.new_voice_id(name)
    try:
        lib.save(
            VoiceMeta(
                voice_id=vid,
                name=name,
                kind=VoiceKind.PRESET,
                engine=engine,
                preset_id=remote_voice_id,
                seed=seed,
                source="preset",
            )  # fmt: skip
        )
    except (VoiceLibraryError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"added cloud voice {vid} -> {engine}:{remote_voice_id}")


@voice.command("delete")
@click.argument("voice_id")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Render output root scanned for assignments (default: settings.output_dir). "
    "Only THIS root is scanned — assignments under other --output-dir roots are not seen.",
)
@_voices_dir_option
def voice_delete(
    voice_id: str, yes: bool, output_dir: Path | None, voices_dir: Path | None
) -> None:
    """Delete a voice (refused while any book's assignment still references it)."""
    from seiyuu.services import ServiceError, delete_voice
    from seiyuu.settings import get_settings
    from seiyuu.voices import VoiceLibrary

    cfg = get_settings()
    lib = VoiceLibrary(voices_dir or cfg.voices_dir)
    if not yes:
        click.confirm(
            f"Delete voice {voice_id!r} and its directory (including any reference.wav — "
            f"the consent-attested source of a clone)?",
            abort=True,
        )
    try:
        gone = delete_voice(voice_id, library=lib, output_dir=output_dir or cfg.output_dir)
    except ServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"deleted {gone}")


@voice.command("clone")
@click.argument("name")
@click.argument("reference", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--engine",
    type=click.Choice(["chatterbox", "indextts2", "elevenlabs"]),
    default="chatterbox",
    show_default=True,
    help="Cloning engine: chatterbox/indextts2 (local) or elevenlabs (cloud IVC).",
)
@click.option(
    "--consent", is_flag=True, help="Attest you have the rights/consent to clone this voice."
)
@click.option(
    "--consent-by",
    default=None,
    help="Who is attesting (recorded in the attestation; default: the OS user).",
)
@click.option("--seed", default=41172, show_default=True, help="Pinned synthesis seed.")
@click.option("--voice-id", default=None, help="Explicit voice_id (default: slug + random).")
@_voices_dir_option
def voice_clone(
    name: str, reference: Path, engine: str, consent: bool, consent_by: str | None,
    seed: int, voice_id: str | None, voices_dir: Path | None,
) -> None:  # fmt: skip
    """Create a cloned voice from a reference clip (chatterbox/indextts2 local, elevenlabs IVC)."""
    import getpass
    import shutil

    from seiyuu.settings import get_settings
    from seiyuu.voices import (
        ConsentAttestation,
        VoiceKind,
        VoiceLibrary,
        VoiceLibraryError,
        VoiceMeta,
    )
    from seiyuu.voices.library import sha256_file

    if not consent:
        raise click.ClickException(
            "cloning requires --consent; you must have the rights/permission to clone this voice"
        )
    import os

    lib = VoiceLibrary(voices_dir or get_settings().voices_dir)
    vid = voice_id or lib.new_voice_id(name)
    # Order matters: copy first, hash the COPY, publish it atomically, THEN save the meta.
    # The attestation provably describes the exact bytes the library will serve — a failed
    # copy can't leave an attested meta pointing at missing/partial audio, and the source
    # clip changing mid-command can't be attested for.
    ref_path = lib.reference_path(vid)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ref_path.with_name("reference.wav.part")
    shutil.copyfile(reference, tmp)
    attestation = ConsentAttestation(
        attested_by=consent_by or getpass.getuser(),
        reference_sha256=sha256_file(tmp),
    )
    os.replace(tmp, ref_path)
    # a re-clone under an existing voice_id replaces the audio: purge conds derived from
    # the OLD reference so nothing can speak the previously-attested speaker
    for stale in ref_path.parent.glob("conds_*.pt"):
        stale.unlink(missing_ok=True)
    try:
        lib.save(
            VoiceMeta(
                voice_id=vid,
                name=name,
                kind=VoiceKind.CLONED,
                engine=engine,
                reference_audio="reference.wav",
                consent_attested=True,
                consent=attestation,
                seed=seed,
                source="user_upload",
            )  # fmt: skip
        )
    except (VoiceLibraryError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"cloned voice {vid} ({engine}) from {reference.name}")
    click.echo(
        f"consent recorded: {attestation.attested_by}, "
        f"sha256 {attestation.reference_sha256[:12]}…, {attestation.attested_at[:19]}"
    )
