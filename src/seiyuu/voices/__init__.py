"""Voice stage: voice library (preset/blend/cloned) + per-book character→voice assignment."""

from seiyuu.voices.assignment import (
    ASSIGNMENT_NAME,
    AssignmentStage,
    VoiceAssignment,
    resolve_voice,
)
from seiyuu.voices.blends import auto_blend_recipe, canonical_recipe, render_voice_args
from seiyuu.voices.casting import cast_book
from seiyuu.voices.cloud import CloudVoiceError, CloudVoiceRegistry, ensure_cloud_voice
from seiyuu.voices.emotion import map_emotion
from seiyuu.voices.library import VoiceLibrary, VoiceLibraryError, sha256_file, slugify
from seiyuu.voices.models import BlendComponent, ConsentAttestation, VoiceKind, VoiceMeta
from seiyuu.voices.series import (
    SERIES_NAME,
    LinkSuggestion,
    Series,
    SeriesRegistry,
    drop_book,
    drop_book_everywhere,
    identity_key,
    load_registry,
    make_series_id,
    prune_dangling_links,
    resolve_series_overrides,
    save_cast_to_series,
    save_registry,
    seed_voice_links,
    suggest_links,
)

__all__ = [
    "ASSIGNMENT_NAME",
    "SERIES_NAME",
    "AssignmentStage",
    "BlendComponent",
    "CloudVoiceError",
    "CloudVoiceRegistry",
    "ConsentAttestation",
    "LinkSuggestion",
    "Series",
    "SeriesRegistry",
    "VoiceAssignment",
    "VoiceKind",
    "VoiceLibrary",
    "VoiceLibraryError",
    "VoiceMeta",
    "auto_blend_recipe",
    "canonical_recipe",
    "cast_book",
    "drop_book",
    "drop_book_everywhere",
    "ensure_cloud_voice",
    "identity_key",
    "load_registry",
    "make_series_id",
    "map_emotion",
    "prune_dangling_links",
    "render_voice_args",
    "resolve_series_overrides",
    "resolve_voice",
    "save_cast_to_series",
    "save_registry",
    "seed_voice_links",
    "sha256_file",
    "slugify",
    "suggest_links",
]
