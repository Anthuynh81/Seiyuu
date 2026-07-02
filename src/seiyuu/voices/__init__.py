"""Voice stage: voice library (preset/blend/cloned) + per-book character→voice assignment."""

from seiyuu.voices.assignment import (
    ASSIGNMENT_NAME,
    AssignmentStage,
    VoiceAssignment,
    resolve_voice,
)
from seiyuu.voices.blends import auto_blend_recipe, canonical_recipe, render_voice_args
from seiyuu.voices.cloud import CloudVoiceError, CloudVoiceRegistry, ensure_cloud_voice
from seiyuu.voices.library import VoiceLibrary, VoiceLibraryError, sha256_file, slugify
from seiyuu.voices.models import BlendComponent, ConsentAttestation, VoiceKind, VoiceMeta

__all__ = [
    "ASSIGNMENT_NAME",
    "AssignmentStage",
    "BlendComponent",
    "CloudVoiceError",
    "CloudVoiceRegistry",
    "ConsentAttestation",
    "VoiceAssignment",
    "VoiceKind",
    "VoiceLibrary",
    "VoiceLibraryError",
    "VoiceMeta",
    "auto_blend_recipe",
    "canonical_recipe",
    "ensure_cloud_voice",
    "render_voice_args",
    "resolve_voice",
    "sha256_file",
    "slugify",
]
