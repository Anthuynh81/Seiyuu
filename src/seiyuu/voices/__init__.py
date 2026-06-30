"""Voice stage: voice library (preset/blend/cloned) + per-book character→voice assignment."""

from seiyuu.voices.assignment import (
    ASSIGNMENT_NAME,
    AssignmentStage,
    VoiceAssignment,
    resolve_voice,
)
from seiyuu.voices.blends import auto_blend_recipe, canonical_recipe
from seiyuu.voices.library import VoiceLibrary, VoiceLibraryError, slugify
from seiyuu.voices.models import BlendComponent, VoiceKind, VoiceMeta

__all__ = [
    "ASSIGNMENT_NAME",
    "AssignmentStage",
    "BlendComponent",
    "VoiceAssignment",
    "VoiceKind",
    "VoiceLibrary",
    "VoiceLibraryError",
    "VoiceMeta",
    "auto_blend_recipe",
    "canonical_recipe",
    "resolve_voice",
    "slugify",
]
