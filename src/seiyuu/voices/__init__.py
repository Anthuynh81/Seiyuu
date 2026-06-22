"""Voice stage: voice library (preset/blend/cloned) + per-book character→voice assignment."""

from seiyuu.voices.assignment import AssignmentStage, VoiceAssignment, resolve_voice
from seiyuu.voices.library import VoiceLibrary, VoiceLibraryError, slugify
from seiyuu.voices.models import BlendComponent, VoiceKind, VoiceMeta

__all__ = [
    "AssignmentStage",
    "BlendComponent",
    "VoiceAssignment",
    "VoiceKind",
    "VoiceLibrary",
    "VoiceLibraryError",
    "VoiceMeta",
    "resolve_voice",
    "slugify",
]
