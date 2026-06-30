"""Assembly stage: render manifest → per-chapter MP3s (M1; .m4b in M4)."""

from seiyuu.assemble.pipeline import (
    AssembleError,
    AssembleResult,
    LoudnessTarget,
    MasterResult,
    PauseProfile,
    assemble_book,
    master_book,
)

__all__ = [
    "AssembleError",
    "AssembleResult",
    "LoudnessTarget",
    "MasterResult",
    "PauseProfile",
    "assemble_book",
    "master_book",
]
