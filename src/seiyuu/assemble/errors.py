"""Assembly-stage exception, in its own module so `pipeline` and `voice_match` share it."""


class AssembleError(Exception):
    """Loud assembly failure naming the chapter/block involved."""


__all__ = ["AssembleError"]
