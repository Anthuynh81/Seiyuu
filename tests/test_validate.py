"""Whisper validation: fuzzy-match a transcript against normalized text (faster-whisper faked)."""

from seiyuu.validate import ValidationError, Validator, match_ratio


class _FakeSeg:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeWhisper:
    """Returns a scripted transcript in the faster-whisper (segments, info) shape."""

    def __init__(self, transcript: str, *, segments: int = 1) -> None:
        self._transcript = transcript
        self._segments = segments
        self.calls: list[tuple] = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        # split into N segment objects (whisper yields multiple), joined back by the validator
        words = self._transcript.split()
        chunk = max(1, len(words) // self._segments)
        parts = [" ".join(words[i : i + chunk]) for i in range(0, len(words), chunk)] or [""]
        return ([_FakeSeg(p) for p in parts], {"language": "en"})


def test_exact_match_passes():
    v = Validator(model=FakeWhisper("It is a truth universally acknowledged."), min_ratio=0.85)
    result = v.validate("seg.wav", "It is a truth universally acknowledged.")
    assert result.ok
    assert result.score == 1.0
    assert result.expected == "It is a truth universally acknowledged."


def test_hallucination_fails():
    v = Validator(model=FakeWhisper("and then the dragon flew over the mountains at dawn"))
    result = v.validate("seg.wav", "It is a truth universally acknowledged.")
    assert not result.ok
    assert result.score < 0.85


def test_case_and_punctuation_are_forgiven():
    # whisper drops the comma and the question mark and lowercases — cosmetic, not drift
    v = Validator(model=FakeWhisper("hello there how are you"), min_ratio=0.85)
    result = v.validate("seg.wav", "Hello there, how are you?")
    assert result.ok


def test_multi_segment_transcript_is_joined():
    whisper = FakeWhisper("the quick brown fox jumps over the lazy dog", segments=3)
    v = Validator(model=whisper, min_ratio=0.85)
    result = v.validate("seg.wav", "The quick brown fox jumps over the lazy dog.")
    assert result.ok
    assert len(whisper.calls) == 1


def test_match_ratio_folds_case_and_punctuation():
    assert match_ratio("Hello, world!", "hello world") == 1.0
    assert match_ratio("twenty-three", "twenty three") == 1.0
    assert match_ratio("totally different", "nothing alike here") < 0.5


def test_transcription_error_is_loud():
    class Boom:
        def transcribe(self, path, **kwargs):
            raise RuntimeError("model exploded")

    v = Validator(model=Boom())
    try:
        v.validate("seg.wav", "anything")
    except ValidationError as exc:
        assert "transcription failed" in str(exc)
    else:
        raise AssertionError("expected ValidationError")
