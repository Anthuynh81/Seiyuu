"""Whisper validation: fuzzy-match a transcript against normalized text (faster-whisper faked)."""

from seiyuu.validate import ValidationError, ValidationResult, Validator, match_ratio


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


# Captured from the 2026-10-22 3080 Ti engine benchmark: word-perfect audio that the old
# difflib scoring failed (autojunk collapsed long segments; digits never matched number words).
_LIGHTHOUSE = (
    "The lighthouse had not been lit in forty years, yet every evening at dusk the old keeper "
    "climbed its two hundred and twelve steps, wiped the salt from the great lens, and stood "
    "watching the horizon until the last of the fishing boats had found the harbor. Nobody in "
    "the village could remember who had first asked him to do it, and nobody had ever asked "
    "him to stop."
)
_LIGHTHOUSE_HEARD = (
    "The lighthouse had not been lit in 40 years, yet every evening at dusk the old keeper "
    "climbed  its 212 steps, wiped the salt from the Great Lens, and stood watching the horizon "
    "until  the last of the fishing boats had found the harbor.  Nobody in the village could "
    "remember who had first asked him to do it, and nobody  had ever asked him to stop."
)
_STORM = (
    "When the storm finally broke, the valley below was unrecognizable. The river had carved "
    "a new channel through the orchard, uprooting trees that had stood for a century, and the "
    "old stone bridge lay in pieces along the bank like the bones of some enormous animal. "
    "Mara stood at the edge of the ridge for a long time, counting the roofs that were still "
    "standing and the ones that were not. Still, the mill wheel turned, slow and patient, as "
    "though nothing at all had happened."
)


def _verdict(expected: str, heard: str):
    return Validator(model=FakeWhisper(heard)).validate("seg.wav", expected)


def test_long_segment_with_a_small_difference_still_passes():
    # one-word slip ("millwheel") in a long paragraph used to score 0.80 (difflib autojunk)
    heard = _STORM.replace("mill wheel", "millwheel").replace("unrecognizable.", "unrecognizable,")
    assert len(_STORM) > 200
    result = _verdict(_STORM, heard)
    assert result.ok and result.score > 0.98 and result.reason is None


def test_written_forms_match_the_spoken_words():
    assert _verdict(_LIGHTHOUSE, _LIGHTHOUSE_HEARD).ok
    assert match_ratio(_LIGHTHOUSE, _LIGHTHOUSE_HEARD) == 1.0
    cases = [
        ('"Ha! Twelve thousand credits, and not a coin less, Captain."',
         "Ha! 12,000 credits and not a coin less, Captain."),
        ("It cost five dollars, ten percent off.", "It cost $5, 10% off."),
        ("the third of May", "the 3rd of May"),
        ("Mister Darcy!", "Mr. Darcy!"),
        ("Doctor Jones and Missus Lee.", "Dr. Jones and Mrs. Lee."),
        ("Saint Paul.", "St. Paul."),
        ("Then and now.", "Then & now."),
        ("Henry the eighth ruled.", "Henry VIII ruled."),
        ("In nineteen eighty-four, it rained.", "In 1984, it rained."),  # year reading
        ("It was one thousand nine hundred and five.", "It was 1905."),  # cardinal reading
    ]  # fmt: skip
    for expected, heard in cases:
        assert _verdict(expected, heard).ok, (expected, heard)


def test_wrong_number_fails_even_when_the_ratio_is_high():
    for expected, heard in [
        ("He climbed two hundred and twelve steps.", "He climbed 213 steps."),
        ("It was eighteen forty-seven.", "It was 1874."),
        (
            "The population was one million, two hundred and thirty-four thousand, five "
            "hundred and sixty-seven.",
            "The population was 1,234,568.",
        ),
    ]:
        result = _verdict(expected, heard)
        assert not result.ok, (expected, heard, result.score)
        assert result.reason.startswith("wrong number"), result.reason


def test_homophone_of_a_number_word_is_not_a_wrong_number():
    assert _verdict("She won the race at last.", "She one the race at last.").ok


def test_long_hallucinations_fail():
    sentences = _STORM.split(". ")
    invented = ". ".join(
        [
            *sentences[:2],
            "And then the dragon ate the whole village and nobody cared",
            *sentences[3:],
        ]
    )
    tail_loop = _LIGHTHOUSE + " " + _LIGHTHOUSE[int(len(_LIGHTHOUSE) * 0.7) :]
    truncated = _LIGHTHOUSE[: int(len(_LIGHTHOUSE) * 0.8)]
    said_twice = _LIGHTHOUSE + " " + _LIGHTHOUSE
    for heard in (invented, tail_loop, truncated, said_twice):
        result = _verdict(_STORM if heard is invented else _LIGHTHOUSE, heard)
        assert not result.ok, heard[-60:]
        assert result.reason


def test_refrain_with_one_extra_word_still_passes():
    # difflib's greedy matcher aligned one chorus copy off and scored this 0.67
    chorus = (
        "Oh the wind will blow and the rain will fall, and the sea will take us one and all, "
        "so raise your cup to the ones who sailed, and drink to the ones who never came home. "
    )
    expected = "The sailors sang: " + chorus * 3
    heard = expected.replace("never came home", "never came home again", 1)
    assert _verdict(expected, heard).ok


def test_rescore_rejudges_a_stored_verdict_without_whisper():
    class NoWhisper:
        def transcribe(self, *a, **k):
            raise AssertionError("rescore must not transcribe")

    stale = ValidationResult(
        ok=False, score=0.2845, transcript=_LIGHTHOUSE_HEARD, expected=_LIGHTHOUSE
    )
    fresh = Validator(model=NoWhisper()).rescore(stale)
    assert fresh.ok and fresh.score == 1.0 and fresh.transcript == stale.transcript


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


def test_resample_to_whisper_hits_16k():
    import numpy as np

    from seiyuu.validate import resample_to_whisper

    out = resample_to_whisper(np.zeros(24_000, dtype=np.float32), 24_000)
    assert out.dtype == np.float32 and len(out) == 16_000
    passthrough = resample_to_whisper(out, 16_000)
    assert passthrough.dtype == np.float32 and len(passthrough) == 16_000


def test_validator_accepts_in_memory_waveform():
    import numpy as np

    whisper = FakeWhisper("hello there")
    v = Validator(model=whisper, min_ratio=0.5)
    waveform = np.zeros(16_000, dtype=np.float32)
    result = v.validate(waveform, "hello there")
    assert result.ok
    source, _kwargs = whisper.calls[0]
    assert isinstance(source, np.ndarray)  # never str()-ed into a bogus path
