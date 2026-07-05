"""F2 — faster-whisper word_timestamps: transcribe_words maps seg.words -> WordTiming, and
validate_with_words returns BOTH the verdict and the words from ONE transcription pass. No live
whisper — a fake model in the faster-whisper (segments, info) shape is injected via model=.
"""

import pytest

from seiyuu.validate import SegmentWords, ValidationError, Validator, WordTiming


class _FakeWord:
    def __init__(self, start: float, end: float, word: str) -> None:
        self.start = start
        self.end = end
        self.word = word


class _FakeSeg:
    def __init__(self, text: str, words: list[_FakeWord] | None) -> None:
        self.text = text
        self.words = words


class FakeWordWhisper:
    """A faster-whisper stand-in whose transcribe(word_timestamps=True) yields scripted
    segments each carrying `.words`. Records call kwargs so tests can assert the flags passed."""

    def __init__(self, segments: list[_FakeSeg]) -> None:
        self._segments = segments
        self.calls: list[dict] = []

    def transcribe(self, path, **kwargs):
        self.calls.append(kwargs)
        return (iter(self._segments), {"language": "en"})


def _seg(text, words):
    return _FakeSeg(text, [_FakeWord(*w) for w in words])


def test_transcribe_words_flattens_seg_words():
    whisper = FakeWordWhisper(
        [
            _seg("Hello there.", [(0.0, 0.4, " Hello"), (0.4, 0.9, " there")]),
            _seg(" How are you?", [(1.0, 1.3, " How"), (1.3, 1.6, " are"), (1.6, 2.0, " you")]),
        ]
    )
    v = Validator(model=whisper)
    words = v.transcribe_words("seg.wav")
    assert [w.word for w in words] == [" Hello", " there", " How", " are", " you"]
    assert all(isinstance(w, WordTiming) for w in words)
    assert words[0].start == 0.0 and words[-1].end == 2.0
    # the word_timestamps flag MUST be passed or faster-whisper omits .words
    assert whisper.calls[0].get("word_timestamps") is True


def test_transcribe_words_tolerates_segment_without_words():
    # a silent/emitted-without-words segment contributes nothing, never crashes
    whisper = FakeWordWhisper([_FakeSeg("", None), _seg("hi", [(0.0, 0.2, " hi")])])
    v = Validator(model=whisper)
    words = v.transcribe_words("seg.wav")
    assert [w.word for w in words] == [" hi"]


def test_validate_with_words_one_pass_verdict_and_words():
    whisper = FakeWordWhisper(
        [_seg("It is a truth", [(0.0, 0.2, " It"), (0.2, 0.4, " is"),
                                (0.4, 0.5, " a"), (0.5, 0.9, " truth")])]
    )  # fmt: skip
    v = Validator(model=whisper, min_ratio=0.85)
    result, words = v.validate_with_words("seg.wav", "It is a truth")
    assert result.ok and result.score == 1.0
    assert result.transcript == "It is a truth"
    assert [w.word for w in words] == [" It", " is", " a", " truth"]
    # ONE transcription pass produced both the score and the words
    assert len(whisper.calls) == 1
    assert whisper.calls[0].get("word_timestamps") is True


def test_validate_with_words_tolerates_fewer_or_misspelled_words():
    # whisper drops a word and misspells another — the verdict still scores, words still map
    whisper = FakeWordWhisper([_seg("hello wrld", [(0.0, 0.3, " hello"), (0.3, 0.7, " wrld")])])
    v = Validator(model=whisper, min_ratio=0.99)
    result, words = v.validate_with_words("seg.wav", "hello world there")
    assert not result.ok  # dropped/misspelled content scores below a strict threshold
    assert [w.word for w in words] == [" hello", " wrld"]


def test_word_alignment_error_is_loud():
    class Boom:
        def transcribe(self, path, **kwargs):
            raise RuntimeError("ct2 exploded")

    v = Validator(model=Boom())
    with pytest.raises(ValidationError, match="word alignment failed"):
        v.transcribe_words("seg.wav")


def test_segment_words_model_roundtrip():
    sw = SegmentWords(words=[WordTiming(start=0.0, end=0.5, word=" hi")], audio_duration=0.5)
    assert sw.source == "whisper"
    assert SegmentWords.model_validate_json(sw.model_dump_json()) == sw
