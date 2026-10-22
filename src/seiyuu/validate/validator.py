"""faster-whisper validation: transcribe a rendered segment and fuzzy-match it against the
normalized text it was asked to speak.

This is the guard against LLM-style TTS hallucination/drift (Chatterbox, Fish): an autoregressive
model can repeat, drop, or invent words, and a plausible-but-wrong segment must never silently
reach assembly. Deterministic engines (Kokoro) don't need it. Runs on CPU (small/int8) by default
per the GPU-discipline rule, so it never contends with a GPU TTS model — the `unload()` /
`device='cuda'` path exists only for users who explicitly opt a spare GPU in.

Scoring lives in `scoring.py` (pure): both sides are folded through the normalizer's spoken-word
rules, so whisper's "Mr."/"12,000" match the "Mister"/"twelve thousand" the engine was given;
numbers are compared by value; then a character similarity must reach `min_ratio` AND an optimal
word alignment must show no long run of wrong words, no lost negation and no pile-up of small
errors. Cosmetic differences are forgiven; dropped, looped or invented clauses, wrong numbers and
a dropped "not" are not. Each failed verdict carries a plain-words `reason`.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from seiyuu.validate.models import ValidationResult, WordTiming
from seiyuu.validate.scoring import DEFAULT_MAX_WORD_GAP, score, similarity

WHISPER_SAMPLE_RATE = 16_000

# What Validator methods take as audio: a wav path, or an in-memory float32 waveform
# ALREADY at WHISPER_SAMPLE_RATE (use resample_to_whisper) — faster-whisper ingests both.
AudioSource = Path | str | np.ndarray


def resample_to_whisper(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mono float32 audio at any rate → the 16 kHz float32 waveform faster-whisper ingests.

    The render loop hands canonical 24 kHz segment audio straight to the validator with
    this, skipping the tmp-wav write + decode round trip a path input would pay."""
    if sample_rate == WHISPER_SAMPLE_RATE:
        return np.asarray(samples, dtype=np.float32)
    import torch  # lazy like faster_whisper below: keep `import seiyuu.validate` light
    import torchaudio

    tensor = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
    return (
        torchaudio.functional.resample(tensor, sample_rate, WHISPER_SAMPLE_RATE)
        .numpy()
        .astype(np.float32)
    )


def _source(wav_path: AudioSource) -> Any:
    # faster-whisper takes a filename or a waveform; never str() an array
    return wav_path if isinstance(wav_path, np.ndarray) else str(wav_path)


def _describe(wav_path: AudioSource) -> str:
    if isinstance(wav_path, np.ndarray):
        return f"<in-memory audio, {wav_path.size} samples>"
    return str(wav_path)


class ValidationError(Exception):
    """Loud validation failure (model load or transcription)."""


def match_ratio(expected: str, transcript: str) -> float:
    """Folded similarity in [0, 1]; 1.0 == the same spoken words once spelling, case and
    punctuation are ignored. The similarity half of a verdict only (see `scoring.score`)."""
    return similarity(expected, transcript)


class Validator:
    """Transcribe-and-compare validator. Inject `model` (a faster-whisper-like object exposing
    `transcribe(path, ...) -> (segments, info)`) in tests; otherwise it lazy-loads WhisperModel."""

    # Capability marker the render loop sniffs (like validate_with_words): this validator
    # accepts an in-memory 16 kHz waveform instead of a wav path. Duck-typed test
    # validators without it keep receiving paths.
    accepts_arrays = True
    # ...and this one takes ``alternates``: other spellings of the expected text (the book text
    # without pronunciation respellings) that whisper may legitimately match instead.
    accepts_alternates = True

    def __init__(
        self,
        *,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        min_ratio: float = 0.85,
        max_word_gap: int = DEFAULT_MAX_WORD_GAP,
        language: str = "en",
        model: Any | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.min_ratio = min_ratio
        self.max_word_gap = max_word_gap
        self.language = language
        self._model = model

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - dependency is pinned
                raise ValidationError("faster-whisper is not installed") from exc
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(self, wav_path: AudioSource) -> str:
        try:
            segments, _info = self._get_model().transcribe(
                _source(wav_path), language=self.language
            )
            return " ".join(seg.text for seg in segments).strip()
        except ValidationError:
            raise
        except Exception as exc:  # surface a loud, contextual failure
            raise ValidationError(
                f"whisper transcription failed for {_describe(wav_path)}: {exc}"
            ) from exc

    def validate(
        self, wav_path: AudioSource, expected_text: str, *, alternates: Sequence[str] = ()
    ) -> ValidationResult:
        transcript = self.transcribe(wav_path)
        return self._score(expected_text, transcript, alternates)

    def _score(
        self, expected_text: str, transcript: str, alternates: Sequence[str] = ()
    ) -> ValidationResult:
        try:
            result = score(
                expected_text,
                transcript,
                alternates=alternates,
                min_ratio=self.min_ratio,
                max_word_gap=self.max_word_gap,
            )
        except Exception as exc:  # e.g. a 300+ digit run num2words can't spell
            # One unscorable transcript fails its own segment (retried, then flagged) rather
            # than aborting the whole render — and every later cache-hit rescore with it.
            return ValidationResult(
                ok=False,
                score=0.0,
                transcript=transcript,
                expected=expected_text,
                reason=f"transcript could not be scored ({type(exc).__name__})",
            )
        return ValidationResult(
            ok=result.ok,
            score=round(result.similarity, 4),
            transcript=transcript,
            expected=expected_text,
            reason=result.reason,
        )

    def rescore(
        self, verdict: ValidationResult, *, alternates: Sequence[str] = ()
    ) -> ValidationResult:
        """Re-judge a stored verdict under the current scoring rules from its saved transcript
        (no whisper pass), so a cache hit never keeps a verdict an older scorer got wrong."""
        return self._score(verdict.expected, verdict.transcript, alternates)

    def transcribe_words(self, wav_path: AudioSource) -> list[WordTiming]:
        """Per-word (start, end, word) spans for forced alignment (F2).

        One `word_timestamps=True` pass, flattening every segment's `.words` in spoken order.
        Errors are wrapped exactly like `transcribe`. CPU-only by policy, so alignment never
        becomes a second resident GPU model."""
        try:
            segments, _info = self._get_model().transcribe(
                _source(wav_path), language=self.language, word_timestamps=True
            )
            return self._flatten_words(segments)
        except ValidationError:
            raise
        except Exception as exc:  # surface a loud, contextual failure
            raise ValidationError(
                f"whisper word alignment failed for {_describe(wav_path)}: {exc}"
            ) from exc

    def validate_with_words(
        self, wav_path: AudioSource, expected_text: str, *, alternates: Sequence[str] = ()
    ) -> tuple[ValidationResult, list[WordTiming]]:
        """Score AND align in ONE transcription pass — the render-loop piggyback for
        `requires_validation` engines, so a Chatterbox/Fish segment yields word timings at
        no extra whisper cost. The transcript scored here is built from the SAME segments the
        words come from, so the verdict matches `validate()` for the same audio."""
        try:
            segments, _info = self._get_model().transcribe(
                _source(wav_path), language=self.language, word_timestamps=True
            )
            segments = list(segments)  # the generator is consumed twice below
            transcript = " ".join(seg.text for seg in segments).strip()
            words = self._flatten_words(segments)
        except ValidationError:
            raise
        except Exception as exc:  # surface a loud, contextual failure
            raise ValidationError(
                f"whisper transcription failed for {_describe(wav_path)}: {exc}"
            ) from exc
        return self._score(expected_text, transcript, alternates), words

    @staticmethod
    def _flatten_words(segments: Any) -> list[WordTiming]:
        """faster-whisper `segments[*].words[*]` (start/end/word) → flat WordTiming list.
        A segment whose `words` is None/absent (no speech, or an old model) contributes none."""
        out: list[WordTiming] = []
        for seg in segments:
            for w in getattr(seg, "words", None) or ():
                out.append(WordTiming(start=float(w.start), end=float(w.end), word=w.word))
        return out

    def unload(self) -> None:  # GpuConsumer protocol (only relevant when device='cuda')
        self._model = None
