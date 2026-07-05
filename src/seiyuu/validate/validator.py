"""faster-whisper validation: transcribe a rendered segment and fuzzy-match it against the
normalized text it was asked to speak.

This is the guard against LLM-style TTS hallucination/drift (Chatterbox, Fish): an autoregressive
model can repeat, drop, or invent words, and a plausible-but-wrong segment must never silently
reach assembly. Deterministic engines (Kokoro) don't need it. Runs on CPU (small/int8) by default
per the GPU-discipline rule, so it never contends with a GPU TTS model — the `unload()` /
`device='cuda'` path exists only for users who explicitly opt a spare GPU in.

The comparison folds case and punctuation before scoring: whisper's spelling/casing/punctuation
differ cosmetically from the normalized text (and number words vs digits drift), so we compare
what was SAID, not how it was written. Gross failures — hallucination, big drops — still score far
below threshold; only cosmetic differences are forgiven.
"""

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from seiyuu.validate.models import ValidationResult, WordTiming

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]")


class ValidationError(Exception):
    """Loud validation failure (model load or transcription)."""


def _fold(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — compare spoken content, not spelling."""
    return _WS.sub(" ", _NON_WORD.sub(" ", text.casefold())).strip()


def match_ratio(expected: str, transcript: str) -> float:
    """Folded fuzzy similarity in [0, 1]; 1.0 == identical once case/punctuation are ignored."""
    return SequenceMatcher(None, _fold(expected), _fold(transcript)).ratio()


class Validator:
    """Transcribe-and-compare validator. Inject `model` (a faster-whisper-like object exposing
    `transcribe(path, ...) -> (segments, info)`) in tests; otherwise it lazy-loads WhisperModel."""

    def __init__(
        self,
        *,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        min_ratio: float = 0.85,
        language: str = "en",
        model: Any | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.min_ratio = min_ratio
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

    def transcribe(self, wav_path: Path) -> str:
        try:
            segments, _info = self._get_model().transcribe(str(wav_path), language=self.language)
            return " ".join(seg.text for seg in segments).strip()
        except ValidationError:
            raise
        except Exception as exc:  # surface a loud, contextual failure
            raise ValidationError(f"whisper transcription failed for {wav_path}: {exc}") from exc

    def validate(self, wav_path: Path, expected_text: str) -> ValidationResult:
        transcript = self.transcribe(wav_path)
        return self._score(expected_text, transcript)

    def _score(self, expected_text: str, transcript: str) -> ValidationResult:
        score = match_ratio(expected_text, transcript)
        return ValidationResult(
            ok=score >= self.min_ratio,
            score=round(score, 4),
            transcript=transcript,
            expected=expected_text,
        )

    def transcribe_words(self, wav_path: Path) -> list[WordTiming]:
        """Per-word (start, end, word) spans for forced alignment (F2).

        One `word_timestamps=True` pass, flattening every segment's `.words` in spoken order.
        Errors are wrapped exactly like `transcribe`. CPU-only by policy, so alignment never
        becomes a second resident GPU model."""
        try:
            segments, _info = self._get_model().transcribe(
                str(wav_path), language=self.language, word_timestamps=True
            )
            return self._flatten_words(segments)
        except ValidationError:
            raise
        except Exception as exc:  # surface a loud, contextual failure
            raise ValidationError(f"whisper word alignment failed for {wav_path}: {exc}") from exc

    def validate_with_words(
        self, wav_path: Path, expected_text: str
    ) -> tuple[ValidationResult, list[WordTiming]]:
        """Score AND align in ONE transcription pass — the render-loop piggyback for
        `requires_validation` engines, so a Chatterbox/Fish segment yields word timings at
        no extra whisper cost. The transcript scored here is built from the SAME segments the
        words come from, so the verdict matches `validate()` for the same audio."""
        try:
            segments, _info = self._get_model().transcribe(
                str(wav_path), language=self.language, word_timestamps=True
            )
            segments = list(segments)  # the generator is consumed twice below
            transcript = " ".join(seg.text for seg in segments).strip()
            words = self._flatten_words(segments)
        except ValidationError:
            raise
        except Exception as exc:  # surface a loud, contextual failure
            raise ValidationError(f"whisper transcription failed for {wav_path}: {exc}") from exc
        return self._score(expected_text, transcript), words

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
