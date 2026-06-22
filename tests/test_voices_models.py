"""VoiceMeta validation per kind + per-engine settings, and VoiceLibrary I/O + consent gate."""

import pytest
from pydantic import ValidationError

from seiyuu.voices import (
    BlendComponent,
    VoiceKind,
    VoiceLibrary,
    VoiceLibraryError,
    VoiceMeta,
)


def test_preset_voice_valid_and_engine_settings():
    v = VoiceMeta(
        voice_id="narrator_ab12",
        name="Narrator",
        kind=VoiceKind.PRESET,
        engine="kokoro",
        preset_id="af_heart",
        settings={"kokoro": {"speed": 1.1}},
    )
    assert v.kind is VoiceKind.PRESET
    assert v.engine_settings() == {"speed": 1.1}
    assert v.seed == 41172  # pinned default


def test_blend_requires_two_same_family_components():
    ok = VoiceMeta(
        voice_id="b1",
        name="B",
        kind=VoiceKind.BLEND,
        engine="kokoro",
        blend=[
            BlendComponent(preset_id="af_bella", weight=0.6),
            BlendComponent(preset_id="af_jessica", weight=0.4),
        ],
    )
    assert len(ok.blend) == 2
    with pytest.raises(ValidationError, match="language families"):
        VoiceMeta(
            voice_id="b2",
            name="B",
            kind=VoiceKind.BLEND,
            engine="kokoro",
            blend=[
                BlendComponent(preset_id="af_bella", weight=1),
                BlendComponent(preset_id="bm_george", weight=1),
            ],
        )
    with pytest.raises(ValidationError, match=">=2 components"):
        VoiceMeta(
            voice_id="b3",
            name="B",
            kind=VoiceKind.BLEND,
            engine="kokoro",
            blend=[BlendComponent(preset_id="af_bella", weight=1)],
        )


def test_blend_weight_must_be_positive():
    with pytest.raises(ValidationError):
        BlendComponent(preset_id="af_bella", weight=0)


def test_cloned_requires_reference_and_rejects_preset():
    v = VoiceMeta(
        voice_id="elena_9f3a",
        name="Elena",
        kind=VoiceKind.CLONED,
        engine="chatterbox",
        reference_audio="reference.wav",
        consent_attested=True,
        settings={"chatterbox": {"exaggeration": 0.5, "cfg_weight": 0.5}},
    )
    assert v.engine_settings()["exaggeration"] == 0.5
    with pytest.raises(ValidationError, match="requires reference_audio"):
        VoiceMeta(voice_id="x", name="X", kind=VoiceKind.CLONED, engine="chatterbox")
    with pytest.raises(ValidationError, match="must not set preset"):
        VoiceMeta(
            voice_id="x",
            name="X",
            kind=VoiceKind.CLONED,
            engine="chatterbox",
            reference_audio="reference.wav",
            preset_id="af_heart",
        )


def test_preset_requires_preset_id():
    with pytest.raises(ValidationError, match="requires preset_id"):
        VoiceMeta(voice_id="x", name="X", kind=VoiceKind.PRESET, engine="kokoro")


def _preset(voice_id="narrator_ab12"):
    return VoiceMeta(
        voice_id=voice_id, name="N", kind=VoiceKind.PRESET, engine="kokoro", preset_id="af_heart"
    )


def test_library_save_load_list_roundtrip(tmp_path):
    lib = VoiceLibrary(tmp_path)
    assert lib.list_voices() == []
    lib.save(_preset())
    assert lib.meta_path("narrator_ab12").is_file()
    assert lib.load("narrator_ab12").preset_id == "af_heart"
    assert [v.voice_id for v in lib.list_voices()] == ["narrator_ab12"]


def test_library_load_missing_is_loud(tmp_path):
    with pytest.raises(VoiceLibraryError, match="not found"):
        VoiceLibrary(tmp_path).load("nope")


def test_library_refuses_cloned_without_consent(tmp_path):
    lib = VoiceLibrary(tmp_path)
    cloned = VoiceMeta(
        voice_id="elena_9f3a",
        name="Elena",
        kind=VoiceKind.CLONED,
        engine="chatterbox",
        reference_audio="reference.wav",
        consent_attested=False,
    )
    with pytest.raises(VoiceLibraryError, match="consent"):
        lib.save(cloned)
    cloned.consent_attested = True
    assert lib.save(cloned).is_file()  # ok once attested


def test_new_voice_id_slug_and_suffix(tmp_path):
    lib = VoiceLibrary(tmp_path)
    assert lib.new_voice_id("Mr. Darcy", suffix="ab12") == "mr_darcy_ab12"
    auto = lib.new_voice_id("Jane")
    assert auto.startswith("jane_") and len(auto) == len("jane_") + 4
