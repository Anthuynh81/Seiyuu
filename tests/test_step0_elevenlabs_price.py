"""ElevenLabs per-model list pricing: table rates, override wins, unknown model -> highest
rate (money safety: the cost gate must never under-estimate), cost_estimate arithmetic."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from seiyuu.api.main import create_app
from seiyuu.api.registry import EngineRegistry
from seiyuu.engines.elevenlabs_engine import ElevenLabsEngine
from seiyuu.settings import (
    ELEVENLABS_PRICE_PER_1K_CHARS,
    Settings,
    elevenlabs_price_per_1k_chars,
)


def _settings(tmp_path, **overrides) -> Settings:
    defaults = dict(
        books_dir=tmp_path / "books",
        output_dir=tmp_path / "output",
        voices_dir=tmp_path / "voices",
        data_dir=tmp_path / "data",
        anthropic_api_key=None,
        elevenlabs_api_key=None,
    )
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _engine(model_id: str, price: float | None = None) -> ElevenLabsEngine:
    return ElevenLabsEngine(api_key="k", model_id=model_id, price_per_1k_chars=price, client=None)


@pytest.mark.parametrize(
    ("model_id", "rate"),
    [
        ("eleven_multilingual_v2", 0.08),
        ("eleven_v3", 0.08),
        ("eleven_turbo_v2_5", 0.04),
        ("eleven_flash_v2_5", 0.04),
    ],
)
def test_per_model_list_rates(model_id, rate):
    assert elevenlabs_price_per_1k_chars(model_id) == pytest.approx(rate)
    assert _engine(model_id).price_per_1k_chars == pytest.approx(rate)


def test_override_wins_over_table():
    assert elevenlabs_price_per_1k_chars("eleven_flash_v2_5", 0.22) == pytest.approx(0.22)
    assert _engine("eleven_flash_v2_5", 0.22).price_per_1k_chars == pytest.approx(0.22)


def test_unknown_model_uses_highest_rate():
    highest = max(ELEVENLABS_PRICE_PER_1K_CHARS.values())
    assert elevenlabs_price_per_1k_chars("eleven_brand_new_v9") == pytest.approx(highest)
    assert _engine("eleven_brand_new_v9").price_per_1k_chars == pytest.approx(highest)
    # ...and the highest is never undercut by any table entry.
    assert all(r <= highest for r in ELEVENLABS_PRICE_PER_1K_CHARS.values())


def test_non_positive_override_is_refused():
    with pytest.raises(ValueError, match="> 0"):
        elevenlabs_price_per_1k_chars("eleven_v3", 0.0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, elevenlabs_price_per_1k_chars=-1)


def test_cost_estimate_arithmetic():
    eng = _engine("eleven_multilingual_v2")
    assert eng.cost_estimate("a" * 1000) == pytest.approx(0.08)
    assert eng.cost_estimate("a" * 12_500) == pytest.approx(1.00)
    assert eng.cost_estimate("") == 0.0
    assert _engine("eleven_flash_v2_5").cost_estimate("a" * 2500) == pytest.approx(0.10)
    assert _engine("eleven_v3", 0.30).cost_estimate("a" * 1000) == pytest.approx(0.30)


def test_settings_default_has_no_override(tmp_path):
    cfg = _settings(tmp_path)
    assert cfg.elevenlabs_price_per_1k_chars is None
    assert cfg.elevenlabs_effective_price_per_1k_chars == pytest.approx(0.08)
    flash = _settings(tmp_path, elevenlabs_model_id="eleven_flash_v2_5")
    assert flash.elevenlabs_effective_price_per_1k_chars == pytest.approx(0.04)
    plan = _settings(tmp_path, elevenlabs_price_per_1k_chars=0.15)
    assert plan.elevenlabs_effective_price_per_1k_chars == pytest.approx(0.15)


def test_registry_engine_prices_from_injected_settings(tmp_path, monkeypatch):
    # The registry path must resolve the price from the INJECTED Settings, never the global.
    import seiyuu.settings as settings_mod

    def _boom():
        raise AssertionError("registry path consulted global get_settings()")

    monkeypatch.setattr(settings_mod, "get_settings", _boom)
    reg = EngineRegistry(
        _settings(tmp_path, elevenlabs_model_id="eleven_turbo_v2_5"), gpu_manager=object()
    )
    assert reg.get("elevenlabs").price_per_1k_chars == pytest.approx(0.04)
    reg2 = EngineRegistry(
        _settings(tmp_path, elevenlabs_price_per_1k_chars=0.5), gpu_manager=object()
    )
    assert reg2.get("elevenlabs").cost_estimate("a" * 1000) == pytest.approx(0.5)


def test_settings_view_reports_effective_rate(tmp_path):
    app = create_app(settings=_settings(tmp_path, elevenlabs_model_id="eleven_flash_v2_5"))
    with TestClient(app) as c:
        body = c.get("/api/settings").json()
    assert body["elevenlabs_model_id"] == "eleven_flash_v2_5"
    assert body["elevenlabs_price_per_1k_chars"] == pytest.approx(0.04)
