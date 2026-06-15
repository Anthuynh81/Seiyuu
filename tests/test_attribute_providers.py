"""Ollama provider transport + base template, exercised with an injected fake client.

No live LLM: the fake client records the request and returns canned JSON, so we verify
schema-enforced output, keep_alive: 0 (GPU discipline), and the down-Ollama error path.
"""

import json
import types

import httpx
import pytest
from openai import APIConnectionError

from seiyuu.attribute.chunking import chunk_blocks
from seiyuu.attribute.models import CharacterRegistry
from seiyuu.attribute.providers import AttributionError, get_provider
from seiyuu.attribute.providers.local import OllamaProvider
from seiyuu.ingest.models import Block, BlockType
from seiyuu.settings import get_settings

PROMPTS_DIR = get_settings().prompts_dir


def _chunk():
    blocks = [Block(id="ch001_b0001", type=BlockType.PARAGRAPH, text="Hello there.")]
    return chunk_blocks(blocks, overlap_blocks=0)[0]


def _fake_client(content: str, recorder: dict):
    def create(**kwargs):
        recorder.update(kwargs)
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    completions = types.SimpleNamespace(create=create)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))


def test_request_is_schema_enforced_and_unloads_model():
    recorder: dict = {}
    content = json.dumps(
        {
            "segments": [{"block_id": "ch001_b0001", "type": "narration", "text": "Hello there."}],
            "characters": [],
        }
    )
    provider = OllamaProvider(
        model="qwen3.5:9b", prompts_dir=PROMPTS_DIR, client=_fake_client(content, recorder)
    )

    result = provider.attribute_chunk(_chunk(), CharacterRegistry())

    assert result.segments[0].text == "Hello there."
    assert recorder["model"] == "qwen3.5:9b"
    assert recorder["response_format"]["type"] == "json_schema"
    assert recorder["extra_body"]["keep_alive"] == 0


def test_unreachable_ollama_raises_actionable_error():
    def boom(**kwargs):
        raise APIConnectionError(request=httpx.Request("POST", "http://localhost:11434"))

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=boom))
    )
    provider = OllamaProvider(model="qwen3.5:9b", prompts_dir=PROMPTS_DIR, client=client)

    with pytest.raises(AttributionError, match="ollama serve"):
        provider.attribute_chunk(_chunk(), CharacterRegistry())


def test_schema_violation_surfaces_as_attribution_error():
    # Missing speaker on a dialogue segment violates the model invariants.
    content = json.dumps(
        {"segments": [{"block_id": "ch001_b0001", "type": "dialogue", "text": "Hi"}]}
    )
    provider = OllamaProvider(model="m", prompts_dir=PROMPTS_DIR, client=_fake_client(content, {}))
    with pytest.raises(AttributionError, match="segment schema"):
        provider.attribute_chunk(_chunk(), CharacterRegistry())


def test_get_provider_rejects_unknown():
    with pytest.raises(ValueError, match="unknown attribution provider"):
        get_provider("bogus", model="m", prompts_dir=PROMPTS_DIR)


def _fake_anthropic_client(input_obj: dict, recorder: dict):
    def create(**kwargs):
        recorder.update(kwargs)
        block = types.SimpleNamespace(type="tool_use", input=input_obj)
        return types.SimpleNamespace(content=[block], stop_reason="tool_use")

    return types.SimpleNamespace(messages=types.SimpleNamespace(create=create))


def test_anthropic_forces_tool_use_and_parses():
    from seiyuu.attribute.providers.anthropic import AnthropicProvider

    recorder: dict = {}
    payload = {
        "segments": [{"block_id": "ch001_b0001", "type": "narration", "text": "Hello there."}],
        "characters": [],
    }
    provider = AnthropicProvider(
        model="claude-opus-4-8",
        prompts_dir=PROMPTS_DIR,
        client=_fake_anthropic_client(payload, recorder),
    )

    result = provider.attribute_chunk(_chunk(), CharacterRegistry())

    assert result.segments[0].text == "Hello there."
    assert recorder["tool_choice"] == {"type": "tool", "name": "emit_attribution"}
    assert recorder["tools"][0]["name"] == "emit_attribution"


def test_anthropic_requires_api_key_when_no_client():
    from seiyuu.attribute.providers.anthropic import AnthropicProvider

    with pytest.raises(AttributionError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(model="claude-opus-4-8", prompts_dir=PROMPTS_DIR, api_key=None)
