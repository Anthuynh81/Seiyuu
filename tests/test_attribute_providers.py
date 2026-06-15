"""Ollama provider transport + base template, exercised with an injected fake client.

No live LLM: the fake client records the request and returns canned JSON, so we verify
schema-enforced output, keep_alive: 0 (GPU discipline), and the down-Ollama error path.
"""

import json
import types
import urllib.error

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


_SEG_JSON = {"segments": [{"block_id": "ch001_b0001", "type": "narration", "text": "Hello there."}]}


def _fake_client(content: str, recorder: dict):
    def create(**kwargs):
        recorder.update(kwargs)
        message = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice])

    completions = types.SimpleNamespace(create=create)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))


def _fake_post(response: dict, recorder: dict):
    def post(url, payload, timeout):
        recorder["url"] = url
        recorder["payload"] = payload
        return response

    return post


# --- native transport (default) ---


def test_native_request_shape_and_parse():
    rec: dict = {}
    resp = {"message": {"content": json.dumps(_SEG_JSON)}, "done_reason": "stop"}
    provider = OllamaProvider(
        model="qwen3.5:9b", prompts_dir=PROMPTS_DIR, post=_fake_post(resp, rec)
    )

    result = provider.attribute_chunk(_chunk(), CharacterRegistry())

    assert result.segments[0].text == "Hello there."
    assert rec["url"].endswith("/api/chat")
    payload = rec["payload"]
    assert payload["format"]["type"] == "object"  # the JSON schema itself
    assert payload["think"] is False
    assert payload["options"]["num_ctx"] == 8192
    assert payload["keep_alive"] == 0


def test_native_truncation_raises_context_error():
    resp = {"message": {"content": ""}, "done_reason": "length"}
    provider = OllamaProvider(
        model="qwen3.5:9b", prompts_dir=PROMPTS_DIR, post=_fake_post(resp, {})
    )
    with pytest.raises(AttributionError, match="num_ctx"):
        provider.attribute_chunk(_chunk(), CharacterRegistry())


def test_native_unreachable_raises_actionable_error():
    def boom(url, payload, timeout):
        raise urllib.error.URLError("Connection refused")

    provider = OllamaProvider(model="qwen3.5:9b", prompts_dir=PROMPTS_DIR, post=boom)
    with pytest.raises(AttributionError, match="ollama serve"):
        provider.attribute_chunk(_chunk(), CharacterRegistry())


def test_native_strips_code_fence():
    resp = {"message": {"content": "```json\n" + json.dumps(_SEG_JSON) + "\n```"}}
    provider = OllamaProvider(model="m", prompts_dir=PROMPTS_DIR, post=_fake_post(resp, {}))
    result = provider.attribute_chunk(_chunk(), CharacterRegistry())
    assert result.segments[0].text == "Hello there."


# --- openai transport (alternate, for non-thinking models) ---


def test_openai_request_is_schema_enforced_and_unloads_model():
    recorder: dict = {}
    provider = OllamaProvider(
        model="qwen3.5:9b",
        prompts_dir=PROMPTS_DIR,
        transport="openai",
        client=_fake_client(json.dumps(_SEG_JSON), recorder),
    )

    result = provider.attribute_chunk(_chunk(), CharacterRegistry())

    assert result.segments[0].text == "Hello there."
    assert recorder["response_format"]["type"] == "json_schema"
    assert recorder["extra_body"]["keep_alive"] == 0


def test_openai_unreachable_raises_actionable_error():
    def boom(**kwargs):
        raise APIConnectionError(request=httpx.Request("POST", "http://localhost:11434"))

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=boom))
    )
    provider = OllamaProvider(
        model="qwen3.5:9b", prompts_dir=PROMPTS_DIR, transport="openai", client=client
    )
    with pytest.raises(AttributionError, match="ollama serve"):
        provider.attribute_chunk(_chunk(), CharacterRegistry())


def test_schema_violation_surfaces_as_attribution_error():
    # Missing speaker on a dialogue segment violates the model invariants.
    content = json.dumps(
        {"segments": [{"block_id": "ch001_b0001", "type": "dialogue", "text": "Hi"}]}
    )
    provider = OllamaProvider(
        model="m", prompts_dir=PROMPTS_DIR, transport="openai", client=_fake_client(content, {})
    )
    with pytest.raises(AttributionError, match="segment schema"):
        provider.attribute_chunk(_chunk(), CharacterRegistry())


def test_unknown_transport_rejected():
    with pytest.raises(ValueError, match="unknown ollama transport"):
        OllamaProvider(model="m", prompts_dir=PROMPTS_DIR, transport="grpc")


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
