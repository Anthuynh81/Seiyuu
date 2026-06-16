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


# The model returns one speaker per block now; the provider derives spans/types and slices
# text from the source. _chunk()'s block has no quotes -> one narration segment.
_LABELS = {"blocks": [{"block_id": "ch001_b0001", "speaker": None}]}


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


def _openai_provider(content: str) -> OllamaProvider:
    return OllamaProvider(
        model="m", prompts_dir=PROMPTS_DIR, transport="openai", client=_fake_client(content, {})
    )


# --- native transport (default) ---


def test_native_request_shape_and_parse():
    rec: dict = {}
    resp = {"message": {"content": json.dumps(_LABELS)}, "done_reason": "stop"}
    provider = OllamaProvider(
        model="qwen3.5:9b", prompts_dir=PROMPTS_DIR, num_ctx=4096, post=_fake_post(resp, rec)
    )

    result = provider.attribute_chunk(_chunk(), CharacterRegistry())

    assert result.segments[0].text == "Hello there."
    assert rec["url"].endswith("/api/chat")
    payload = rec["payload"]
    assert payload["format"]["type"] == "object"  # the JSON schema itself
    assert payload["think"] is False
    assert payload["options"]["num_ctx"] == 4096
    assert payload["keep_alive"] == "5m"  # stays warm between chunks


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
    resp = {"message": {"content": "```json\n" + json.dumps(_LABELS) + "\n```"}}
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
        client=_fake_client(json.dumps(_LABELS), recorder),
    )

    result = provider.attribute_chunk(_chunk(), CharacterRegistry())

    assert result.segments[0].text == "Hello there."
    assert recorder["response_format"]["type"] == "json_schema"
    assert recorder["extra_body"]["keep_alive"] == "5m"


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


def _quote_chunk():
    # A block with dialogue -> three spans: narration, quoted, narration.
    blocks = [
        Block(id="ch001_b0001", type=BlockType.PARAGRAPH, text='He paused. "Hello," she said.')
    ]
    return chunk_blocks(blocks, overlap_blocks=0)[0]


def test_spans_assemble_segments_from_source_text():
    # The model gives one speaker per block; text/types are derived and sliced from source.
    labels = {"blocks": [{"block_id": "ch001_b0001", "speaker": "Jane"}]}
    provider = _openai_provider(json.dumps(labels))
    result = provider.attribute_chunk(_quote_chunk(), CharacterRegistry())
    texts = [s.text for s in result.segments]
    assert texts == ["He paused. ", '"Hello,"', " she said."]
    assert "".join(texts) == 'He paused. "Hello," she said.'  # exact reconstruction
    # Quoted span -> dialogue by the block's speaker; prose -> narration.
    assert [s.type.value for s in result.segments] == ["narration", "dialogue", "narration"]
    assert result.segments[1].speaker == "Jane"


def test_unattributed_quote_degrades_to_narration():
    # A quoted block with no speaker is NOT an error — the quote becomes narration.
    labels = {"blocks": [{"block_id": "ch001_b0001", "speaker": None}]}
    result = _openai_provider(json.dumps(labels)).attribute_chunk(
        _quote_chunk(), CharacterRegistry()
    )
    assert all(s.type.value == "narration" for s in result.segments)
    assert "".join(s.text for s in result.segments) == 'He paused. "Hello," she said.'


def test_omitted_block_still_produces_narration():
    # The model returns no entry for the block -> it still gets narration (never a missing
    # block / reconstruction failure).
    result = _openai_provider(json.dumps({"blocks": []})).attribute_chunk(
        _quote_chunk(), CharacterRegistry()
    )
    assert result.segments and all(s.type.value == "narration" for s in result.segments)


def test_malformed_character_mention_dropped_not_fatal():
    # The model echoed the registry shape (canonical_name/id, no `name`) for a character.
    # That bad mention must be dropped, not reject the whole chunk.
    content = json.dumps(
        {
            "blocks": [{"block_id": "ch001_b0001", "speaker": None}],
            "characters": [{"id": "x", "canonical_name": "X", "gender": "male"}],
        }
    )
    result = _openai_provider(content).attribute_chunk(_chunk(), CharacterRegistry())
    assert result.segments[0].text == "Hello there."
    assert result.characters == []  # dropped, not fatal


def test_render_prompt_shows_registry_and_blocks():
    from seiyuu.attribute.models import Character
    from seiyuu.attribute.providers.base import _prompt_template, render_prompt

    registry = CharacterRegistry(
        characters=[Character(id="mr_bennet", canonical_name="Mr. Bennet", gender="male")]
    )
    prompt = render_prompt(_prompt_template(PROMPTS_DIR, "v3"), registry, _quote_chunk())
    assert '"name": "Mr. Bennet"' in prompt
    # The rendered registry must use the mention shape, not leak the internal Character field.
    assert '"canonical_name": "Mr. Bennet"' not in prompt
    assert 'He paused. "Hello," she said.' in prompt  # block text shown for attribution


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
        "blocks": [{"block_id": "ch001_b0001", "labels": [{"type": "narration", "speaker": None}]}],
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
