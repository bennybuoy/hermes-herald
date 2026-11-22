"""Unit and integration tests for multimodal image support in llm_call / llm_direct."""

from __future__ import annotations

import base64
import copy
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_herald import config, multimodal, tools


# ---------------------------------------------------------------------------
# Minimal valid raster image fixtures (generated in memory, no binary assets)
# ---------------------------------------------------------------------------

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)
JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
GIF_BYTES = b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"
WEBP_BYTES = b"RIFF\x1a\x00\x00\x00WEBPVP8 \x0e\x00\x00\x000\x01\x00\x9d\x01*\x01\x00\x01\x00"


def _data_url(mime: str, raw: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _host_result(text: str = "vision ok", provider: str = "openai-codex", model: str = "gpt-5.6-sol"):
    return SimpleNamespace(
        text=text,
        provider=provider,
        model=model,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=None,
        ),
    )


def _inventory():
    return {
        "configured_default": {
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
        },
        "providers": [
            {"provider": "openai-codex", "models": ["gpt-5.6-sol"]},
            {"provider": "anthropic", "models": ["claude-sonnet-4-6"]},
        ],
    }


def _enable_llm_direct(monkeypatch):
    monkeypatch.setenv("HERMES_HERALD_TEST_MM_KEY", "super-secret-direct-key")
    monkeypatch.setattr(
        config,
        "_load_config",
        lambda: {
            "llm_direct": {
                "enabled": True,
                "default_endpoint": "visionlab",
                "endpoints": {
                    "visionlab": {
                        "base_url": "http://127.0.0.1:9/v1",
                        "api_key": "${HERMES_HERALD_TEST_MM_KEY}",
                        "default_model": "vision-model",
                    },
                },
            },
        },
    )


def _forbid_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must fail closed before any network call")

    monkeypatch.setattr(tools, "urlopen", boom)


@pytest.fixture
def host_llm(monkeypatch):
    facade = MagicMock()
    facade.complete.return_value = _host_result()
    monkeypatch.setattr(tools, "_configured_local_model_inventory", lambda **kwargs: _inventory())
    return facade


# ---------------------------------------------------------------------------
# Backward compatibility & immutability
# ---------------------------------------------------------------------------

def test_text_only_messages_remain_plain_strings_unchanged(host_llm):
    original = [{"role": "user", "content": "Hello world"}]
    snapshot = copy.deepcopy(original)

    result = json.loads(tools.handle_llm_call({"messages": original}, _llm=host_llm))

    assert result["text"] == "vision ok"
    assert host_llm.complete.call_args.kwargs["messages"] == [
        {"role": "user", "content": "Hello world"}
    ]
    assert original == snapshot


def test_empty_images_list_is_noop_and_keeps_string_content(host_llm):
    result = json.loads(
        tools.handle_llm_call(
            {"messages": [{"role": "user", "content": "Plain text"}], "images": []},
            _llm=host_llm,
        )
    )
    assert result["text"] == "vision ok"
    assert host_llm.complete.call_args.kwargs["messages"] == [
        {"role": "user", "content": "Plain text"}
    ]


def test_normalization_does_not_mutate_caller_messages_or_images(tmp_path):
    img = tmp_path / "page.png"
    img.write_bytes(PNG_BYTES)
    messages = [{"role": "user", "content": [{"type": "text", "text": "Read"}]}]
    images = [str(img)]
    messages_copy = copy.deepcopy(messages)
    images_copy = list(images)

    normalized = multimodal.normalize_llm_messages(messages, images=images)

    assert len(normalized[0]["content"]) == 2
    assert messages == messages_copy
    assert images == images_copy


# ---------------------------------------------------------------------------
# Local file paths: PNG, JPEG, GIF, WebP, magic-byte sniffing, ~ expansion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("filename", "raw_bytes", "expected_mime"),
    [
        ("scan.png", PNG_BYTES, "image/png"),
        ("photo.jpg", JPEG_BYTES, "image/jpeg"),
        ("anim.gif", GIF_BYTES, "image/gif"),
        ("pic.webp", WEBP_BYTES, "image/webp"),
        # Magic bytes are authoritative even if extension is wrong or absent
        ("misnamed.png", JPEG_BYTES, "image/jpeg"),
        ("no_extension", PNG_BYTES, "image/png"),
    ],
)
def test_local_image_paths_via_top_level_images(tmp_path, host_llm, filename, raw_bytes, expected_mime):
    img_path = tmp_path / filename
    img_path.write_bytes(raw_bytes)

    result = json.loads(
        tools.handle_llm_call(
            {
                "messages": [{"role": "user", "content": "Transcribe this page."}],
                "images": [str(img_path)],
            },
            _llm=host_llm,
        )
    )

    assert result["text"] == "vision ok"
    sent = host_llm.complete.call_args.kwargs["messages"]
    assert len(sent) == 1
    assert sent[0]["role"] == "user"
    parts = sent[0]["content"]
    assert parts == [
        {"type": "text", "text": "Transcribe this page."},
        {"type": "image_url", "image_url": {"url": _data_url(expected_mime, raw_bytes)}},
    ]


def test_local_image_path_with_tilde_expansion(tmp_path, monkeypatch, host_llm):
    monkeypatch.setenv("HOME", str(tmp_path))
    img_path = tmp_path / "exam.png"
    img_path.write_bytes(PNG_BYTES)

    result = json.loads(
        tools.handle_llm_call(
            {
                "messages": [{"role": "user", "content": "Grade."}],
                "images": ["~/exam.png"],
            },
            _llm=host_llm,
        )
    )

    assert result["text"] == "vision ok"
    part = host_llm.complete.call_args.kwargs["messages"][0]["content"][1]
    assert part["image_url"]["url"] == _data_url("image/png", PNG_BYTES)


def test_local_image_path_inside_openai_content_parts_with_detail(tmp_path, host_llm):
    page1 = tmp_path / "p1.png"
    page2 = tmp_path / "p2.jpg"
    page1.write_bytes(PNG_BYTES)
    page2.write_bytes(JPEG_BYTES)

    result = json.loads(
        tools.handle_llm_call(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Page 1:"},
                            {"type": "image_url", "image_url": {"url": str(page1), "detail": "high"}},
                            {"type": "text", "text": "Page 2:"},
                            {"type": "image_url", "image_url": {"url": str(page2), "detail": "low"}},
                        ],
                    }
                ],
                "provider": "openai-codex",
                "model": "gpt-5.6-sol",
            },
            _llm=host_llm,
        )
    )

    assert result["text"] == "vision ok"
    sent_parts = host_llm.complete.call_args.kwargs["messages"][0]["content"]
    assert sent_parts == [
        {"type": "text", "text": "Page 1:"},
        {"type": "image_url", "image_url": {"url": _data_url("image/png", PNG_BYTES), "detail": "high"}},
        {"type": "text", "text": "Page 2:"},
        {"type": "image_url", "image_url": {"url": _data_url("image/jpeg", JPEG_BYTES), "detail": "low"}},
    ]


# ---------------------------------------------------------------------------
# Base64 data: URLs & public https:// URLs
# ---------------------------------------------------------------------------

def test_data_url_jpg_alias_whitespace_and_case_normalized(host_llm):
    b64 = base64.b64encode(JPEG_BYTES).decode("ascii")
    wrapped = b64[:8] + "\n" + b64[8:]
    raw_url = f"DATA:Image/JPG;Base64,{wrapped}"

    result = json.loads(
        tools.handle_llm_call(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": raw_url}},
                            {"type": "text", "text": "Describe."},
                        ],
                    }
                ]
            },
            _llm=host_llm,
        )
    )

    assert result["text"] == "vision ok"
    sent_parts = host_llm.complete.call_args.kwargs["messages"][0]["content"]
    assert sent_parts[0] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
    }
    assert "detail" not in sent_parts[0]["image_url"]


def test_public_https_image_url_passes_through(monkeypatch, host_llm):
    monkeypatch.setattr(
        multimodal.socket,
        "getaddrinfo",
        lambda host, port: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
    )
    url = "https://cdn.example.com/images/exam-page-1.png"

    result = json.loads(
        tools.handle_llm_call(
            {
                "messages": [{"role": "user", "content": "Read the diagram."}],
                "images": [url],
            },
            _llm=host_llm,
        )
    )

    assert result["text"] == "vision ok"
    sent = host_llm.complete.call_args.kwargs["messages"]
    assert sent[0]["content"][1] == {"type": "image_url", "image_url": {"url": url}}


def test_public_https_image_url_succeeds_when_dns_is_offline(monkeypatch, host_llm):
    def offline(*args, **kwargs):
        raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")

    monkeypatch.setattr(multimodal.socket, "getaddrinfo", offline)
    url = "https://cdn.example.com/chart.webp"

    result = json.loads(
        tools.handle_llm_call(
            {"messages": [{"role": "user", "content": "Analyze"}], "images": [url]},
            _llm=host_llm,
        )
    )

    assert result["text"] == "vision ok"
    assert host_llm.complete.call_args.kwargs["messages"][0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": url},
    }


def test_public_ip_literal_https_url_is_allowed():
    assert multimodal.validate_https_image_url("https://93.184.216.34/pic.png") == (
        "https://93.184.216.34/pic.png"
    )
    assert multimodal.validate_https_image_url(
        "https://[2606:2800:220:1:248:1893:25c8:1946]/pic.png"
    ) == "https://[2606:2800:220:1:248:1893:25c8:1946]/pic.png"


# ---------------------------------------------------------------------------
# Multi-turn, combined images, json_mode, and system_prompt
# ---------------------------------------------------------------------------

def test_top_level_images_attach_to_last_user_message_only(host_llm):
    png_url = _data_url("image/png", PNG_BYTES)
    result = json.loads(
        tools.handle_llm_call(
            {
                "messages": [
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "Now inspect this image"},
                ],
                "images": [png_url],
            },
            _llm=host_llm,
        )
    )

    assert result["text"] == "vision ok"
    sent = host_llm.complete.call_args.kwargs["messages"]
    assert sent[0] == {"role": "user", "content": "First question"}
    assert sent[1] == {"role": "assistant", "content": "First answer"}
    assert sent[2] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Now inspect this image"},
            {"type": "image_url", "image_url": {"url": png_url}},
        ],
    }


def test_top_level_images_append_to_existing_user_content_parts(host_llm):
    png_url = _data_url("image/png", PNG_BYTES)
    jpg_url = _data_url("image/jpeg", JPEG_BYTES)

    result = json.loads(
        tools.handle_llm_call(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Compare both"},
                            {"type": "image_url", "image_url": {"url": png_url}},
                        ],
                    }
                ],
                "images": [jpg_url],
            },
            _llm=host_llm,
        )
    )

    assert result["text"] == "vision ok"
    parts = host_llm.complete.call_args.kwargs["messages"][0]["content"]
    assert len(parts) == 3
    assert parts[1]["image_url"]["url"] == png_url
    assert parts[2]["image_url"]["url"] == jpg_url


def test_multimodal_with_system_prompt_and_json_mode(host_llm):
    host_llm.complete.return_value = _host_result(text='{"score": 8, "max": 10}')
    png_url = _data_url("image/png", PNG_BYTES)

    result = json.loads(
        tools.handle_llm_call(
            {
                "system_prompt": "You are an exam marker.",
                "messages": [{"role": "user", "content": "Mark question 1."}],
                "images": [png_url],
                "json_mode": True,
            },
            _llm=host_llm,
        )
    )

    assert result["text"] == '{"score":8,"max":10}'
    sent = host_llm.complete.call_args.kwargs["messages"]
    assert sent[0]["role"] == "system"
    assert "You are an exam marker." in sent[0]["content"]
    assert "valid JSON object" in sent[0]["content"]
    assert sent[1]["role"] == "user"
    assert sent[1]["content"][1] == {"type": "image_url", "image_url": {"url": png_url}}


def test_empty_string_user_content_with_images_omits_empty_text_part(host_llm):
    png_url = _data_url("image/png", PNG_BYTES)

    tools.handle_llm_call(
        {"messages": [{"role": "user", "content": ""}], "images": [png_url]},
        _llm=host_llm,
    )

    sent = host_llm.complete.call_args.kwargs["messages"]
    assert sent == [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": png_url}}]}]


# ---------------------------------------------------------------------------
# Real Hermes agent.plugin_llm integration (public API contract)
# ---------------------------------------------------------------------------

def test_real_plugin_llm_forwards_multimodal_messages_intact(tmp_path, monkeypatch):
    from agent.plugin_llm import _TrustPolicy, make_plugin_llm_for_test

    img = tmp_path / "sheet.png"
    img.write_bytes(PNG_BYTES)
    captured = {}

    def sync_caller(**kwargs):
        captured.update(kwargs)
        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Handwriting says: 42", role="assistant"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5, total_tokens=25),
        )
        return "openai-codex", "gpt-5.6-sol", resp

    policy = _TrustPolicy(
        plugin_id="hermes-herald",
        allow_provider_override=True,
        allow_any_provider=True,
        allow_model_override=True,
        allow_any_model=True,
    )
    llm = make_plugin_llm_for_test(
        plugin_id="hermes-herald",
        policy=policy,
        sync_caller=sync_caller,
    )
    monkeypatch.setattr(tools, "_configured_local_model_inventory", lambda **kwargs: _inventory())

    result = json.loads(
        tools.handle_llm_call(
            {
                "messages": [{"role": "user", "content": "Read the answer."}],
                "images": [str(img)],
                "provider": "openai-codex",
                "model": "gpt-5.6-sol",
            },
            _llm=llm,
        )
    )

    assert result["text"] == "Handwriting says: 42"
    assert result["provider"] == "openai-codex"
    assert result["model"] == "gpt-5.6-sol"
    assert captured["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Read the answer."},
                {"type": "image_url", "image_url": {"url": _data_url("image/png", PNG_BYTES)}},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# llm_direct multimodal support + response list-content + redaction
# ---------------------------------------------------------------------------

def test_llm_direct_sends_multimodal_payload_and_reads_list_content(tmp_path, monkeypatch):
    _enable_llm_direct(monkeypatch)
    img = tmp_path / "diagram.webp"
    img.write_bytes(WEBP_BYTES)
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({
                "model": "vision-model",
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Part A. "},
                            {"type": "text", "text": "Part B."},
                        ],
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            }).encode()

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(tools, "urlopen", fake_urlopen)

    result = json.loads(
        tools.handle_llm_direct(
            {
                "messages": [
                    {"role": "system", "content": "Be exact."},
                    {"role": "user", "content": "Describe the diagram."},
                ],
                "images": [str(img)],
            }
        )
    )

    assert "error" not in result, result
    assert result["text"] == "Part A. Part B."
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "Be exact."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the diagram."},
                {"type": "image_url", "image_url": {"url": _data_url("image/webp", WEBP_BYTES)}},
            ],
        },
    ]


def test_llm_direct_rejects_invalid_image_before_network(monkeypatch):
    _enable_llm_direct(monkeypatch)
    _forbid_network(monkeypatch)

    result = json.loads(
        tools.handle_llm_direct(
            {
                "messages": [{"role": "user", "content": "Inspect"}],
                "images": ["http://169.254.169.254/latest/meta-data/"],
            }
        )
    )

    assert result["status"] == "error"
    assert "http://" in result["error"]


# ---------------------------------------------------------------------------
# Validation errors: roles, part shapes, limits, MIME, magic bytes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role", ["system", "assistant"])
def test_non_user_roles_reject_content_arrays(role, host_llm):
    result = json.loads(
        tools.handle_llm_call(
            {
                "messages": [
                    {
                        "role": role,
                        "content": [{"type": "image_url", "image_url": {"url": _data_url("image/png", PNG_BYTES)}}],
                    }
                ]
            },
            _llm=host_llm,
        )
    )
    assert result["status"] == "error"
    assert "role='user'" in result["error"]
    host_llm.complete.assert_not_called()


@pytest.mark.parametrize(
    "bad_args",
    [
        {"messages": [{"role": "user", "content": []}]},
        {"messages": [{"role": "user", "content": ["not-a-dict"]}]},
        {"messages": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}]}]},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": 42}]}]},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "ok", "extra": 1}]}]},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "not-a-dict"}]}]},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _data_url("image/png", PNG_BYTES), "detail": "ultra"}}]}]},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _data_url("image/png", PNG_BYTES), "extra": 1}}]}]},
        {"messages": [{"role": "system", "content": "only system"}], "images": [_data_url("image/png", PNG_BYTES)]},
        {"messages": [{"role": "user", "content": "hi"}], "images": "not-a-list"},
        {"messages": [{"role": "user", "content": "hi"}], "images": [""]},
        {"messages": [{"role": "user", "content": "hi"}], "images": [123]},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}] * (multimodal.MAX_CONTENT_PARTS_PER_MESSAGE + 1)}]},
    ],
)
def test_malformed_multimodal_inputs_fail_closed(bad_args, host_llm):
    result = json.loads(tools.handle_llm_call(bad_args, _llm=host_llm))
    assert result["status"] == "error"
    host_llm.complete.assert_not_called()


def test_max_images_per_call_enforced_across_messages_and_top_level():
    png = _data_url("image/png", PNG_BYTES)
    parts = [{"type": "image_url", "image_url": {"url": png}} for _ in range(6)]

    # 6 + 4 == 10 succeeds
    out = multimodal.normalize_llm_messages(
        [{"role": "user", "content": parts}],
        images=[png] * 4,
    )
    assert len(out[0]["content"]) == 10

    # 6 + 5 == 11 fails
    with pytest.raises(ValueError, match="Too many images"):
        multimodal.normalize_llm_messages(
            [{"role": "user", "content": parts}],
            images=[png] * 5,
        )


def test_per_image_size_limit_enforced_on_data_url_and_local_file(tmp_path):
    # Use small custom limits to avoid allocating 20+ MiB in tests.
    oversized_png = PNG_BYTES + (b"\x00" * 1024)
    big_file = tmp_path / "big.png"
    big_file.write_bytes(oversized_png)

    with pytest.raises(ValueError, match="per-image size limit"):
        multimodal.normalize_llm_messages(
            [{"role": "user", "content": "check"}],
            images=[str(big_file)],
            max_image_bytes=512,
        )

    big_data_url = _data_url("image/png", oversized_png)
    with pytest.raises(ValueError, match="per-image size limit") as excinfo:
        multimodal.normalize_llm_messages(
            [{"role": "user", "content": "check"}],
            images=[big_data_url],
            max_image_bytes=512,
        )
    # Never echo base64 payload in errors, and render sub-MiB limits accurately.
    assert base64.b64encode(oversized_png).decode("ascii")[:24] not in str(excinfo.value)
    assert "0 MiB" not in str(excinfo.value)
    assert "512 bytes" in str(excinfo.value)


def test_total_image_bytes_limit_enforced_across_multiple_images():
    img_a = PNG_BYTES + (b"\x00" * 300)
    img_b = PNG_BYTES + (b"\x00" * 300)
    url_a = _data_url("image/png", img_a)
    url_b = _data_url("image/png", img_b)

    # Each image is < 500 bytes, but combined > 600 bytes.
    with pytest.raises(ValueError, match="Total image payload exceeds limit"):
        multimodal.normalize_llm_messages(
            [{"role": "user", "content": "check"}],
            images=[url_a, url_b],
            max_image_bytes=500,
            max_total_image_bytes=600,
        )


@pytest.mark.parametrize(
    "bad_data_url",
    [
        "data:image/png;base64",  # missing comma
        "data:image/png,raw-not-base64",
        "data:image/svg+xml;base64," + base64.b64encode(b"<svg></svg>").decode(),
        "data:image/bmp;base64," + base64.b64encode(b"BM\x00\x00\x00\x00").decode(),
        "data:text/plain;base64," + base64.b64encode(b"hello secret").decode(),
        "data:image/png;base64,",
        "data:image/png;base64,!!!INVALID_BASE64!!!",
        "data:image/png;base64," + base64.b64encode(b"not an image at all").decode(),
        # MIME mismatch: declared PNG, actual bytes are JPEG
        "data:image/png;base64," + base64.b64encode(JPEG_BYTES).decode(),
        # base64 not final parameter
        "data:image/png;base64;charset=utf-8," + base64.b64encode(PNG_BYTES).decode(),
    ],
)
def test_invalid_data_urls_rejected_without_leaking_payload(bad_data_url):
    with pytest.raises(ValueError) as excinfo:
        multimodal.validate_image_data_url(bad_data_url, label="images[0]")
    err = str(excinfo.value)
    assert "INVALID_BASE64" not in err
    assert "hello secret" not in err
    assert base64.b64encode(JPEG_BYTES).decode() not in err


# ---------------------------------------------------------------------------
# Local path security guards
# ---------------------------------------------------------------------------

def test_local_path_rejects_traversal_control_chars_missing_empty_and_dir(tmp_path):
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")

    with pytest.raises(ValueError, match="traversal"):
        multimodal.load_local_image_as_data_url(f"{tmp_path}/../escape.png")

    with pytest.raises(ValueError, match="traversal"):
        multimodal.load_local_image_as_data_url("../escape.png")

    with pytest.raises(ValueError, match="control characters"):
        multimodal.load_local_image_as_data_url(f"{tmp_path}/bad\x00.png")

    with pytest.raises(ValueError, match="not found"):
        multimodal.load_local_image_as_data_url(str(tmp_path / "does-not-exist.png"))

    with pytest.raises(ValueError, match="regular file"):
        multimodal.load_local_image_as_data_url(str(tmp_path))

    with pytest.raises(ValueError, match="empty"):
        multimodal.load_local_image_as_data_url(str(empty))


def test_local_path_rejects_non_image_file_and_does_not_leak_contents(tmp_path):
    fake = tmp_path / "fake.png"
    fake.write_text("TOP_SECRET_TOKEN=hunter2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a valid PNG, JPEG, GIF, or WebP") as excinfo:
        multimodal.load_local_image_as_data_url(str(fake))

    assert "hunter2" not in str(excinfo.value)
    assert "TOP_SECRET_TOKEN" not in str(excinfo.value)


def test_local_path_rejects_sensitive_credential_files_and_symlinks(tmp_path):
    # Even if a sensitive file somehow starts with PNG magic bytes, refuse it.
    env_file = tmp_path / ".env"
    env_file.write_bytes(PNG_BYTES)
    env_local = tmp_path / ".env.production"
    env_local.write_bytes(PNG_BYTES)
    pem_file = tmp_path / "client.pem"
    pem_file.write_bytes(PNG_BYTES)
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    ssh_key = ssh_dir / "photo.png"
    ssh_key.write_bytes(PNG_BYTES)
    link_to_env = tmp_path / "innocent.png"
    link_to_env.symlink_to(env_file)

    for target in (env_file, env_local, pem_file, ssh_key, link_to_env):
        with pytest.raises(ValueError, match="credential or sensitive"):
            multimodal.load_local_image_as_data_url(str(target))

    for pseudo in ("/proc/self/environ", "/dev/null", "/sys/kernel/hostname"):
        with pytest.raises(ValueError, match="credential or sensitive"):
            multimodal.load_local_image_as_data_url(pseudo)


def test_local_path_rejects_fifo_without_blocking(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO not supported on this platform")
    fifo = tmp_path / "pipe.png"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="regular file"):
        multimodal.load_local_image_as_data_url(str(fifo))


# ---------------------------------------------------------------------------
# SSRF & remote URL security guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_url",
    [
        "http://example.com/img.png",
        "file:///etc/passwd",
        "ftp://example.com/img.png",
        "javascript:alert(1)",
        "gopher://127.0.0.1:6379/_INFO",
        "https://user:supersecret@example.com/img.png",
        "https://user@example.com/img.png",
        "https://:supersecret@example.com/img.png",
        "https://localhost/img.png",
        "https://localhost./img.png",
        "https://sub.localhost/img.png",
        "https://metadata.google.internal/computeMetadata/v1/",
        "https://metadata.goog/img.png",
        "https://printer.local/img.png",
        "https://service.internal/img.png",
        "https://intranet/img.png",
        "https://127.0.0.1/img.png",
        "https://127.1/img.png",
        "https://0.0.0.0/img.png",
        "https://10.0.0.1/img.png",
        "https://172.16.5.4/img.png",
        "https://192.168.1.1/img.png",
        "https://169.254.169.254/latest/meta-data/",
        "https://100.64.0.1/img.png",
        "https://100.100.100.200/latest/meta-data/",
        "https://[::1]/img.png",
        "https://[::ffff:127.0.0.1]/img.png",
        "https://[::ffff:169.254.169.254]/img.png",
        "https://[::ffff:0:127.0.0.1]/img.png",
        "https://[fd00:ec2::254]/latest/meta-data/",
        "https://[fe80::1]/img.png",
        "https://[fec0::1]/img.png",
        "https://2130706433/img.png",
        "https://0x7f000001/img.png",
        "https://0177.0.0.1/img.png",
        # Unicode / IDNA / UTS#46 SSRF bypass vectors
        "https://169.254.169.254\u3002/latest/meta-data/",
        "https://169.254.169.254\uff0e/latest/meta-data/",
        "https://169.254.169.254\uff61/latest/meta-data/",
        "https://\uff4c\uff4f\uff43\uff41\uff4c\uff48\uff4f\uff53\uff54.\uff4c\uff4f\uff43\uff41\uff4c/img.png",
        "https://\uff11\uff12\uff17.\uff10.\uff10.\uff11/img.png",
        "https://\uff11\uff16\uff19\u3002\uff12\uff15\uff14\u3002\uff11\uff16\uff19\u3002\uff12\uff15\uff14/latest/meta-data/",
        "https://example.com:99999/img.png",
        "https://127.0.0.1\\@example.com/img.png",
        "https://example.com/img with space.png",
        "https://example.com/" + ("a" * multimodal.MAX_IMAGE_URL_LENGTH),
    ],
)
def test_unsafe_or_ssrf_image_urls_rejected_without_leaking_credentials(bad_url):
    with pytest.raises(ValueError) as excinfo:
        multimodal.normalize_llm_messages(
            [{"role": "user", "content": "Inspect"}],
            images=[bad_url],
        )
    assert "supersecret" not in str(excinfo.value)


@pytest.mark.parametrize("resolved_ip", ["127.0.0.1", "10.0.0.5", "169.254.169.254", "::1", "fd00:ec2::254"])
def test_https_url_resolving_to_private_ip_is_blocked(monkeypatch, resolved_ip):
    family = socket.AF_INET6 if ":" in resolved_ip else socket.AF_INET
    addr = (resolved_ip, 0, 0, 0) if family == socket.AF_INET6 else (resolved_ip, 0)
    monkeypatch.setattr(
        multimodal.socket,
        "getaddrinfo",
        lambda host, port: [(family, socket.SOCK_STREAM, 6, "", addr)],
    )

    with pytest.raises(ValueError, match="resolves to a private, loopback, or reserved IP"):
        multimodal.validate_https_image_url("https://rebind.example.com/pic.png")


# ---------------------------------------------------------------------------
# Base64 / secret redaction in tool error paths
# ---------------------------------------------------------------------------

def test_llm_call_redacts_base64_echoed_in_provider_exception(host_llm):
    png_url = _data_url("image/png", PNG_BYTES)
    raw_b64 = png_url.split(",", 1)[1]
    host_llm.complete.side_effect = RuntimeError(f"Bad image_url payload: {png_url} rejected due to format")

    result = json.loads(
        tools.handle_llm_call(
            {"messages": [{"role": "user", "content": "Inspect"}], "images": [png_url]},
            _llm=host_llm,
        )
    )

    assert result["status"] == "error"
    assert raw_b64 not in result["error"]
    assert "data:image/png;base64,[redacted]" in result["error"]
    assert "rejected due to format" in result["error"]


def test_redact_data_urls_handles_multiple_urls_and_preserves_surrounding_text():
    png_url = _data_url("image/png", PNG_BYTES)
    jpg_url = _data_url("image/jpeg", JPEG_BYTES)
    png_b64 = png_url.split(",", 1)[1]
    jpg_b64 = jpg_url.split(",", 1)[1]

    raw = f"first={png_url} and second={jpg_url} rejected"
    redacted = multimodal.redact_data_urls(raw)

    assert redacted == (
        "first=data:image/png;base64,[redacted] and "
        "second=data:image/jpeg;base64,[redacted] rejected"
    )
    assert png_b64 not in redacted
    assert jpg_b64 not in redacted
    # Idempotent if applied twice.
    assert multimodal.redact_data_urls(redacted) == redacted


def test_llm_direct_redacts_base64_and_secret_from_provider_error(monkeypatch):
    _enable_llm_direct(monkeypatch)
    png_url = _data_url("image/png", PNG_BYTES)
    raw_b64 = png_url.split(",", 1)[1]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
                "error": {
                    "message": f"failed for key super-secret-direct-key on {png_url}"
                },
            }).encode()

    monkeypatch.setattr(tools, "urlopen", lambda *a, **k: FakeResponse())

    result = json.loads(
        tools.handle_llm_direct(
            {"messages": [{"role": "user", "content": "Inspect"}], "images": [png_url]}
        )
    )

    assert result["status"] == "error"
    assert "super-secret-direct-key" not in result["error"]
    assert raw_b64 not in result["error"]
    assert "data:image/png;base64,[redacted]" in result["error"]


# ---------------------------------------------------------------------------
# Schema contracts
# ---------------------------------------------------------------------------

def test_both_llm_schemas_expose_multimodal_content_and_images():
    for schema in (tools.LLM_CALL_SCHEMA, tools.LLM_DIRECT_SCHEMA):
        props = schema["parameters"]["properties"]
        assert "images" in props
        assert props["images"]["type"] == "array"
        assert props["images"]["maxItems"] == multimodal.MAX_IMAGES_PER_CALL

        content_schema = props["messages"]["items"]["properties"]["content"]
        assert "anyOf" in content_schema
        branch_types = [b["type"] for b in content_schema["anyOf"]]
        assert branch_types == ["string", "array"]
        # Ensure singular string types everywhere (compatible with Gemini/llama.cpp).
        part_props = content_schema["anyOf"][1]["items"]["properties"]
        assert part_props["type"]["enum"] == ["text", "image_url"]
        assert part_props["image_url"]["properties"]["detail"]["enum"] == ["auto", "low", "high"]


def test_multimodal_payload_and_schemas_survive_hermes_provider_adapters(tmp_path):
    """Verify normalized payloads translate cleanly through Hermes's provider adapters."""
    img = tmp_path / "scan.png"
    img.write_bytes(PNG_BYTES)
    messages = multimodal.normalize_llm_messages(
        [{"role": "user", "content": "Grade Q1"}],
        images=[str(img)],
    )

    from agent import auxiliary_client

    convert_anth = getattr(auxiliary_client, "_convert_openai_images_to_anthropic", None)
    if callable(convert_anth):
        anth = convert_anth(messages)
        assert anth[0]["content"][1] == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(PNG_BYTES).decode("ascii"),
            },
        }

    try:
        from agent.codex_responses_adapter import _chat_content_to_responses_parts
    except ImportError:
        _chat_content_to_responses_parts = None
    if callable(_chat_content_to_responses_parts):
        parts = _chat_content_to_responses_parts(messages[0]["content"], role="user")
        assert parts == [
            {"type": "input_text", "text": "Grade Q1"},
            {"type": "input_image", "image_url": _data_url("image/png", PNG_BYTES)},
        ]

    try:
        from agent.gemini_native_adapter import _extract_multimodal_parts
    except ImportError:
        _extract_multimodal_parts = None
    if callable(_extract_multimodal_parts):
        gparts = _extract_multimodal_parts(messages[0]["content"])
        assert gparts == [
            {"text": "Grade Q1"},
            {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": base64.b64encode(PNG_BYTES).decode("ascii"),
                }
            },
        ]

    try:
        from tools.schema_sanitizer import sanitize_tool_schemas
    except ImportError:
        sanitize_tool_schemas = None
    if callable(sanitize_tool_schemas):
        wrapped = [
            {"type": "function", "function": copy.deepcopy(tools.LLM_CALL_SCHEMA)},
            {"type": "function", "function": copy.deepcopy(tools.LLM_DIRECT_SCHEMA)},
        ]
        sanitized = sanitize_tool_schemas(wrapped)
        assert len(sanitized) == 2
        for entry in sanitized:
            props = entry["function"]["parameters"]["properties"]
            assert "images" in props
            content_schema = props["messages"]["items"]["properties"]["content"]
            assert "anyOf" in content_schema


def test_public_idn_https_url_is_allowed(monkeypatch):
    monkeypatch.setattr(
        multimodal.socket,
        "getaddrinfo",
        lambda host, port: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
    )
    url = "https://münchen.example.com/pic.png"
    assert multimodal.validate_https_image_url(url) == url
