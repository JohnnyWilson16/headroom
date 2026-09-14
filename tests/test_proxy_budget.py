"""Integration tests for proxy budget enforcement (Issue #3374).

Verifies that `--budget` limits are strictly enforced on all generation routes:
- OpenAI chat completions (`/v1/chat/completions`)
- OpenAI responses (`/v1/responses` HTTP and WebSocket)
- Gemini generate content (`/v1beta/models/{model}:generateContent`)
- Gemini stream generate content (`/v1beta/models/{model}:streamGenerateContent`)
- Anthropic messages (`/v1/messages`) for parity
"""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from headroom.proxy.server import ProxyConfig, create_app


class _CountingMockTransport(httpx.AsyncBaseTransport):
    """Mock transport that tracks calls and returns valid mock LLM responses."""

    def __init__(self) -> None:
        self.call_count = 0
        self.captured_requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        self.captured_requests.append(request)

        url_str = str(request.url)

        if "chat/completions" in url_str:
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )

        if "responses" in url_str:
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "id": "resp-mock",
                    "object": "response",
                    "created": 1234567890,
                    "model": "gpt-4o",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Hello!"}],
                        }
                    ],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )

        if "generateContent" in url_str:
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "Hello from Gemini!"}],
                                "role": "model",
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 15,
                    },
                },
            )

        # Anthropic messages
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello from Claude!"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                },
            },
        )


def _build_proxy_client(
    *,
    budget_limit_usd: float | None = None,
    budget_period: str = "daily",
    cost_tracking_enabled: bool = True,
) -> tuple[TestClient, _CountingMockTransport]:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=cost_tracking_enabled,
        budget_limit_usd=budget_limit_usd,
        budget_period=budget_period,  # type: ignore[arg-type]
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        anthropic_api_url="https://api.anthropic.test",
        openai_api_url="https://api.openai.test",
        gemini_api_url="https://api.gemini.test",
    )
    app = create_app(config)
    transport = _CountingMockTransport()
    proxy = app.state.proxy
    proxy.http_client = httpx.AsyncClient(transport=transport)
    return TestClient(app), transport


def test_zero_budget_blocks_openai_chat_completions() -> None:
    client, transport = _build_proxy_client(budget_limit_usd=0.0)

    response = client.post(
        "/v1/chat/completions",
        headers={"authorization": "Bearer sk-test"},
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "Budget exceeded for daily period"
    assert transport.call_count == 0


def test_zero_budget_blocks_openai_responses() -> None:
    client, transport = _build_proxy_client(budget_limit_usd=0.0)

    response = client.post(
        "/v1/responses",
        headers={"authorization": "Bearer sk-test"},
        json={
            "model": "gpt-4o",
            "input": "Hello",
        },
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "Budget exceeded for daily period"
    assert transport.call_count == 0


def test_zero_budget_blocks_gemini_generate_content() -> None:
    client, transport = _build_proxy_client(budget_limit_usd=0.0)

    response = client.post(
        "/v1beta/models/gemini-1.5-pro:generateContent",
        headers={"x-goog-api-key": "test-gemini-key"},
        json={
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        },
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "Budget exceeded for daily period"
    assert transport.call_count == 0


def test_zero_budget_blocks_gemini_stream_generate_content() -> None:
    client, transport = _build_proxy_client(budget_limit_usd=0.0)

    response = client.post(
        "/v1beta/models/gemini-1.5-pro:streamGenerateContent",
        headers={"x-goog-api-key": "test-gemini-key"},
        json={
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        },
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "Budget exceeded for daily period"
    assert transport.call_count == 0


def test_zero_budget_blocks_anthropic_messages_parity() -> None:
    client, transport = _build_proxy_client(budget_limit_usd=0.0)

    response = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "sk-ant-test",
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        },
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "Budget exceeded for daily period"
    assert transport.call_count == 0


def test_unlimited_budget_allows_all_generation_routes() -> None:
    client, transport = _build_proxy_client(budget_limit_usd=None)

    # OpenAI chat
    resp_chat = client.post(
        "/v1/chat/completions",
        headers={"authorization": "Bearer sk-test"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert resp_chat.status_code == 200

    # OpenAI responses
    resp_resp = client.post(
        "/v1/responses",
        headers={"authorization": "Bearer sk-test"},
        json={"model": "gpt-4o", "input": "Hello"},
    )
    assert resp_resp.status_code == 200

    # Gemini generateContent
    resp_gem = client.post(
        "/v1beta/models/gemini-1.5-pro:generateContent",
        headers={"x-goog-api-key": "test-key"},
        json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
    )
    assert resp_gem.status_code == 200

    assert transport.call_count == 3


def test_zero_budget_websocket_preflight_rejected() -> None:
    client, _ = _build_proxy_client(budget_limit_usd=0.0)

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/v1/responses"):
            pass

    assert exc_info.value.code == 1008


def test_dynamic_budget_exhaustion_blocks_all_providers() -> None:
    client, transport = _build_proxy_client(budget_limit_usd=0.05)

    # First request succeeds
    resp1 = client.post(
        "/v1/chat/completions",
        headers={"authorization": "Bearer sk-test"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert resp1.status_code == 200
    assert transport.call_count == 1

    # Simulate accumulated spend that exceeds the $0.05 budget
    proxy = client.app.state.proxy  # type: ignore[attr-defined]
    assert proxy.cost_tracker is not None
    proxy.cost_tracker.record_tokens(
        model="gpt-4o",
        tokens_saved=0,
        tokens_sent=100_000,
        output_tokens=50_000,
    )

    allowed, _ = proxy.cost_tracker.check_budget()
    assert not allowed

    # Now all provider generation routes must reject with 429
    routes_and_payloads = [
        (
            "/v1/chat/completions",
            {"authorization": "Bearer sk-test"},
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        ),
        (
            "/v1/responses",
            {"authorization": "Bearer sk-test"},
            {"model": "gpt-4o", "input": "Hi"},
        ),
        (
            "/v1beta/models/gemini-1.5-pro:generateContent",
            {"x-goog-api-key": "test-key"},
            {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]},
        ),
        (
            "/v1/messages",
            {"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
            {
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
            },
        ),
    ]

    for route, headers, body in routes_and_payloads:
        resp = client.post(route, headers=headers, json=body)
        assert resp.status_code == 429
        assert "Budget exceeded for daily period" in resp.json()["detail"]

    # Upstream should not have received any additional calls after budget exhaustion
    assert transport.call_count == 1
