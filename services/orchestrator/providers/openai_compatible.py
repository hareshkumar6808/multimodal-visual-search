import base64
import json
import socket
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import httpx

from services.orchestrator.providers.base import Provider, ProviderError


class OpenAICompatibleProvider(Provider):
    def __init__(
        self,
        *,
        name: str,
        base_url: str | None,
        model: str | None,
        api_key: str | None,
        supports_vision: bool,
        daily_budget: int,
        timeout: float,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        family: str | None = None,
        preferred_experts: frozenset[str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            daily_budget=daily_budget,
            is_cloud=name != "local",
            family=family,
            preferred_experts=preferred_experts,
        )
        self.name = name
        self.base_url = base_url.rstrip("/") if base_url else None
        self.model = model
        self.api_key = api_key
        self.supports_vision = supports_vision
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.transport_provided = transport is not None
        request_timeout = httpx.Timeout(timeout, connect=min(timeout, 3.0))
        self.client = httpx.AsyncClient(timeout=request_timeout, transport=transport)

    @property
    def configured(self) -> bool:
        required_key_present = self.name == "local" or bool(self.api_key)
        return bool(self.base_url and self.model and required_key_present)

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({"text", "vision"} if self.supports_vision else {"text"})

    def available(self) -> bool:
        if not super().available():
            return False
        if self.name != "local" or self.transport_provided or not self.base_url:
            return True
        parsed = urlparse(self.base_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            return False
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            return False

    async def _generate(self, prompt: str, image_bytes: bytes | None, mime_type: str) -> str:
        if not self.base_url or not self.model:
            raise ProviderError(f"Provider {self.name} is not configured")
        user_content: str | list[dict[str, Any]] = prompt
        if image_bytes is not None:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
            ]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = await self.client.post(
            f"{self.base_url}/chat/completions", headers=headers, json=body
        )
        if response.is_error:
            raise ProviderError(f"Provider {self.name} returned HTTP {response.status_code}")
        try:
            return str(response.json()["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"Provider {self.name} returned an invalid response") from exc

    async def stream_generate(
        self, prompt: str, image_bytes: bytes | None = None, mime_type: str = "image/png"
    ) -> AsyncIterator[str]:
        if not self.available() or not self.base_url or not self.model:
            raise ProviderError(f"Provider {self.name} is unavailable or not configured")
        if image_bytes is not None and not self.supports_vision:
            raise ProviderError(f"Provider {self.name} does not support vision")
        user_content: str | list[dict[str, Any]] = prompt
        if image_bytes is not None:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
            ]
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        self.stats.session_requests += 1
        self.stats.requests_today += 1
        started = __import__("time").perf_counter()
        try:
            async with self.client.stream(
                "POST", f"{self.base_url}/chat/completions", headers=headers, json=body
            ) as response:
                if response.is_error:
                    raise ProviderError(
                        f"Provider {self.name} returned HTTP {response.status_code}"
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content")
                    except (KeyError, IndexError, TypeError, ValueError) as exc:
                        raise ProviderError(
                            f"Provider {self.name} returned an invalid stream"
                        ) from exc
                    if delta:
                        yield str(delta)
        except Exception:
            self.stats.failures += 1
            raise
        self.stats.total_latency_ms += round((__import__("time").perf_counter() - started) * 1000)

    async def aclose(self) -> None:
        await self.client.aclose()
