import base64
from typing import Any

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
    ) -> None:
        super().__init__(daily_budget=daily_budget)
        self.name = name
        self.base_url = base_url.rstrip("/") if base_url else None
        self.model = model
        self.api_key = api_key
        self.supports_vision = supports_vision
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        required_key_present = self.name == "local" or bool(self.api_key)
        return bool(self.base_url and self.model and required_key_present)

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({"text", "vision"} if self.supports_vision else {"text"})

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
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=body
            )
        if response.is_error:
            raise ProviderError(f"Provider {self.name} returned HTTP {response.status_code}")
        try:
            return str(response.json()["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"Provider {self.name} returned an invalid response") from exc
