import base64
from urllib.parse import quote

import httpx

from services.orchestrator.providers.base import Provider, ProviderError


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model: str | None,
        daily_budget: int,
        timeout: float,
    ) -> None:
        super().__init__(daily_budget=daily_budget)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({"text", "vision"})

    async def _generate(self, prompt: str, image_bytes: bytes | None, mime_type: str) -> str:
        if not self.api_key or not self.model:
            raise ProviderError("Provider gemini is not configured")
        parts: list[dict[str, object]] = [{"text": prompt}]
        if image_bytes is not None:
            parts.insert(
                0,
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(image_bytes).decode("ascii"),
                    }
                },
            )
        url = f"{self.base_url}/models/{quote(self.model, safe='')}:generateContent"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json={
                    "contents": [{"role": "user", "parts": parts}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
                },
            )
        if response.is_error:
            raise ProviderError(f"Provider gemini returned HTTP {response.status_code}")
        try:
            return str(response.json()["candidates"][0]["content"]["parts"][0]["text"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("Provider gemini returned an invalid response") from exc
