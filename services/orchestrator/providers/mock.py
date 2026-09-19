from services.orchestrator.providers.base import Provider, ProviderError


class MockProvider(Provider):
    """Deterministic provider used only by tests."""

    def __init__(
        self,
        name: str = "mock",
        *,
        vision: bool = True,
        response: str = "Mock answer",
        fail: bool = False,
        is_cloud: bool | None = None,
    ) -> None:
        super().__init__(is_cloud=name != "local" if is_cloud is None else is_cloud)
        self.name = name
        self.vision = vision
        self.response = response
        self.fail = fail
        self.received_images: list[bool] = []
        self.prompts: list[str] = []

    @property
    def configured(self) -> bool:
        return True

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({"text", "vision"} if self.vision else {"text"})

    async def _generate(self, prompt: str, image_bytes: bytes | None, mime_type: str) -> str:
        self.prompts.append(prompt)
        self.received_images.append(image_bytes is not None)
        if self.fail:
            raise ProviderError("planned mock failure")
        return self.response
