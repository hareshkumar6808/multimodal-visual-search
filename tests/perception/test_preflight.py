from __future__ import annotations

import hashlib
import io

import pytest
from PIL import Image

from services.perception.preflight import PreflightError, validate_and_extract_metadata


def make_test_image(
    width: int = 200,
    height: int = 100,
    color: tuple[int, int, int] = (255, 255, 255),
    fmt: str = "PNG",
) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_preflight_valid_png() -> None:
    data = make_test_image(300, 150, fmt="PNG")
    result = validate_and_extract_metadata(data)

    assert result.format == "PNG"
    assert result.width == 300
    assert result.height == 150
    assert result.aspect_ratio == 2.0
    assert result.sha256 == hashlib.sha256(data).hexdigest()
    assert 0.0 <= result.brightness <= 1.0
    assert 0.0 <= result.contrast <= 1.0


def test_preflight_valid_jpeg() -> None:
    data = make_test_image(400, 200, fmt="JPEG")
    result = validate_and_extract_metadata(data)

    assert result.format == "JPEG"
    assert result.width == 400
    assert result.height == 200


def test_preflight_valid_bmp() -> None:
    data = make_test_image(100, 100, fmt="BMP")
    result = validate_and_extract_metadata(data)
    assert result.format == "BMP"


def test_preflight_empty_bytes_rejected() -> None:
    with pytest.raises(PreflightError, match="empty"):
        validate_and_extract_metadata(b"")


def test_preflight_corrupt_bytes_rejected() -> None:
    with pytest.raises(PreflightError):
        validate_and_extract_metadata(b"not-an-image-payload-data")


def test_preflight_corrupt_png_header_rejected() -> None:
    # Starts with PNG magic header but has corrupted body
    corrupt_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    with pytest.raises(PreflightError):
        validate_and_extract_metadata(corrupt_png)


def test_preflight_canonical_image_info_omits_aspect_ratio() -> None:
    data = make_test_image(800, 500)
    result = validate_and_extract_metadata(data)
    info = result.to_image_info()

    assert info.width == 800
    assert info.height == 500
    assert info.sha256 == hashlib.sha256(data).hexdigest()
    # Ensure aspect_ratio is not a field in ImageInfo
    assert not hasattr(info, "aspect_ratio")
    dumped = info.model_dump()
    assert "aspect_ratio" not in dumped
    assert set(dumped.keys()) == {"sha256", "width", "height"}
