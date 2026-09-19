from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass

from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError

from services.perception.models import ImageInfo

# Supported magic byte signatures
_MAGIC_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"\xff\xd8\xff", "JPEG"),
    (b"BM", "BMP"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
]


class PreflightError(ValueError):
    """Raised when screenshot bytes fail pre-flight validation."""

    pass


@dataclass(frozen=True)
class ImagePreflightResult:
    """Internal pre-flight metadata.

    Includes internal metrics like aspect_ratio, brightness, contrast, and sharpness
    without exposing them in the canonical MIR ImageInfo.
    """

    sha256: str
    width: int
    height: int
    format: str
    aspect_ratio: float
    brightness: float
    contrast: float
    sharpness: float

    def to_image_info(self) -> ImageInfo:
        """Export canonical Stage 1 ImageInfo strictly conforming to MIR v0.1."""
        return ImageInfo(
            sha256=self.sha256,
            width=self.width,
            height=self.height,
        )


def _detect_format(image_bytes: bytes) -> str:
    """Inspect magic bytes to detect image format quickly."""
    if not image_bytes:
        raise PreflightError("Image payload is empty (0 bytes).")

    for signature, fmt in _MAGIC_SIGNATURES:
        if image_bytes.startswith(signature):
            return fmt

    # WebP check (RIFF header + WEBP signature)
    if len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "WEBP"

    raise PreflightError(
        "Unsupported or unrecognized image format. Must be PNG, JPEG, WebP, BMP, or GIF."
    )


def validate_and_extract_metadata(image_bytes: bytes) -> ImagePreflightResult:
    """Run local deterministic pre-flight checks on captured screenshot bytes."""
    fmt = _detect_format(image_bytes)
    sha256_hash = hashlib.sha256(image_bytes).hexdigest()

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size

            if width <= 0 or height <= 0:
                raise PreflightError(f"Invalid image dimensions: {width}x{height}.")
            if width > 10000 or height > 10000:
                raise PreflightError(
                    "Image dimensions exceed safe processing limit: "
                    f"{width}x{height} (max 10000x10000)."
                )

            aspect_ratio = round(float(width) / float(height), 4)

            # Convert to grayscale for lightweight quality metrics
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)

            # Brightness: normalized mean intensity in [0.0, 1.0]
            mean_intensity = stat.mean[0]
            brightness = round(mean_intensity / 255.0, 4)

            # Contrast: normalized standard deviation in [0.0, 1.0]
            stddev = stat.stddev[0]
            contrast = round(min(1.0, stddev / 128.0), 4)

            # Sharpness: variance of Laplacian filter normalized
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edges)
            sharpness_val = edge_stat.var[0]
            # Normalizing sharpness indicator into roughly [0.0, 1.0]
            sharpness = round(min(1.0, math.sqrt(sharpness_val) / 64.0), 4)

            return ImagePreflightResult(
                sha256=sha256_hash,
                width=width,
                height=height,
                format=fmt,
                aspect_ratio=aspect_ratio,
                brightness=brightness,
                contrast=contrast,
                sharpness=sharpness,
            )

    except (UnidentifiedImageError, OSError) as exc:
        raise PreflightError(f"Failed to decode valid image stream: {exc}") from exc
