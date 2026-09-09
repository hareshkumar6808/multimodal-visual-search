from __future__ import annotations

import struct


class InvalidImageError(ValueError):
    pass


def validate_image(image_bytes: bytes) -> str:
    """Validate a small set of desktop-capture formats and return canonical MIME type."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        if (
            len(image_bytes) < 45
            or image_bytes[8:12] != b"\x00\x00\x00\x0d"
            or image_bytes[12:16] != b"IHDR"
            or image_bytes[-12:-8] != b"\x00\x00\x00\x00"
            or image_bytes[-8:-4] != b"IEND"
        ):
            raise InvalidImageError("invalid PNG structure")
        width, height = struct.unpack(">II", image_bytes[16:24])
        if width == 0 or height == 0:
            raise InvalidImageError("invalid PNG dimensions")
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        if len(image_bytes) < 4 or not image_bytes.endswith(b"\xff\xd9"):
            raise InvalidImageError("invalid JPEG structure")
        return "image/jpeg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        if len(image_bytes) < 10:
            raise InvalidImageError("invalid GIF structure")
        return "image/gif"
    if image_bytes.startswith(b"BM"):
        if len(image_bytes) < 26:
            raise InvalidImageError("invalid BMP structure")
        return "image/bmp"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        if len(image_bytes) < 16:
            raise InvalidImageError("invalid WebP structure")
        return "image/webp"
    raise InvalidImageError("unsupported or malformed image content")
