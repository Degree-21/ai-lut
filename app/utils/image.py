from __future__ import annotations
import base64
import io
import os
from pathlib import Path
from typing import Tuple

from PIL import Image

HEIF_MIME_TYPES = {"image/heic", "image/heif"}
HEIF_EXTENSIONS = {".heic", ".heif"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"} | HEIF_EXTENSIONS


def guess_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    if ext in {".heic", ".heif"}:
        return "image/heic"
    return "image/png"


def extension_from_mime(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/webp":
        return ".webp"
    if mime_type in {"image/heic", "image/heif"}:
        return ".heic"
    if mime_type == "image/png":
        return ".png"
    return ""


def is_heif_image(filename: str | None, mime_type: str | None) -> bool:
    if mime_type and mime_type.lower() in HEIF_MIME_TYPES:
        return True
    suffix = Path(filename or "").suffix.lower()
    return suffix in HEIF_EXTENSIONS


def try_register_heif_opener() -> bool:
    try:
        import pillow_heif
    except Exception:
        return False
    try:
        pillow_heif.register_heif_opener()
    except Exception:
        return False
    return True


def ensure_heif_support() -> None:
    if try_register_heif_opener():
        return
    raise RuntimeError(
        "检测到 HEIC/HEIF 图片，请先安装 pillow-heif 以解码。"
    )


def resolve_image_suffix(filename: str | None, mime_type: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".jpeg":
        suffix = ".jpg"
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return suffix
    if mime_type:
        ext = extension_from_mime(mime_type.lower())
        if ext:
            return ext
    return ".png"


def normalize_image_input(
    image_bytes: bytes,
    *,
    filename: str | None,
    mime_type: str | None,
) -> Tuple[bytes, Image.Image, str, str]:
    try_register_heif_opener()
    resolved_mime = (mime_type or "").strip().lower()
    resolved_suffix = resolve_image_suffix(filename, resolved_mime)
    if is_heif_image(filename, resolved_mime) or resolved_suffix in HEIF_EXTENSIONS:
        ensure_heif_support()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), image, "image/png", ".png"
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if not resolved_mime:
        resolved_mime = guess_mime_type(Path(f"file{resolved_suffix}"))
    return image_bytes, image, resolved_mime, resolved_suffix


def load_image_base64(path: Path) -> Tuple[str, Image.Image, str]:
    image_bytes = path.read_bytes()
    mime_type = guess_mime_type(path)
    image_bytes, image, mime_type, _ = normalize_image_input(
        image_bytes, filename=path.name, mime_type=mime_type
    )
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return encoded, image, mime_type
