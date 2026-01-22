from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

@dataclass(frozen=True)
class QiniuConfig:
    access_key: str
    secret_key: str
    bucket: str
    domain: str

def load_qiniu_config(settings: Dict[str, Any]) -> QiniuConfig | None:
    access_key = str(settings.get("qiniu_access_key", "")).strip()
    secret_key = str(settings.get("qiniu_secret_key", "")).strip()
    bucket = str(settings.get("qiniu_bucket", "")).strip()
    domain = str(settings.get("qiniu_domain", "")).strip()
    if not access_key or not secret_key or not bucket:
        return None
    if domain and not domain.startswith("http"):
        domain = f"https://{domain}"
    domain = domain.rstrip("/")
    if not domain:
        return None
    return QiniuConfig(
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        domain=domain,
    )

def upload_to_qiniu(config: QiniuConfig, file_path: Path, key: str, mime_type: str | None = None) -> str:
    try:
        from qiniu import Auth, put_file
    except Exception as exc:
        raise RuntimeError("缺少七牛 SDK，请先安装 requirements.txt。") from exc
    auth = Auth(config.access_key, config.secret_key)
    token = auth.upload_token(config.bucket, key)
    ret, info = put_file(token, key, str(file_path), mime_type=mime_type)
    if info.status_code != 200:
        raise RuntimeError(f"七牛上传失败: {info.status_code}")
    return f"{config.domain}/{key}"

def upload_run_outputs(
    out_dir: Path,
    run_id: str,
    config: QiniuConfig,
    existing: Dict[str, str] | None = None,
) -> Dict[str, str]:
    uploads: Dict[str, str] = dict(existing or {})
    for path in sorted(out_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name in uploads:
            continue
        key = f"{run_id}/{path.name}"
        mime_type = None
        if path.suffix.lower() == ".txt":
            mime_type = "text/plain; charset=utf-8"
        uploads[path.name] = upload_to_qiniu(config, path, key, mime_type=mime_type)
    return uploads

def append_qiniu_image_suffix(url: str, suffix: str = "?imageMogr2/thumbnail/200x/strip/format/jpg") -> str:
    if not suffix:
        return url
    if "?" in url:
        if suffix.startswith("?"):
            return f"{url}&{suffix[1:]}"
        return f"{url}&{suffix}"
    if suffix.startswith("?"):
        return f"{url}{suffix}"
    return f"{url}?{suffix}"
