from __future__ import annotations
import json
import time
import requests
from typing import Dict, Any, Generator

from app.config import AppConfig, env_flag

def log_request(url: str, payload: Dict, attempt: int) -> None:
    print(f"[debug] GRSAI Request {attempt + 1}: POST {url}", flush=True)
    try:
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        print(f"[debug] Payload serialization failed: {exc}", flush=True)

def fetch_stream_with_retry(
    url: str,
    payload: Dict,
    headers: Dict | None = None,
    retries: int = 3,
    backoff: float = 1.0,
) -> requests.Response:
    if headers is None:
        headers = {"Content-Type": "application/json"}
    
    for attempt in range(retries + 1):
        try:
            if env_flag("DEBUG_REQUESTS", "0"):
                log_request(url, payload, attempt)
                
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=120, stream=True
            )
            
            if response.status_code == 401:
                raise RuntimeError("GRSAI API Unauthorized (401)")
                
            if not response.ok:
                detail = response.text
                raise RuntimeError(f"HTTP {response.status_code} Failed: {detail}")
                
            response.raise_for_status()
            return response
        except Exception as exc:
            if attempt >= retries or "401" in str(exc):
                raise
            time.sleep(backoff)
            backoff *= 2
            
    raise RuntimeError("Request failed after retries")

def generate_image_stream(
    prompt: str,
    config: AppConfig,
    size: str = "1:1",
    variants: int = 1,
    urls: list[str] | None = None,
) -> Generator[str, None, None]:
    if not config.grsai_api_key:
        raise RuntimeError("GRSAI API Key not configured")

    payload = {
        "model": config.grsai_model,
        "prompt": prompt,
        "size": size,
        "variants": variants,
        "shutProgress": False
    }
    
    if urls:
        payload["urls"] = urls

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.grsai_api_key}"
    }

    response = fetch_stream_with_retry(
        config.grsai_api_url,
        payload,
        headers=headers,
        retries=config.retries
    )

    try:
        for line in response.iter_lines(decode_unicode=False):
            if not line:
                continue
            try:
                chunk = line.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
                
            if chunk.startswith("data:"):
                chunk = chunk[len("data:") :].strip()
            
            if not chunk or chunk == "[DONE]":
                continue

            yield chunk + "\n"
            
    finally:
        response.close()
