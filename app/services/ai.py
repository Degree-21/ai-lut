from __future__ import annotations
import base64
import json
import time
import requests
from typing import Dict, List, Any
from openai import OpenAI

from app.config import AppConfig, env_flag
from app.services.styles import StylePreset, get_scene_profile
from app.utils.common import normalize_style_strength, describe_style_strength

ANALYSIS_PROMPT = (
    "请作为世界顶级的电影调色师和摄影指导，对这张静帧进行深度的技术分析。请忽略画面的艺术含义，专注于视觉的物理和光学属性。请输出以下四个维度的详细报告：\n"
    "1. 【物理结构与构图】：详细描述画面中的所有物体、材质（如皮肤、金属、织物）、景深关系和空间透视。这是重建画面的骨架。\n"
    "2. 【光影逻辑】：分析主光、辅光、轮廓光的位置、硬度（硬光/柔光）和光比。指出高光和阴影的具体分布区域。\n"
    "3. 【当前色彩与影调】：分析当前的白平衡、色温倾向、饱和度水平以及主导色相。\n"
    "4. 【调色潜力分析】：指出画面中哪些区域适合推入冷色或暖色，哪些区域（如肤色、天空）需要被保护。分析画面的动态范围，指出暗部是否可以提亮或压暗。\n"
    "5. 【场景分类判别】：请严格判断画面属于【人像】（Portrait）还是【风光】（Landscape）。如果画面主体是人（包括全身、半身、特写），必须归为人像。如果画面主体是自然风光、城市建筑或无人静物，归为风光。请在最后一行单独输出分类结果，格式严格为：SCENE_CATEGORY: portrait 或 SCENE_CATEGORY: landscape。\n"
    "请输出一段极其详尽、专业且客观的描述，不要使用文学修辞，要使用技术语言。这段描述将作为后续AI重绘的唯一结构依据，必须精确。"
)

DEBUG_REQUESTS_STATE = False

def log_request(url: str, payload: Dict, attempt: int) -> None:
    redacted_url = redact_key(url)
    print(f"[debug] 请求 {attempt + 1}: POST {redacted_url}", flush=True)
    try:
        scrubbed = scrub_payload(payload)
        print(json.dumps(scrubbed, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        print(f"[debug] 请求体序列化失败: {exc}", flush=True)

def redact_key(url: str) -> str:
    if "key=" not in url:
        return url
    prefix, rest = url.split("key=", 1)
    if "&" in rest:
        _, suffix = rest.split("&", 1)
        return f"{prefix}key=***&{suffix}"
    return f"{prefix}key=***"

def scrub_payload(value):
    if isinstance(value, dict):
        scrubbed = {}
        for key, item in value.items():
            if key == "data" and isinstance(item, str):
                scrubbed[key] = f"<{len(item)} chars base64>"
            else:
                scrubbed[key] = scrub_payload(item)
        return scrubbed
    if isinstance(value, list):
        return [scrub_payload(item) for item in value]
    return value

def fetch_with_retry(
        url: str,
        payload: Dict,
        headers: Dict | None = None,
        retries: int = 5,
        backoff: float = 1.0,
) -> Dict:
    if headers is None:
        headers = {"Content-Type": "application/json"}
    for attempt in range(retries + 1):
        try:
            if DEBUG_REQUESTS_STATE or env_flag("DEBUG_REQUESTS", "0"):
                log_request(url, payload, attempt)
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
            if response.status_code == 401:
                raise RuntimeError("API授权失败 (401)，请检查 Key 或模型访问权限。")
            if not response.ok:
                detail = ""
                try:
                    detail = json.dumps(response.json(), ensure_ascii=False)
                except Exception:
                    detail = response.text
                raise RuntimeError(f"HTTP {response.status_code} 请求失败: {detail}")
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            if attempt >= retries or "401" in str(exc):
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("请求失败")

def resolve_doubao_image_input(
    image_b64: str, mime_type: str, image_url: str | None = None
) -> str:
    if image_url:
        return image_url
    if image_b64.startswith(("http://", "https://")):
        return image_b64
    return f"data:{mime_type};base64,{image_b64}"

def extract_image_bytes(result: Dict) -> bytes:
    candidates = result.get("candidates", [])
    for candidate in candidates:
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            inline = part.get("inlineData")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    raise RuntimeError("未获取到生成图像数据")

def extract_doubao_image_bytes(result: Dict) -> bytes:
    data_item = result.get("data", [{}])[0]
    b64_data = data_item.get("b64_json")
    if b64_data:
        return base64.b64decode(b64_data)
    image_url = data_item.get("url")
    if image_url:
        image_response = requests.get(image_url, timeout=120)
        image_response.raise_for_status()
        return image_response.content
    raise RuntimeError("未获取到生成图像数据")

def analyze_scene(
    image_b64: str,
    mime_type: str,
    config: AppConfig,
    retries: int,
    image_url: str | None = None,
) -> str:
    if config.analysis_model.startswith("doubao-"):
        if not config.doubao_api_key:
            raise RuntimeError("未提供豆包 API Key，请通过环境变量 ARK_API_KEY 设置。")

        image_input = resolve_doubao_image_input(image_b64, mime_type, image_url)
        client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=config.doubao_api_key,
        )

        response = client.chat.completions.create(
            model=config.analysis_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_input
                            },
                        },
                        {
                            "type": "text",
                            "text": ANALYSIS_PROMPT,
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content or "检测到复杂场景"

    if not config.api_key:
        raise RuntimeError(
            "未提供 API Key，请通过环境变量 GOOGLE_API_KEY/GEMINI_API_KEY/API_KEY 设置。"
        )

    ANALYSIS_MODEL = config.analysis_model
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            ANALYSIS_PROMPT
                        )
                    },
                    {"inlineData": {"mimeType": mime_type, "data": image_b64}},
                ]
            }
        ]
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{ANALYSIS_MODEL}:generateContent?key={config.api_key}"
    )
    result = fetch_with_retry(url, payload, retries=retries)
    return (
        result.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "检测到复杂场景")
    )

def fetch_stream_with_retry(
    url: str,
    payload: Dict,
    headers: Dict | None = None,
    retries: int = 5,
    backoff: float = 1.0,
) -> requests.Response:
    if headers is None:
        headers = {"Content-Type": "application/json"}
    for attempt in range(retries + 1):
        try:
            if DEBUG_REQUESTS_STATE or env_flag("DEBUG_REQUESTS", "0"):
                log_request(url, payload, attempt)
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=120, stream=True
            )
            if response.status_code == 401:
                raise RuntimeError("API授权失败 (401)，请检查 Key 或模型访问权限。")
            if not response.ok:
                detail = response.text
                raise RuntimeError(f"HTTP {response.status_code} 请求失败: {detail}")
            response.raise_for_status()
            return response
        except Exception as exc:
            if attempt >= retries or "401" in str(exc):
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("请求失败")

def iter_doubao_stream(response: requests.Response):
    for line in response.iter_lines(decode_unicode=False):
        if not line:
            continue
        try:
            chunk = line.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        if chunk.startswith("data:"):
            chunk = chunk[len("data:") :].strip()
        if chunk == "[DONE]":
            break
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        delta = data.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if content:
            yield content

def stream_analyze_scene(
    image_b64: str,
    mime_type: str,
    config: AppConfig,
    retries: int,
    image_url: str | None = None,
):
    if not config.analysis_model.startswith("doubao-"):
        yield analyze_scene(image_b64, mime_type, config, retries, image_url=image_url)
        return
    if not config.doubao_api_key:
        raise RuntimeError("未提供豆包 API Key，请通过环境变量 ARK_API_KEY 设置。")

    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    image_input = resolve_doubao_image_input(image_b64, mime_type, image_url)
    payload = {
        "model": config.analysis_model,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_input}},
                    {"type": "text", "text": ANALYSIS_PROMPT},
                ],
            }
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.doubao_api_key}",
    }

    response = fetch_stream_with_retry(url, payload, headers=headers, retries=retries)
    try:
        for chunk in iter_doubao_stream(response):
            yield chunk
    finally:
        response.close()

def build_style_prompt(scene_description: str, style: StylePreset, config: AppConfig) -> str:
    scene_profile = get_scene_profile(config.scene_type)
    scene_hint = str(scene_profile.get("prompt", "")).strip()
    strength = normalize_style_strength(config.style_strength)
    strength_label = describe_style_strength(strength)
    
    parts = [
        "Role: You are an expert colorist and cinematographer.",
        f"Task: Re-grade the scene described below to match the '{style.name}' style, while maintaining 100% structural fidelity.",
        "\n[Source Scene Analysis]",
        f"{scene_description}",
        "\n[Target Style Profile]",
        f"Style Name: {style.name}",
        f"Visual Characteristics: {style.description}",
        "\n[Grading Instructions]",
        "1. Structure: CRITICAL - Keep the original composition, subject identity, facial features, and physical structure exactly as described in the analysis. Do not add or remove objects.",
        "2. Lighting: Retain the original light source direction and hardness described in the analysis, but adjust the contrast and falloff to match the target style.",
        f"3. Color: Apply the target color palette. {style.description}. Blend the colors naturally with the original materials.",
        f"4. Intensity: {strength_label} color grading, refined and controlled.",
        "5. Quality: Output in 8k resolution, cinematic texture, detailed highlights and shadows.",
    ]
    
    if scene_hint:
        parts.append(f"\n[Scene Constraints]\n{scene_hint}")
        
    parts.append("\nFinal Check: Ensure the output looks like the exact same shot but with a different color grade.")
    
    return "\n".join(parts)

def generate_style_image(
        scene_description: str,
        style: StylePreset,
        image_b64: str,
        mime_type: str,
        config: AppConfig,
        retries: int,
        image_url: str | None = None,
) -> bytes:
    if config.image_model.startswith("doubao-"):
        if not config.doubao_api_key:
            raise RuntimeError("未提供豆包 API Key，请通过环境变量 ARK_API_KEY 设置。")
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        image_input = resolve_doubao_image_input(image_b64, mime_type, image_url)
        prompt = build_style_prompt(scene_description, style, config)
        payload = {
            "model": config.image_model,
            "prompt": prompt,
            "image": [image_input],
            "size": "2k",
            "watermark": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.doubao_api_key}",
        }
        result = fetch_with_retry(url, payload, headers=headers, retries=retries)
        return extract_doubao_image_bytes(result)
    else:
        if not config.api_key:
            raise RuntimeError("未提供 API Key，请通过环境变量 GOOGLE_API_KEY/GEMINI_API_KEY/API_KEY 设置。")
        prompt = build_style_prompt(scene_description, style, config)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": mime_type, "data": image_b64}},
                    ]
                }
            ],
            "generationConfig": {"responseModalities": ["IMAGE"]},
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{config.image_model}:generateContent?key={config.api_key}"
        )
        result = fetch_with_retry(url, payload, retries=retries)
        try:
            return extract_image_bytes(result)
        except RuntimeError as exc:
            raise RuntimeError(f"未获取到 {style.id} 的生成结果") from exc
