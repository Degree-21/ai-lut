from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from PIL import Image
from flask import Flask, jsonify, render_template, request, send_from_directory

from config import load_settings

ANALYSIS_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash"

DEBUG_REQUESTS_STATE = False


@dataclass(frozen=True)
class StylePreset:
    id: str
    name: str
    description: str


STYLE_PRESETS = [
    StylePreset("cinematic", "好莱坞电影", "青橙色调 (Teal & Orange)，高对比度，深邃阴影，极具戏剧感。"),
    StylePreset("vintage", "经典胶片", "Kodak 暖黄色调，柔和的高光溢出，低饱和度，怀旧质感。"),
    StylePreset("minimal", "清新日系", "高调照明 (High-key)，低对比度，淡蓝色或偏白影调，干净明亮。"),
    StylePreset("noir", "暗黑悬疑", "低色温，强调阴影细节，冷峻的青蓝色系，压抑且迷人。"),
    StylePreset("commercial", "时尚商业", "高饱和，色彩还原准确且明亮，光影分布均匀，质感通透。"),
    StylePreset("cyber", "赛博都市", "霓虹冷暖色差，强烈的紫色与青色碰撞，极具现代冲击力。"),
]


class ColorTools:
    @staticmethod
    def gamma_to_linear(v: np.ndarray) -> np.ndarray:
        return np.power(np.maximum(0.0, v), 2.4)

    @staticmethod
    def linear_to_gamma(v: np.ndarray) -> np.ndarray:
        return np.power(np.maximum(0.0, v), 1 / 2.4)

    @staticmethod
    def rgb_to_oklab(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
        l_ = np.cbrt(l)
        m_ = np.cbrt(m)
        s_ = np.cbrt(s)
        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720403 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        return L, a, b

    @staticmethod
    def oklab_to_rgb(L: float, a: float, b: float) -> Tuple[float, float, float]:
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b
        l = l_ * l_ * l_
        m = m_ * m_ * m_
        s = s_ * s_ * s_
        r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
        return r, g, b


@dataclass(frozen=True)
class AppConfig:
    image_path: Path
    api_key: str
    out_dir: Path
    styles: str
    no_lut: bool
    sample_size: int
    lut_size: int
    retries: int
    debug_requests: bool


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config() -> AppConfig:
    settings = load_settings()
    image_path = Path(settings.get("image_path", "input.png"))
    out_dir = Path(settings.get("out_dir", "outputs"))
    styles = str(settings.get("styles", "all"))
    no_lut = bool(settings.get("no_lut", False))
    sample_size = int(settings.get("sample_size", 128))
    lut_size = int(settings.get("lut_size", 65))
    retries = int(settings.get("retries", 5))
    return AppConfig(
        image_path=image_path,
        api_key=str(settings.get("api_key", "")),
        out_dir=out_dir,
        styles=styles,
        no_lut=no_lut,
        sample_size=sample_size,
        lut_size=lut_size,
        retries=retries,
        debug_requests=bool(settings.get("debug_requests", False)),
    )


def require_dependencies() -> None:
    missing = []
    try:
        import requests  # noqa: F401
    except Exception:
        missing.append("requests")
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        missing.append("Pillow")
    try:
        import numpy  # noqa: F401
    except Exception:
        missing.append("numpy")
    try:
        import yaml  # noqa: F401
    except Exception:
        missing.append("PyYAML")
    try:
        import flask  # noqa: F401
    except Exception:
        missing.append("Flask")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"缺少依赖: {joined}。请先安装后再运行。")


def guess_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


def load_image_base64(path: Path) -> Tuple[str, Image.Image, str]:
    image = Image.open(path).convert("RGB")
    with path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return encoded, image, guess_mime_type(path)


def fetch_with_retry(url: str, payload: Dict, retries: int = 5, backoff: float = 1.0) -> Dict:
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


def resolve_style_ids(styles_value: str) -> List[str]:
    if not styles_value or styles_value.strip().lower() == "all":
        return [style.id for style in STYLE_PRESETS]
    wanted = {item.strip() for item in styles_value.split(",") if item.strip()}
    return [style.id for style in STYLE_PRESETS if style.id in wanted]


def analyze_scene(image_b64: str, mime_type: str, api_key: str, retries: int) -> str:
    if not api_key:
        raise RuntimeError("未提供 API Key，请通过环境变量 GOOGLE_API_KEY/GEMINI_API_KEY/API_KEY 设置。")
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "请作为电影调色师和摄影指导，详细描述这张静帧。包含："
                            "1.画面构图和所有主体元素。2.现有的光源位置和性质。3.画面的物理结构。"
                            "请输出一段极其详尽的描述，用于指导另一个AI生成相同结构但不同影调的图片。"
                        )
                    },
                    {"inlineData": {"mimeType": mime_type, "data": image_b64}},
                ]
            }
        ]
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{ANALYSIS_MODEL}:generateContent?key={api_key}"
    )
    result = fetch_with_retry(url, payload, retries=retries)
    return (
        result.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "检测到复杂场景")
    )


def extract_image_bytes(result: Dict) -> bytes:
    candidates = result.get("candidates", [])
    for candidate in candidates:
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            inline = part.get("inlineData")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    raise RuntimeError("未获取到生成图像数据")


def generate_style_image(
    scene_description: str,
    style: StylePreset,
    image_b64: str,
    mime_type: str,
    api_key: str,
    retries: int,
) -> bytes:
    if not api_key:
        raise RuntimeError("未提供 API Key，请通过环境变量 GOOGLE_API_KEY/GEMINI_API_KEY/API_KEY 设置。")
    prompt = (
        f"A professional movie frame with {style.name} color grade. {style.description}."
        f" Base scene description: {scene_description}."
        " CRITICAL: Keep the original composition, subject identity, and physical structure 100% identical."
        " Only change lighting and color grading. High detail, cinematic lighting, 4k quality."
    )
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
        f"{IMAGE_MODEL}:generateContent?key={api_key}"
    )
    result = fetch_with_retry(url, payload, retries=retries)
    try:
        return extract_image_bytes(result)
    except RuntimeError as exc:
        raise RuntimeError(f"未获取到 {style.id} 的生成结果") from exc


def extract_histogram(image: Image.Image, sample_size: int) -> Dict[str, np.ndarray]:
    resized = image.resize((sample_size, sample_size), Image.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    linear = ColorTools.gamma_to_linear(pixels)
    r = linear[:, :, 0]
    g = linear[:, :, 1]
    b = linear[:, :, 2]
    L, a, b = ColorTools.rgb_to_oklab(r, g, b)
    l_idx = np.clip((L * 255).astype(np.int32), 0, 255)
    a_idx = np.clip(((a + 0.4) * (255 / 0.8)).astype(np.int32), 0, 255)
    b_idx = np.clip(((b + 0.4) * (255 / 0.8)).astype(np.int32), 0, 255)

    hist = {
        "L": np.bincount(l_idx.ravel(), minlength=256).astype(np.float64),
        "a": np.bincount(a_idx.ravel(), minlength=256).astype(np.float64),
        "b": np.bincount(b_idx.ravel(), minlength=256).astype(np.float64),
    }
    return hist


def get_cdf(hist: np.ndarray) -> np.ndarray:
    total = hist.sum()
    if total <= 0:
        return np.zeros_like(hist)
    return np.cumsum(hist) / total


def build_lookup(src_cdf: np.ndarray, tgt_cdf: np.ndarray) -> np.ndarray:
    return np.searchsorted(tgt_cdf, src_cdf, side="left").clip(0, 255)


def generate_lut(
    source: Image.Image,
    target: Image.Image,
    style_id: str,
    output_path: Path,
    sample_size: int = 128,
    lut_size: int = 65,
) -> None:
    src_hist = extract_histogram(source, sample_size)
    tgt_hist = extract_histogram(target, sample_size)

    lookups = {
        "L": build_lookup(get_cdf(src_hist["L"]), get_cdf(tgt_hist["L"])),
        "a": build_lookup(get_cdf(src_hist["a"]), get_cdf(tgt_hist["a"])),
        "b": build_lookup(get_cdf(src_hist["b"]), get_cdf(tgt_hist["b"])),
    }

    levels = np.linspace(0.0, 1.0, lut_size)
    linear_levels = ColorTools.gamma_to_linear(levels)

    lines: List[str] = [
        f'TITLE "AI_{style_id.upper()}"',
        f"LUT_3D_SIZE {lut_size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        "",
    ]

    for b in range(lut_size):
        lb = linear_levels[b]
        for g in range(lut_size):
            lg = linear_levels[g]
            for r in range(lut_size):
                lr = linear_levels[r]
                L, a, b_val = ColorTools.rgb_to_oklab(np.array(lr), np.array(lg), np.array(lb))
                l_idx = int(np.clip(np.floor(L * 255), 0, 255))
                a_idx = int(np.clip(np.floor((a + 0.4) * (255 / 0.8)), 0, 255))
                b_idx = int(np.clip(np.floor((b_val + 0.4) * (255 / 0.8)), 0, 255))
                nL = lookups["L"][l_idx] / 255.0
                na = (lookups["a"][a_idx] / (255 / 0.8)) - 0.4
                nb = (lookups["b"][b_idx] / (255 / 0.8)) - 0.4
                fr, fg, fb = ColorTools.oklab_to_rgb(nL, na, nb)
                fr = ColorTools.linear_to_gamma(np.array(fr))
                fg = ColorTools.linear_to_gamma(np.array(fg))
                fb = ColorTools.linear_to_gamma(np.array(fb))
                fr = float(np.clip(fr, 0.0, 1.0))
                fg = float(np.clip(fg, 0.0, 1.0))
                fb = float(np.clip(fb, 0.0, 1.0))
                lines.append(f"{fr:.6f} {fg:.6f} {fb:.6f}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    config: AppConfig,
    image_b64: str,
    source_image: Image.Image,
    mime_type: str,
    style_ids: List[str],
    debug_requests: bool = False,
) -> Tuple[str, List[Dict[str, object]]]:
    global DEBUG_REQUESTS_STATE
    previous_debug = DEBUG_REQUESTS_STATE
    DEBUG_REQUESTS_STATE = debug_requests or config.debug_requests
    try:
        out_dir = config.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        selected_styles = [style for style in STYLE_PRESETS if style.id in style_ids]
        if not selected_styles:
            raise RuntimeError("未选择任何风格。")

        scene_description = analyze_scene(image_b64, mime_type, config.api_key, config.retries)
        analysis_path = out_dir / "analysis.txt"
        analysis_path.write_text(scene_description, encoding="utf-8")

        results: List[Dict[str, object]] = []
        for style in selected_styles:
            image_bytes = generate_style_image(
                scene_description,
                style,
                image_b64,
                mime_type,
                config.api_key,
                config.retries,
            )
            image_out = out_dir / f"ref_{style.id}.png"
            image_out.write_bytes(image_bytes)

            lut_filename = None
            if not config.no_lut:
                target_image = Image.open(image_out).convert("RGB")
                lut_out = out_dir / f"AI_Grade_{style.id}_{config.lut_size}.cube"
                generate_lut(
                    source_image,
                    target_image,
                    style.id,
                    lut_out,
                    sample_size=config.sample_size,
                    lut_size=config.lut_size,
                )
                lut_filename = lut_out.name

            results.append(
                {
                    "id": style.id,
                    "name": style.name,
                    "description": style.description,
                    "image_bytes": image_bytes,
                    "lut_filename": lut_filename,
                }
            )

        return scene_description, results
    finally:
        DEBUG_REQUESTS_STATE = previous_debug


def run_cli() -> None:
    require_dependencies()
    config = load_config()
    if not config.api_key:
        print(
            "未提供 API Key，已跳过分析流程。设置环境变量后重新运行即可开始生成。",
            file=sys.stderr,
        )
        return
    if not config.image_path.exists():
        raise FileNotFoundError(f"找不到输入图片: {config.image_path}，请设置 IMAGE_PATH。")

    image_b64, source_image, mime_type = load_image_base64(config.image_path)
    style_ids = resolve_style_ids(config.styles)
    scene_description, _ = run_pipeline(
        config, image_b64, source_image, mime_type, style_ids, debug_requests=config.debug_requests
    )
    print(f"场景分析已保存: {config.out_dir / 'analysis.txt'}")
    print(scene_description)


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")

    @app.route("/")
    def index():
        settings = load_settings()
        return render_template("index.html", api_key=str(settings.get("api_key", "")))

    @app.route("/api/generate", methods=["POST"])
    def api_generate():
        require_dependencies()
        settings = load_settings()
        api_key = request.form.get("api_key", "").strip() or str(settings.get("api_key", ""))
        if not api_key:
            return jsonify({"error": "未提供 API Key，请先填写。"}), 400

        image_file = request.files.get("image")
        if image_file is None:
            return jsonify({"error": "请先上传静帧。"}), 400

        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({"error": "上传的图片为空。"}), 400

        generate_lut_flag = request.form.get("generate_lut", "1").lower() in {"1", "true", "yes", "on"}
        debug_requests = request.form.get("debug_requests", "0").lower() in {"1", "true", "yes", "on"}
        style_ids = request.form.getlist("styles")
        if not style_ids:
            style_ids = [style.id for style in STYLE_PRESETS]

        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        base_out_dir = Path(settings.get("out_dir", "outputs"))
        out_dir = base_out_dir / run_id

        config = AppConfig(
            image_path=Path(settings.get("image_path", "input.png")),
            api_key=api_key,
            out_dir=out_dir,
            styles=",".join(style_ids),
            no_lut=not generate_lut_flag,
            sample_size=int(settings.get("sample_size", 128)),
            lut_size=int(settings.get("lut_size", 65)),
            retries=int(settings.get("retries", 5)),
            debug_requests=bool(settings.get("debug_requests", False)),
        )

        mime_type = image_file.mimetype or "image/png"
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        source_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        try:
            scene_description, results = run_pipeline(
                config, image_b64, source_image, mime_type, style_ids, debug_requests=debug_requests
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        payload_results = []
        for item in results:
            image_bytes = item["image_bytes"]
            image_base64 = base64.b64encode(image_bytes).decode("ascii")
            lut_filename = item.get("lut_filename")
            lut_url = f"/api/download/{out_dir.name}/{lut_filename}" if lut_filename else None
            payload_results.append(
                {
                    "id": item["id"],
                    "name": item["name"],
                    "description": item["description"],
                    "image": f"data:image/png;base64,{image_base64}",
                    "lut_url": lut_url,
                }
            )

        return jsonify(
            {
                "analysis": scene_description,
                "results": payload_results,
                "run_id": out_dir.name,
            }
        )

    @app.route("/api/download/<run_id>/<path:filename>")
    def api_download(run_id: str, filename: str):
        base_out_dir = Path(load_settings().get("out_dir", "outputs"))
        safe_dir = (base_out_dir / run_id).resolve()
        file_path = (safe_dir / filename).resolve()
        if safe_dir not in file_path.parents:
            return jsonify({"error": "非法路径。"}), 400
        if not file_path.exists():
            return jsonify({"error": "文件不存在。"}), 404
        return send_from_directory(safe_dir, filename, as_attachment=True)

    return app


def main() -> None:
    if env_flag("CLI_MODE", "0"):
        run_cli()
    else:
        app = create_app()
        host = os.getenv("HOST", "127.0.0.1")
        port = int(os.getenv("PORT", "7860"))
        app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)
