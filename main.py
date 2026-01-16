from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from PIL import Image

from config import load_settings

# 建议将分析模型改为稳定版
ANALYSIS_MODEL = "gemini-2.5-flash"
# 修改后
IMAGE_MODEL = "gemini-2.5-flash-image"

DEBUG_REQUESTS_STATE = True


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
    style_names: List[str],
) -> Tuple[str, Dict[str, Image.Image], Dict[str, Path]]:
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_styles = [style for style in STYLE_PRESETS if style.name in style_names]
    if not selected_styles:
        raise RuntimeError("未选择任何风格。")

    scene_description = analyze_scene(image_b64, mime_type, config.api_key, config.retries)
    analysis_path = out_dir / "analysis.txt"
    analysis_path.write_text(scene_description, encoding="utf-8")

    images: Dict[str, Image.Image] = {}
    lut_files: Dict[str, Path] = {}
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
        images[style.name] = Image.open(io.BytesIO(image_bytes)).convert("RGB")

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
            lut_files[style.name] = lut_out

    return scene_description, images, lut_files


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
    style_names = [style.name for style in STYLE_PRESETS]
    scene_description, _, _ = run_pipeline(config, image_b64, source_image, mime_type, style_names)
    print(f"场景分析已保存: {config.out_dir / 'analysis.txt'}")
    print(scene_description)


def run_web() -> None:
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少依赖: gradio。请先安装 requirements.txt 后再启动。") from exc
    except Exception as exc:
        raise RuntimeError(f"gradio 导入失败: {exc}") from exc

    try:
        from gradio_client import utils as gr_client_utils
    except Exception:
        gr_client_utils = None

    if gr_client_utils is not None:
        original_schema_to_type = gr_client_utils._json_schema_to_python_type
        original_get_type = gr_client_utils.get_type

        def _patched_json_schema_to_python_type(schema, defs=None):
            if isinstance(schema, bool):
                return "Any"
            return original_schema_to_type(schema, defs)

        def _patched_get_type(schema):
            if isinstance(schema, bool):
                return "any"
            return original_get_type(schema)

        gr_client_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
        gr_client_utils.get_type = _patched_get_type

    require_dependencies()
    style_names = [style.name for style in STYLE_PRESETS]
    style_name_set = set(style_names)
    show_key_state = gr.State(False)

    def handle_generate(
        image: Image.Image | None,
        api_key: str,
        out_dir: str,
        selected_styles: List[str],
        generate_lut_flag: bool,
        debug_requests: bool,
    ):
        if image is None:
            return ("", "请先上传静帧。") + (None,) * (len(style_names) * 2)
        if not api_key:
            return ("", "未提供 API Key，请在输入框填写后再分析。") + (None,) * (len(style_names) * 2)

        global DEBUG_REQUESTS_STATE
        DEBUG_REQUESTS_STATE = bool(debug_requests)
        config = load_config()
        config = AppConfig(
            image_path=config.image_path,
            api_key=api_key.strip(),
            out_dir=Path(out_dir or "outputs"),
            styles="all",
            no_lut=not generate_lut_flag,
            sample_size=config.sample_size,
            lut_size=config.lut_size,
            retries=config.retries,
        )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        mime_type = "image/png"
        source_image = image.convert("RGB")
        selected = [name for name in selected_styles if name in style_name_set] or style_names

        try:
            scene_description, images, luts = run_pipeline(
                config, image_b64, source_image, mime_type, selected
            )
        except Exception as exc:
            return ("", str(exc)) + (None,) * (len(style_names) * 2)

        outputs: List[object] = [scene_description, ""]
        for name in style_names:
            outputs.append(images.get(name))
        for name in style_names:
            outputs.append(str(luts[name]) if name in luts else None)
        return tuple(outputs)

    with gr.Blocks(
        title="调色灵感专家",
        analytics_enabled=False,
        css=".tiny-button button{min-width:32px;height:32px;padding:0 6px;font-size:16px;line-height:1;}",
    ) as demo:
        gr.Markdown(
            "# 调色灵感专家 (Color Grading Master)\n"
            "上传静帧后点击分析，系统将基于 Gemini 分析并生成 6 种调色参考。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="上传静帧", type="pil")
                settings = load_settings()
                with gr.Row():
                    api_key_input = gr.Textbox(
                        label="API Key",
                        type="password",
                        value=str(settings.get("api_key", "")),
                        placeholder="请输入 Google API Key",
                    )
                    toggle_key_button = gr.Button("👁️", elem_classes=["tiny-button"])
                debug_requests_input = gr.Checkbox(
                    label="打印请求日志",
                    value=env_flag("DEBUG_REQUESTS", "0"),
                )
                out_dir_input = gr.Textbox(label="输出目录", value="outputs")
                styles_input = gr.CheckboxGroup(
                    choices=style_names,
                    value=style_names,
                    label="风格选择",
                )
                generate_lut_input = gr.Checkbox(label="生成 3D LUT", value=True)
                run_button = gr.Button("分析并生成", variant="primary")

            with gr.Column(scale=1):
                analysis_output = gr.Textbox(label="AI 场景分析", lines=8)
                error_output = gr.Markdown()

        with gr.Row():
            image_outputs = []
            lut_outputs = []
            for style in STYLE_PRESETS:
                with gr.Column():
                    image_outputs.append(gr.Image(label=f"{style.name} 参考图"))
                    lut_outputs.append(gr.File(label=f"{style.name} LUT"))

        run_button.click(
            handle_generate,
            inputs=[
                image_input,
                api_key_input,
                out_dir_input,
                styles_input,
                generate_lut_input,
                debug_requests_input,
            ],
            outputs=[analysis_output, error_output] + image_outputs + lut_outputs,
        )
        toggle_key_button.click(
            lambda visible: (
                not visible,
                gr.Textbox.update(type="text" if not visible else "password"),
                gr.Button.update(value="🙈" if not visible else "👁️"),
            ),
            inputs=[show_key_state],
            outputs=[show_key_state, api_key_input, toggle_key_button],
        )

    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port_env = os.getenv("GRADIO_SERVER_PORT")
    share = env_flag("GRADIO_SHARE", "0")
    if server_port_env:
        demo.launch(server_name=server_name, server_port=int(server_port_env), share=share)
    else:
        demo.launch(server_name=server_name, share=share)


def main() -> None:
    if env_flag("CLI_MODE", "0"):
        run_cli()
    else:
        run_web()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)
