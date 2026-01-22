from __future__ import annotations
import sys
from pathlib import Path
import base64

from app.config import load_settings, AppConfig, DEFAULT_LUT_SPACE, DEFAULT_SCENE_TYPE, DEFAULT_STYLE_STRENGTH
from app.services.styles import normalize_lut_space, normalize_scene_type
from app.utils.common import normalize_style_strength, require_dependencies
from app.utils.image import load_image_base64
from app.services.pipeline import run_pipeline, resolve_style_ids

def load_config() -> AppConfig:
    settings = load_settings()
    image_path = Path(settings.get("image_path", "input.png"))
    out_dir = Path(settings.get("out_dir", "outputs"))
    styles = str(settings.get("styles", "all"))
    no_lut = bool(settings.get("no_lut", False))
    sample_size = int(settings.get("sample_size", 128))
    lut_size = int(settings.get("lut_size", 65))
    lut_space = normalize_lut_space(settings.get("lut_space", DEFAULT_LUT_SPACE))
    scene_type = normalize_scene_type(settings.get("scene_type", DEFAULT_SCENE_TYPE))
    style_strength = normalize_style_strength(
        settings.get("style_strength", DEFAULT_STYLE_STRENGTH)
    )
    retries = int(settings.get("retries", 5))
    return AppConfig(
        image_path=image_path,
        api_key=str(settings.get("api_key", "")),
        doubao_api_key=str(settings.get("doubao_api_key", "")),
        analysis_model=str(settings.get("analysis_model", "gemini-1.5-flash")),
        image_model=str(settings.get("image_model", "gemini-1.5-flash")),
        out_dir=out_dir,
        styles=styles,
        no_lut=no_lut,
        sample_size=sample_size,
        lut_size=lut_size,
        lut_space=lut_space,
        scene_type=scene_type,
        style_strength=style_strength,
        retries=retries,
        debug_requests=bool(settings.get("debug_requests", False)),
    )

def run_cli() -> None:
    require_dependencies(require_db=False)
    config = load_config()
    if not config.api_key and not config.analysis_model.startswith("doubao-"):
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
        config,
        image_b64,
        source_image,
        mime_type,
        style_ids,
        debug_requests=config.debug_requests,
    )
    print(f"场景分析已保存: {config.out_dir / 'analysis.txt'}")
    print(scene_description)
