from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import base64
from PIL import Image

from app.config import AppConfig
from app.services.styles import STYLE_PRESETS, get_lut_space_info, StylePreset
from app.services.ai import analyze_scene, generate_style_image
from app.services.lut import generate_lut

DEBUG_REQUESTS_STATE = False

def resolve_style_ids(styles_value: str) -> List[str]:
    if not styles_value or styles_value.strip().lower() == "all":
        return [style.id for style in STYLE_PRESETS]
    wanted = {item.strip() for item in styles_value.split(",") if item.strip()}
    return [style.id for style in STYLE_PRESETS if style.id in wanted]

def run_pipeline(
    config: AppConfig,
    image_b64: str,
    source_image: Image.Image,
    mime_type: str,
    style_ids: List[str],
    image_url: str | None = None,
    analysis_override: str | None = None,
    debug_requests: bool = False,
) -> Tuple[str, List[Dict[str, object]]]:
    # Note: debug_requests handling is simplified here as global state is tricky
    # Ideally pass it down to ai services
    
    try:
        out_dir = config.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        selected_styles = [style for style in STYLE_PRESETS if style.id in style_ids]
        if not selected_styles:
            raise RuntimeError("未选择任何风格。")

        if analysis_override:
            scene_description = analysis_override
        else:
            scene_description = analyze_scene(
                image_b64, mime_type, config, config.retries, image_url=image_url
            )
        analysis_path = out_dir / "analysis.txt"
        analysis_path.write_text(scene_description, encoding="utf-8")

        results: List[Dict[str, object]] = []
        _, _, lut_file_tag, _ = get_lut_space_info(config.lut_space)
        for style in selected_styles:
            image_bytes = generate_style_image(
                scene_description,
                style,
                image_b64,
                mime_type,
                config,
                config.retries,
                image_url=image_url,
            )
            image_out = out_dir / f"ref_{style.id}.png"
            image_out.write_bytes(image_bytes)

            lut_filename = None
            lut_content = None
            if not config.no_lut:
                target_image = Image.open(image_out).convert("RGB")
                lut_out = out_dir / f"AI_Grade_{style.id}_{config.lut_size}_{lut_file_tag}.cube"
                lut_content = generate_lut(
                    source_image,
                    target_image,
                    style, # Pass the style object
                    lut_out,
                    sample_size=config.sample_size,
                    lut_size=config.lut_size,
                    lut_space=config.lut_space,
                    scene_type=config.scene_type,
                    style_strength=config.style_strength,
                )
                lut_filename = lut_out.name

            results.append(
                {
                    "id": style.id,
                    "name": style.name,
                    "description": style.description,
                    "image_bytes": image_bytes,
                    "lut_filename": lut_filename,
                    "lut_content": lut_content,
                }
            )

        return scene_description, results
    finally:
        pass
