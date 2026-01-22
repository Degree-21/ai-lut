from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass(frozen=True)
class StylePreset:
    id: str
    name: str
    description: str
    category: str = "landscape"  # landscape or portrait
    # New: LUT generation parameters specific to this style
    lut_params: Dict[str, float] = field(default_factory=dict)

STYLE_PRESETS = [
    # --- Landscape Styles (风光) ---
    StylePreset(
        "blue_gold",
        "蓝金色调 (Blue & Gold)",
        "风光摄影最主流。主色蓝、辅色金黄/橙黄，冷天暖光，通透壮阔，适合日出日落、云海、雪山。",
        category="landscape",
        lut_params={"chroma_weight": 1.1, "luma_weight": 1.0}
    ),
    StylePreset(
        "teal_orange",
        "青橙色调 (Teal & Orange)",
        "电影风格迁移。主色青、辅色橙，冷暖强对冲，戏剧化冲击，适合大场景与强光向。风光慎用，易翻车。",
        category="landscape",
        lut_params={"chroma_weight": 1.2, "luma_weight": 1.1}
    ),
    StylePreset(
        "blue_cyan",
        "蓝青冷色调 (Blue/Cyan)",
        "极简克制。主色蓝/青，几乎无辅色，整体偏冷。视觉感受孤独、冷静、空旷，适合雪山、极地、清晨。",
        category="landscape",
        lut_params={"chroma_weight": 0.9, "luma_weight": 1.0}
    ),
    StylePreset(
        "warm_golden",
        "暖橙金色调 (Warm/Golden)",
        "暖色主导。主色黄/橙/红，少量蓝青平衡。温暖厚重，适合秋季森林、沙漠、丹霞。",
        category="landscape",
        lut_params={"chroma_weight": 1.1, "luma_weight": 1.0}
    ),
    StylePreset(
        "blue_green",
        "蓝绿色调 (Blue-Green)",
        "自然生态型。主色蓝+绿，少量黄。清新自然，适合草原、湖泊、夏季山地。",
        category="landscape",
        lut_params={"chroma_weight": 1.0, "luma_weight": 1.0}
    ),
    StylePreset(
        "muted_nordic",
        "灰蓝低饱和 (Muted/Nordic)",
        "高级感代表。主色灰蓝/灰青，极少辅色，微冷。安静克制，适合阴天、雾景、北欧风光。",
        category="landscape",
        lut_params={"chroma_weight": 0.8, "luma_weight": 0.9}
    ),
    StylePreset(
        "monotone",
        "单色倾向 (Monotone)",
        "色彩极简。单一色相主导（蓝/暖/绿），强情绪，适合雾、雪、剪影等极简场景。",
        category="landscape",
        lut_params={"chroma_weight": 0.6, "luma_weight": 1.2}
    ),
    StylePreset(
        "black_white",
        "黑白/准黑白 (B&W)",
        "脱离色彩。以明暗结构为主，强调纹理与力量，适合高反差地形与强纹理。",
        category="landscape",
        lut_params={"chroma_weight": 0.0, "luma_weight": 1.3}
    ),
    
    # --- Portrait Styles (人像) ---
    StylePreset(
        "teal_orange_portrait",
        "青橙色调 (Teal & Orange Portrait)",
        "人像使用率最高。主色青(背景/阴影)，核心色橙(肤色)。强对比，立体电影感，适合商业、街拍。",
        category="portrait",
        lut_params={"chroma_weight": 1.1, "luma_weight": 1.0}
    ),
    StylePreset(
        "warm_skin_cool_bg",
        "暖肤冷背景 (Warm Skin/Cool BG)",
        "青橙的自然版。主色暖肤色，辅色冷灰/冷蓝背景。干净耐看，适合肖像、商业人像。",
        category="portrait",
        lut_params={"chroma_weight": 1.0, "luma_weight": 1.0}
    ),
    StylePreset(
        "soft_warm_pastel",
        "日系清透 (Soft Warm/Pastel)",
        "亚洲主流。主色浅暖(米色/淡黄)，辅色低饱和绿/蓝。干净温柔空气感，适合日常、校园。",
        category="portrait",
        lut_params={"chroma_weight": 0.9, "luma_weight": 1.05}
    ),
    StylePreset(
        "creamy_beige",
        "奶油色调 (Creamy/Beige)",
        "高级感人像。主色米白、奶油黄、浅棕。偏暖，柔和轻奢，适合棚拍、女性肖像。",
        category="portrait",
        lut_params={"chroma_weight": 0.85, "luma_weight": 1.0}
    ),
    StylePreset(
        "cool_cinematic",
        "冷灰电影 (Cool/Cinematic)",
        "男性/情绪常用。主色冷灰、蓝灰，少量暖肤。克制硬朗，适合男性肖像、街头。",
        category="portrait",
        lut_params={"chroma_weight": 0.8, "luma_weight": 1.1}
    ),
    StylePreset(
        "vintage_brown",
        "暖棕复古 (Vintage/Brown)",
        "复古情绪。主色棕色、橙棕，辅色暗绿/暗蓝。弱对比，怀旧胶片感，适合复古穿搭。",
        category="portrait",
        lut_params={"chroma_weight": 0.95, "luma_weight": 0.95}
    ),
    StylePreset(
        "bw_contrast_portrait",
        "高对比黑白 (High Contrast B&W)",
        "结构型人像。强调明暗，力量感与戏剧性，适合男性、老年、纪实。",
        category="portrait",
        lut_params={"chroma_weight": 0.0, "luma_weight": 1.2}
    ),
    StylePreset(
        "monotone_portrait",
        "单色倾向人像 (Monotone Portrait)",
        "实验情绪。单一色相主导，极少辅色。强情绪风格化，适合概念人像。",
        category="portrait",
        lut_params={"chroma_weight": 0.7, "luma_weight": 1.1}
    ),
]

SCENE_TYPE_PRESETS = {
    "auto": {
        "label": "自动",
        "prompt": "",
        "chroma_scale": 1.0,
    },
    "portrait": {
        "label": "人像",
        "prompt": (
            "Portrait scene. Preserve natural skin tones, avoid hue shifts, "
            "keep skin texture clean and realistic."
        ),
        "chroma_scale": 0.55,
    },
    "landscape": {
        "label": "风光",
        "prompt": (
            "Landscape scene. Keep skies and foliage natural, avoid oversaturation."
        ),
        "chroma_scale": 1.0,
    },
    "city": {
        "label": "城市/建筑",
        "prompt": (
            "Urban/architecture scene. Keep neutral grays stable, avoid neon clipping."
        ),
        "chroma_scale": 0.9,
    },
    "night": {
        "label": "夜景",
        "prompt": (
            "Night scene. Preserve highlight detail, avoid crushed blacks and color clipping."
        ),
        "chroma_scale": 0.8,
    },
    "indoor": {
        "label": "室内",
        "prompt": (
            "Indoor scene. Maintain realistic white balance, avoid strong color casts."
        ),
        "chroma_scale": 0.8,
    },
}

LUT_SPACE_PRESETS = {
    "rec709_sdr": {
        "label": "Rec.709 SDR (BT.1886)",
        "gamma": 2.4,
        "file_tag": "709SDR",
    },
    "rec709a": {
        "label": "Rec.709-A (Apple Display Gamma)",
        "gamma": 1.96,
        "file_tag": "709A",
    },
}

def normalize_lut_space(value: str | None) -> str:
    if not value:
        return "rec709_sdr"
    key = str(value).strip().lower().replace(" ", "")
    if key in {"rec709a", "rec.709a", "rec709-a", "709a"}:
        return "rec709a"
    if key in {
        "rec709_sdr", "rec709sdr", "rec.709sdr", "rec709_24", "rec709",
        "rec.709", "709", "g24", "gamma24", "2.4", "rec709-g24",
        "bt1886", "bt.1886",
    }:
        return "rec709_sdr"
    return "rec709_sdr"

def get_lut_space_info(lut_space: str | None):
    normalized = normalize_lut_space(lut_space)
    info = LUT_SPACE_PRESETS[normalized]
    return normalized, float(info["gamma"]), str(info["file_tag"]), str(info["label"])

def normalize_scene_type(value: str | None) -> str:
    if not value:
        return "auto"
    key = str(value).strip().lower().replace(" ", "")
    aliases = {
        "auto": "auto", "automatic": "auto",
        "portrait": "portrait", "people": "portrait", "person": "portrait",
        "human": "portrait", "face": "portrait", "人像": "portrait",
        "landscape": "landscape", "scenery": "landscape", "风光": "landscape",
        "city": "city", "urban": "city", "architecture": "city", "城市": "city",
        "night": "night", "夜景": "night",
        "indoor": "indoor", "interior": "indoor", "室内": "indoor",
    }
    return aliases.get(key, "auto")

def get_scene_profile(scene_type: str | None) -> Dict[str, object]:
    normalized = normalize_scene_type(scene_type)
    return SCENE_TYPE_PRESETS[normalized]
