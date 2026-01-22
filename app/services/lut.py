from __future__ import annotations
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple

from app.services.styles import StylePreset, get_scene_profile, get_lut_space_info
from app.utils.common import normalize_style_strength

class ColorTools:
    @staticmethod
    def gamma_to_linear(v: np.ndarray, gamma: float) -> np.ndarray:
        return np.power(np.maximum(0.0, v), gamma)

    @staticmethod
    def linear_to_gamma(v: np.ndarray, gamma: float) -> np.ndarray:
        return np.power(np.maximum(0.0, v), 1 / gamma)

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

def extract_histogram(
    image: Image.Image,
    sample_size: int,
    gamma: float,
) -> Dict[str, np.ndarray]:
    resized = image.resize((sample_size, sample_size), Image.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    linear = ColorTools.gamma_to_linear(pixels, gamma)
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
        style: StylePreset,
        output_path: Path,
        sample_size: int = 128,
        lut_size: int = 65,
        lut_space: str = "rec709_sdr",
        scene_type: str = "auto",
        style_strength: float = 0.7,
) -> str:
    _, gamma, file_tag, space_label = get_lut_space_info(lut_space)
    scene_profile = get_scene_profile(scene_type)
    chroma_scale = float(scene_profile.get("chroma_scale", 1.0))
    strength = normalize_style_strength(style_strength)
    
    # Enhance: Use style-specific parameters
    style_chroma_weight = style.lut_params.get("chroma_weight", 1.0)
    style_luma_weight = style.lut_params.get("luma_weight", 1.0)
    
    chroma_strength = strength * chroma_scale * style_chroma_weight
    luma_strength = strength * style_luma_weight

    src_hist = extract_histogram(source, sample_size, gamma)
    tgt_hist = extract_histogram(target, sample_size, gamma)

    lookups = {
        "L": build_lookup(get_cdf(src_hist["L"]), get_cdf(tgt_hist["L"])),
        "a": build_lookup(get_cdf(src_hist["a"]), get_cdf(tgt_hist["a"])),
        "b": build_lookup(get_cdf(src_hist["b"]), get_cdf(tgt_hist["b"])),
    }

    levels = np.linspace(0.0, 1.0, lut_size)
    linear_levels = ColorTools.gamma_to_linear(levels, gamma)

    lines: List[str] = [
        f'TITLE "AI_{style.id.upper()}_{file_tag}"',
        f"LUT_3D_SIZE {lut_size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        f"# LUT_SPACE {space_label}",
        f"# GAMMA {gamma:.3f}",
        "",
    ]

    for b in range(lut_size):
        lb = linear_levels[b]
        for g in range(lut_size):
            lg = linear_levels[g]
            for r in range(lut_size):
                lr = linear_levels[r]
                L, a, b_val = ColorTools.rgb_to_oklab(np.array(lr), np.array(lg), np.array(lb))
                L = float(L)
                a = float(a)
                b_val = float(b_val)
                l_idx = int(np.clip(np.floor(L * 255), 0, 255))
                a_idx = int(np.clip(np.floor((a + 0.4) * (255 / 0.8)), 0, 255))
                b_idx = int(np.clip(np.floor((b_val + 0.4) * (255 / 0.8)), 0, 255))
                nL = float(lookups["L"][l_idx] / 255.0)
                na = float((lookups["a"][a_idx] / (255 / 0.8)) - 0.4)
                nb = float((lookups["b"][b_idx] / (255 / 0.8)) - 0.4)
                
                # Apply strengths
                out_L = L + luma_strength * (nL - L)
                out_a = a + chroma_strength * (na - a)
                out_b = b_val + chroma_strength * (nb - b_val)
                
                fr, fg, fb = ColorTools.oklab_to_rgb(out_L, out_a, out_b)
                fr = ColorTools.linear_to_gamma(np.array(fr), gamma)
                fg = ColorTools.linear_to_gamma(np.array(fg), gamma)
                fb = ColorTools.linear_to_gamma(np.array(fb), gamma)
                fr = float(np.clip(fr, 0.0, 1.0))
                fg = float(np.clip(fg, 0.0, 1.0))
                fb = float(np.clip(fb, 0.0, 1.0))
                lines.append(f"{fr:.6f} {fg:.6f} {fb:.6f}")

    lut_text = "\n".join(lines)
    output_path.write_text(lut_text, encoding="utf-8")
    return lut_text
