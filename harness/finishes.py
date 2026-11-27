"""Bake replayable decal, scuff and crease layers onto registered PBR maps."""

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .campaign import digest, write


def linear(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def srgb(v):
    return np.where(
        v <= 0.0031308, v * 12.92, 1.055 * np.maximum(v, 0) ** (1 / 2.4) - 0.055
    )


def read_map(path, mode, size):
    with Image.open(path) as im:
        return (
            np.asarray(
                im.convert(mode).resize((size, size), Image.Resampling.LANCZOS),
                np.float32,
            )
            / 255.0
        )


def bake(recipe_path, out):
    cfg = tomllib.loads(recipe_path.read_text())
    folder = recipe_path.resolve().parent
    size = cfg["resolution"]
    count = cfg["variants"]
    period = cfg["repeat_m"]
    if min(count, period) <= 0 or not 64 <= size <= 8192:
        raise ValueError("invalid repeat length, image count or size")
    paths = {
        k: (folder / cfg["maps"][k]).resolve() for k in ("base", "normal", "rough")
    }
    decals = [(folder / n).resolve() for n in cfg.get("decals", [])]
    base = linear(read_map(paths["base"], "RGB", size))
    normal = read_map(paths["normal"], "RGB", size) * 2 - 1
    rough = read_map(paths["rough"], "L", size)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "recipe_sha256": digest(recipe_path),
        "seed": cfg["seed"],
        "repeat_m": period,
        "inputs": {k: digest(p) for k, p in paths.items()},
        "decals": {p.name: digest(p) for p in decals},
        "variants": {},
    }
    for index in range(count):
        rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], index]))
        name = f"finish{index:02d}"
        sub = out / name
        sub.mkdir(exist_ok=True)
        scratches = Image.new("L", (size, size))
        draw = ImageDraw.Draw(scratches)
        marks = []
        for _ in range(int(rng.integers(*cfg["scuff_count"]))):
            center = rng.uniform(0.02, 0.98, 2) * size
            angle = rng.uniform(-np.pi, np.pi)
            length_mm = float(rng.uniform(*cfg["scuff_length_mm"]))
            width_mm = float(rng.uniform(*cfg["scuff_width_mm"]))
            along = np.array([np.cos(angle), np.sin(angle)])
            cross = np.array([-along[1], along[0]])
            length = length_mm / 1000 / period * size
            points = [
                tuple(
                    center
                    + along * t * length
                    + cross * rng.uniform(-0.035, 0.035) * length
                )
                for t in np.linspace(-0.5, 0.5, 6)
            ]
            draw.line(
                points,
                fill=int(rng.integers(90, 220)),
                width=max(1, round(width_mm / 1000 / period * size)),
                joint="curve",
            )
            marks.append(
                {
                    "center_uv": (center / size).tolist(),
                    "angle_rad": angle,
                    "length_mm": length_mm,
                    "width_mm": width_mm,
                }
            )
        wear = (
            np.asarray(
                scratches.filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32
            )
            / 255.0
        )
        pigment = (
            base * (1 - wear[:, :, None])
            + linear(np.array(cfg["wear_srgb"], dtype=np.float32)) * wear[:, :, None]
        )
        height = -cfg["groove_depth_mm"] * 0.001 * wear
        r = rough * (1 - wear) + cfg["wear_roughness"] * wear
        decal_record = None
        if decals and rng.random() < cfg.get("decal_probability", 1):
            d = decals[int(rng.integers(len(decals)))]
            with Image.open(d) as f:
                width = round(
                    rng.uniform(*cfg["decal_width_mm"]) * 0.001 * size / period
                )
                width = max(2, min(size // 2, width))
                art = f.convert("RGBA").resize(
                    (width, max(2, round(width * f.height / f.width))),
                    Image.Resampling.LANCZOS,
                )
            angle = float(rng.uniform(-24, 24))
            art = art.rotate(angle, Image.Resampling.BICUBIC, expand=True)
            if max(art.size) >= size:
                raise ValueError("decal exceeds repeat tile")
            origin = [int(rng.integers(0, size - n)) for n in art.size]
            canvas = Image.new("RGBA", (size, size))
            canvas.paste(art, tuple(origin))
            rgba = np.asarray(canvas, dtype=np.float32) / 255.0
            alpha = rgba[:, :, 3] * (1 - wear)
            pigment = (
                pigment * (1 - alpha[:, :, None])
                + linear(rgba[:, :, :3]) * alpha[:, :, None]
            )
            r = r * (1 - alpha) + cfg["decal_roughness"] * alpha
            height += cfg["decal_thickness_mm"] * 0.001 * alpha
            decal_record = {
                "source": d.name,
                "origin_uv": [v / size for v in origin],
                "width_m": width / size * period,
                "angle_deg": angle,
            }
        dy, dx = np.gradient(height, period / size)
        slopes = normal[:, :, :2] / np.maximum(normal[:, :, 2:], 0.1)
        slopes = slopes + np.stack((-dx, dy), axis=-1)
        n = np.dstack((slopes, np.ones_like(r)))
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        maps = {"base": srgb(pigment), "rough": r, "normal": n * 0.5 + 0.5}
        for k, img in maps.items():
            Image.fromarray(np.uint8(np.clip(img * 255 + 0.5, 0, 255))).save(
                sub / (k + ".png")
            )
        result["variants"][name] = {
            "scuffs": marks,
            "decal": decal_record,
            "hashes": {k: digest(sub / (k + ".png")) for k in maps},
        }
    write(out / "finishes.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = bake(args.recipe, args.out)
    print(json.dumps({"variants": len(result["variants"]), "output": str(args.out)}))
