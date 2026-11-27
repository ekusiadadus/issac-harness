"""Review exported instance contours, native ranges and RGB at their original resolution."""

import argparse
import colorsys
import html
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .campaign import inspect_frame, read, write


def draw_samples(root, out, indices, max_depth):
    if max_depth <= 0 or not indices or len(indices) != len(set(indices)):
        raise ValueError("choose unique frames and a positive range limit")
    out.mkdir(parents=True, exist_ok=True)
    header = read(root / "campaign.json")
    sheets = []
    sections = []
    reviews = {}
    for idx in indices:
        folder = root / "frames" / f"{idx:08d}"
        frame = inspect_frame(folder, header)
        with Image.open(folder / "rgb.png") as im:
            rgb = im.convert("RGB")
        with Image.open(folder / "instances.png") as im:
            ids = np.array(im)
        composite = np.array(rgb)
        for obj in frame["objects"]:
            ident = obj["instance_id"]
            mask = ids == ident
            binary = Image.fromarray(np.uint8(mask) * 255)
            edge = mask & (np.asarray(binary.filter(ImageFilter.MinFilter(5))) == 0)
            color = (
                np.array(
                    colorsys.hsv_to_rgb(
                        (0.10 + obj["category_id"] * 0.61803) % 1, 0.85, 1
                    )
                )
                * 255
            )
            composite[edge] = color
        overlay = Image.fromarray(composite)
        draw = ImageDraw.Draw(overlay)
        for obj in frame["objects"]:
            x, y, w, h = obj["bbox_xywh"]
            draw.rectangle((x, y, x + w - 1, y + h - 1), outline="#fff", width=2)
            draw.text(
                (x + 2, max(1, y - 14)),
                f"{obj['class']} / {obj['instance_id']}",
                fill="white",
                stroke_fill="black",
                stroke_width=2,
            )
        with np.load(folder / "range_m.npz") as f:
            z = f["range_m"]
        dist = (
            np.minimum(np.where(np.isfinite(z) & (z > 0), z, max_depth), max_depth)
            / max_depth
        )
        depth = Image.fromarray(np.uint8((1 - dist) * 255)).convert("RGB")
        width, height = rgb.size
        strip = Image.new("RGB", (3 * width, height))
        for i, panel in enumerate((rgb, overlay, depth)):
            strip.paste(panel, (width * i, 0))
        name = f"frame-{idx:08d}.jpg"
        strip.save(out / name, quality=92)
        preview = strip.copy()
        preview.thumbnail((1020, 340))
        sheets.append(
            (
                preview,
                f"{idx}   {frame['sample']['camera']['id']}   {frame['sample']['layout']}",
            )
        )
        src = html.escape(os.path.relpath(folder / "rgb.png", out), quote=True)
        desc = html.escape(sheets[-1][1])
        sections.append(
            f'<section><h2>{desc}</h2><a href="{name}"><img src="{name}" alt="{desc}"></a><p><a href="{src}">RGB 原寸 PNG</a></p></section>'
        )
        reviews[str(idx)] = {
            "rgb_sha256": frame["sha256"]["rgb.png"],
            "observation": "",
            "decision": "pending",
        }
    sh = max(i.height for i, t in sheets) + 26
    sheet = Image.new("RGB", (1020, sh * len(sheets)), (24, 27, 31))
    draw = ImageDraw.Draw(sheet)
    for i, (img, title) in enumerate(sheets):
        draw.text((6, i * sh + 5), title, fill="white")
        sheet.paste(img, (0, i * sh + 24))
    sheet.save(out / "contact-sheet.jpg", quality=90)
    (out / "index.html").write_text(
        '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RTX フレーム検査</title><style>body{max-width:1400px;margin:2em auto;padding:0 1em;font-family:system-ui}img{width:100%;height:auto}h2{font-size:1.15em}</style><h1>RTX フレーム検査</h1><p>左から RGB、可視輪郭と矩形、カメラからの距離。距離は白が近く、黒が '
        + str(max_depth)
        + " m 以上または無効画素。</p>"
        + "".join(sections)
        + "</html>"
    )
    if not (out / "review.json").exists():
        write(
            out / "review.json",
            {"campaign_key": header["campaign_key"], "frames": reviews},
        )


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--indices", nargs="+", type=int, required=True)
    p.add_argument("--max-depth", type=float, default=3.5)
    a = p.parse_args(argv)
    draw_samples(a.root, a.out, a.indices, a.max_depth)
