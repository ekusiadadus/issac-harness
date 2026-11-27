"""Compare two explicitly selected patches of exposed material in saved sRGB images."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def rectangle(value):
    numbers = [int(p) for p in value.split(",")]
    if len(numbers) != 4:
        raise argparse.ArgumentTypeError("expected x0,y0,x1,y1 in original image pixels")
    return tuple(numbers)


def describe(path, bounds):
    with Image.open(path) as original:
        im = ImageOps.exif_transpose(original).convert("RGB")
        x0, y0, x1, y1 = bounds
        if not (0 <= x0 < x1 <= im.width and 0 <= y0 < y1 <= im.height):
            raise ValueError(f"{path}: ROI outside {im.size}")
        region = im.crop(bounds)
        pixels = np.asarray(region)
        luma = np.asarray(region.convert("L"))
        return {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bounds_xyxy_px": bounds,
            "rgb_median_8bit": np.median(pixels, axis=(0, 1)).tolist(),
            "luma_percentiles_5_50_95_8bit": np.percentile(luma, [5, 50, 95]).tolist(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("photo", "render"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-rect", type=rectangle, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    photo = describe(args.photo, args.photo_rect)
    render = describe(args.render, args.render_rect)
    result = {
        "space": "display-referred 8-bit RGB; JPEG/PNG values, uncalibrated lighting",
        "photo": photo,
        "render": render,
        "render_minus_photo_luma_median_8bit": render["luma_percentiles_5_50_95_8bit"][1]
        - photo["luma_percentiles_5_50_95_8bit"][1],
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
