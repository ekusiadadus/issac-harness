"""Check a published or measured plumb-bob color profile against lossless calibration board photos."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from .campaign import digest, read, write


def score(profile, paths, pattern):
    cfg = read(profile)
    K = np.array(cfg["K"], dtype=np.float64).reshape(3, 3)
    D = np.array(cfg["D"], dtype=np.float64)
    xx, yy = np.meshgrid(np.arange(pattern[0]), np.arange(pattern[1]))
    obj = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)].astype(np.float32)
    for_fit = ((xx + yy) % 2 == 0).ravel()
    photos = []
    pooled = []
    for path in paths:
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if im is None or im.shape != (cfg["height"], cfg["width"]):
            raise ValueError(f"wrong photo or resolution: {path}")
        found, coords = cv2.findChessboardCornersSB(
            im, tuple(pattern), flags=cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            raise ValueError(f"no complete board: {path}")
        coords = coords[:, 0, :]
        ok, r, t = cv2.solvePnP(
            obj[for_fit], coords[for_fit], K, D, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            raise ValueError(f"pose solve failed: {path}")
        pred = cv2.projectPoints(obj[~for_fit], r, t, K, D)[0][:, 0, :]
        error = np.linalg.norm(coords[~for_fit] - pred, axis=1)
        pooled.extend(error.tolist())
        photos.append(
            {
                "photo": path.name,
                "sha256": digest(path),
                "pose_points": int(for_fit.sum()),
                "scored_points": len(error),
                "median_pixel_error": float(np.median(error)),
                "p95_pixel_error": float(np.quantile(error, 0.95)),
            }
        )
    return {
        "profile_sha256": digest(profile),
        "inner_pattern": pattern,
        "images": photos,
        "scored_points": len(pooled),
        "median_pixel_error": float(np.median(pooled)),
        "p95_pixel_error": float(np.quantile(pooled, 0.95)),
        "method": "Hold out alternate detected checkerboard intersections from pose estimation. The lens coefficients are fixed.",
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("images", nargs="+", type=Path)
    cli.add_argument("--profile", type=Path, required=True)
    cli.add_argument("--pattern", nargs=2, type=int, required=True)
    cli.add_argument("--out", required=True, type=Path)
    a = cli.parse_args()
    result = score(a.profile, a.images, a.pattern)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write(a.out, result)
    print(
        result["scored_points"], result["median_pixel_error"], result["p95_pixel_error"]
    )
