"""Reusable color camera transfer for a pinhole source and Brown-Conrady target.

Run ``python -m harness.color_sensor --help``. Pixel coordinates follow OpenCV:
pixel centers are integer (u, v); D order is k1, k2, p1, p2, k3.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(raw: object) -> np.ndarray:
    if isinstance(raw, dict):
        raw = raw["data"]
    matrix = np.asarray(raw, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("K must be a finite 3x3 positive-focal-length matrix")
    if not np.allclose(matrix[2], [0, 0, 1]) or abs(matrix[0, 1]) > 1e-12:
        raise ValueError("only ordinary zero-skew OpenCV camera matrices are supported")
    return matrix


@dataclass(frozen=True)
class ColorProfile:
    name: str
    provenance: str
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray

    @classmethod
    def from_json(cls, path: str | Path) -> "ColorProfile":
        obj = json.loads(Path(path).read_text())
        model = obj.get("distortion_model", "plumb_bob")
        if model != "plumb_bob":
            raise ValueError("only OpenCV/ROS plumb_bob is supported")
        K = _matrix(obj.get("K", obj.get("camera_matrix")))
        raw_D = obj.get("D", obj.get("distortion_coefficients"))
        if isinstance(raw_D, dict):
            raw_D = raw_D["data"]
        D = np.asarray(raw_D, dtype=np.float64)
        if D.shape not in ((4,), (5,)) or not np.isfinite(D).all():
            raise ValueError("D must be k1,k2,p1,p2[,k3]; rational/fisheye models are unsupported")
        if len(D) == 4:
            D = np.r_[D, 0.0]
        width = int(obj.get("width", obj.get("image_width", 0)))
        height = int(obj.get("height", obj.get("image_height", 0)))
        if width <= 0 or height <= 0:
            raise ValueError("positive target width and height required")
        return cls(obj["name"], obj["provenance"], width, height, K, D)


def distort_normalized(x: np.ndarray, y: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV five-coefficient radial and tangential forward projection."""
    k1, k2, p1, p2, k3 = D
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    return (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def undistort_normalized(xd: np.ndarray, yd: np.ndarray, D: np.ndarray,
                         iterations: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterative inverse, with a forward reprojection check for invalid rays."""
    x, y = xd.copy(), yd.copy()
    k1, k2, p1, p2, k3 = D
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        tx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        ty = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        safe = np.abs(radial) > 1e-8
        x = np.where(safe, (xd - tx) / radial, np.nan)
        y = np.where(safe, (yd - ty) / radial, np.nan)
    check_x, check_y = distort_normalized(x, y, D)
    good = np.isfinite(x) & np.isfinite(y) & (np.hypot(check_x - xd, check_y - yd) < 1e-5)
    return x, y, good


def source_map(profile: ColorProfile, source_K: np.ndarray, source_shape: tuple[int, int]
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Target distorted pixels to source ideal pinhole pixels."""
    K = _matrix(source_K)
    h, w = source_shape
    vv, uu = np.indices((profile.height, profile.width), dtype=np.float64)
    xd = (uu - profile.K[0, 2]) / profile.K[0, 0]
    yd = (vv - profile.K[1, 2]) / profile.K[1, 1]
    x, y, converged = undistort_normalized(xd, yd, profile.D)
    sx = x * K[0, 0] + K[0, 2]
    sy = y * K[1, 1] + K[1, 2]
    valid = converged & (sx >= 0) & (sy >= 0) & (sx <= w - 1) & (sy <= h - 1)
    return sx, sy, valid


def _srgb_to_linear(value: np.ndarray) -> np.ndarray:
    v = value / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(value: np.ndarray) -> np.ndarray:
    v = np.clip(value, 0, 1)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055) * 255


def transfer(rgb: np.ndarray, classes: np.ndarray, instances: np.ndarray,
             range_m: np.ndarray, source_K: np.ndarray, profile: ColorProfile, *,
             seed: int = 0, noise: dict | None = None
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return registered RGB, classes, instances, radial range, and valid mask."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("source RGB must be HxWx3 uint8")
    h, w = rgb.shape[:2]
    if any(a.shape != (h, w) for a in (classes, instances, range_m)):
        raise ValueError("all source planes must share the RGB pixel grid")
    if not np.issubdtype(range_m.dtype, np.floating):
        raise ValueError("radial range must be floating point metres")
    sx, sy, valid = source_map(profile, source_K, (h, w))
    coords = [np.where(valid, sy, -1), np.where(valid, sx, -1)]
    sampled = np.stack([map_coordinates(rgb[:, :, c].astype(np.float64), coords,
                                        order=1, mode="constant", cval=0)
                        for c in range(3)], axis=2)
    if noise is not None:
        read_var = np.asarray(noise["read_variance_linear"], dtype=np.float64)
        shot = np.asarray(noise["shot_coefficient_linear"], dtype=np.float64)
        if read_var.shape != (3,) or shot.shape != (3,) or np.any(read_var < 0) or np.any(shot < 0):
            raise ValueError("noise requires nonnegative three-channel linear variance coefficients")
        linear = _srgb_to_linear(sampled)
        sigma = np.sqrt(read_var + shot * linear)
        linear += np.random.default_rng(seed).normal(size=linear.shape) * sigma
        sampled = _linear_to_srgb(linear)
    output_rgb = np.rint(np.clip(sampled, 0, 255)).astype(np.uint8)
    output_rgb[~valid] = 0
    ix = np.rint(np.where(valid, sx, 0)).clip(0, w - 1).astype(np.int32)
    iy = np.rint(np.where(valid, sy, 0)).clip(0, h - 1).astype(np.int32)
    output_classes = np.where(valid, classes[iy, ix], 0).astype(classes.dtype)
    output_instances = np.where(valid, instances[iy, ix], 0).astype(instances.dtype)
    output_range = np.where(valid, range_m[iy, ix], np.nan).astype(range_m.dtype)
    return output_rgb, output_classes, output_instances, output_range, valid


def _linear_flat(path: Path) -> np.ndarray:
    if path.suffix != ".npy":
        raise ValueError("flat frames must be lossless NPY arrays in normalized linear units")
    a = np.load(path, allow_pickle=False)
    if a.ndim != 3 or a.shape[2] != 3 or not np.issubdtype(a.dtype, np.floating):
        raise ValueError("flat frame must be HxWx3 floating-point linear RGB")
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError("flat frame values must be finite and in [0,1]")
    return a.astype(np.float64)


def fit_noise(manifest_path: str | Path) -> dict:
    """Photon-transfer variance fit from >=3 matched-exposure flat pairs."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    pairs = manifest["pairs"]
    if len(pairs) < 3:
        raise ValueError("at least three flat intensity levels are required")
    levels, variances, sources = [], [], []
    shape = None
    for entry in pairs:
        if float(entry["exposure_ms_a"]) != float(entry["exposure_ms_b"]):
            raise ValueError("each flat pair must have matched exposures")
        a_path = (manifest_path.parent / entry["a"]).resolve()
        b_path = (manifest_path.parent / entry["b"]).resolve()
        a, b = _linear_flat(a_path), _linear_flat(b_path)
        if a.shape != b.shape or (shape is not None and a.shape != shape):
            raise ValueError("all flat frames must have one HxWx3 shape")
        shape = a.shape
        roi = entry.get("roi_yxyx", [0, shape[0], 0, shape[1]])
        y0, y1, x0, x1 = map(int, roi)
        if not (0 <= y0 < y1 <= shape[0] and 0 <= x0 < x1 <= shape[1]):
            raise ValueError("flat ROI outside image")
        aa, bb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
        if aa.shape[0] * aa.shape[1] < 64:
            raise ValueError("flat ROI needs at least 64 pixels")
        mean = ((aa + bb) / 2).mean(axis=(0, 1))
        difference = aa - bb
        variance = difference.var(axis=(0, 1), ddof=1) / 2
        levels.append(mean)
        variances.append(variance)
        sources.append({"a": str(a_path), "a_sha256": digest(a_path),
                        "b": str(b_path), "b_sha256": digest(b_path),
                        "exposure_ms": float(entry["exposure_ms_a"]), "roi_yxyx": roi})
    x, y = np.asarray(levels), np.asarray(variances)
    if np.any(np.ptp(x, axis=0) < 0.05):
        raise ValueError("flat levels must span at least 0.05 linear units per channel")
    coefficients = np.stack([np.linalg.lstsq(np.c_[np.ones(len(x)), x[:, c]], y[:, c], rcond=None)[0]
                             for c in range(3)])
    if np.any(coefficients < -1e-12):
        raise ValueError("fitted variance coefficients are negative; inspect flats and exposure")
    coefficients = np.maximum(coefficients, 0)
    predicted = coefficients[:, 0][None, :] + x * coefficients[:, 1][None, :]
    return {"model": "variance_linear = read_variance_linear + shot_coefficient_linear * signal_linear",
            "units": "normalized linear RGB; variances in normalized-linear squared units",
            "source_manifest": str(manifest_path.resolve()),
            "source_manifest_sha256": digest(manifest_path),
            "sources": sources,
            "mean_linear_by_level": x.tolist(), "variance_linear_by_level": y.tolist(),
            "read_variance_linear": coefficients[:, 0].tolist(),
            "shot_coefficient_linear": coefficients[:, 1].tolist(),
            "fit_rmse_variance_linear": np.sqrt(np.mean((predicted - y)**2, axis=0)).tolist(),
            "provenance": manifest.get("provenance", "user-supplied flats; provenance unspecified")}


def _camera_K(annotation: dict) -> np.ndarray:
    camera = annotation.get("readback", {}).get("camera") or annotation["sample"]["camera"]
    return _matrix(camera["K"])


def apply_frame(profile_path: Path, frame: Path, output: Path, *, seed: int = 0,
                noise_path: Path | None = None) -> dict:
    profile = ColorProfile.from_json(profile_path)
    annotation = json.loads((frame / "annotation.json").read_text())
    K = _camera_K(annotation)
    with Image.open(frame / "rgb.png") as im:
        rgb = np.asarray(im.convert("RGB"))
    with Image.open(frame / "classes.png") as im:
        classes = np.asarray(im)
    with Image.open(frame / "instances.png") as im:
        instances = np.asarray(im)
    with np.load(frame / "range_m.npz", allow_pickle=False) as data:
        ranges = data["range_m"]
    noise = json.loads(noise_path.read_text()) if noise_path else None
    result = transfer(rgb, classes, instances, ranges, K, profile, seed=seed, noise=noise)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    out_rgb, out_classes, out_instances, out_range, valid = result
    Image.fromarray(out_rgb).save(output / "rgb.png")
    Image.fromarray(out_classes).save(output / "classes.png")
    Image.fromarray(out_instances).save(output / "instances.png")
    Image.fromarray(valid.astype(np.uint8) * 255).save(output / "valid.png")
    np.savez_compressed(output / "range_m.npz", range_m=out_range)
    record = {"profile": profile.name, "profile_provenance": profile.provenance,
              "profile_sha256": digest(profile_path), "source_frame": str(frame.resolve()),
              "source_sha256": {n: digest(frame / n) for n in
                                ("annotation.json", "rgb.png", "classes.png", "instances.png", "range_m.npz")},
              "source_K": K.tolist(), "target_K": profile.K.tolist(), "target_D": profile.D.tolist(),
              "seed": seed, "noise_fit_sha256": digest(noise_path) if noise_path else None,
              "noise_provenance": noise["provenance"] if noise else "disabled",
              "valid_pixel_fraction": float(valid.mean()),
              "range_unit": "radial metres, nearest source sample",
              "source_annotation": annotation}
    (output / "annotation.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    apply = sub.add_parser("apply")
    apply.add_argument("--profile", type=Path, required=True)
    apply.add_argument("--frame", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    apply.add_argument("--seed", type=int, default=0)
    apply.add_argument("--noise-fit", type=Path)
    fit = sub.add_parser("fit-noise")
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "apply":
        report = apply_frame(args.profile, args.frame, args.output, seed=args.seed, noise_path=args.noise_fit)
    else:
        report = fit_noise(args.manifest)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_annotation"}, indent=2))


if __name__ == "__main__":
    main()
