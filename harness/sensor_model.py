"""TUM Mono-VO sequence 01 camera model; see docs/sensor-model.md."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class SensorProfile:
    fx: float
    fy: float
    cx: float
    cy: float
    omega: float
    width: int
    height: int
    inverse_response: np.ndarray
    vignette: np.ndarray
    source: str


def load_tum_profile(directory: str | Path) -> SensorProfile:
    directory = Path(directory)
    lines = (directory / "camera.txt").read_text().splitlines()
    fx, fy, cx, cy, omega = (float(v) for v in lines[0].split())
    width, height = (int(v) for v in lines[1].split())
    inverse_response = np.loadtxt(directory / "pcalib.txt", dtype=np.float64)
    with Image.open(directory / "vignette.png") as image:
        vignette = np.asarray(image, dtype=np.float64)
    if inverse_response.shape != (256,) or np.any(np.diff(inverse_response) <= 0):
        raise ValueError("expected a strictly increasing 256-entry inverse response")
    if vignette.shape != (height, width) or vignette.max() <= 0:
        raise ValueError("vignette resolution or values disagree with camera.txt")
    return SensorProfile(fx, fy, cx, cy, omega, width, height,
                         inverse_response, vignette / vignette.max(),
                         "TUM Mono-VO sequence_01, IDS uEye UI-3241LE-M-GL monochrome")


def inverse_fov_map(profile: SensorProfile, source_K: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """For each raw sensor pixel, return its ideal pinhole source pixel."""
    yy, xx = np.indices((profile.height, profile.width), dtype=np.float64)
    xd = ((xx + 0.5) / profile.width - profile.cx) / profile.fx
    yd = ((yy + 0.5) / profile.height - profile.cy) / profile.fy
    rd = np.hypot(xd, yd)
    scale = np.ones_like(rd)
    good = rd > 1e-12
    within_fov = good & (rd * profile.omega < np.pi / 2)
    scale[within_fov] = np.tan(rd[within_fov] * profile.omega) / (2 * rd[within_fov] * np.tan(profile.omega / 2))
    scale[good & ~within_fov] = 0
    xu, yu = xd * scale, yd * scale
    if source_K is None:
        source_x = (xu * profile.fx + profile.cx) * profile.width - 0.5
        source_y = (yu * profile.fy + profile.cy) * profile.height - 0.5
    else:
        K = np.asarray(source_K, dtype=np.float64)
        if K.shape != (3, 3) or not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("source_K must be a valid 3x3 pixel intrinsic matrix")
        source_x = xu * K[0, 0] + K[0, 2] - 0.5
        source_y = yu * K[1, 1] + K[1, 2] - 0.5
    beyond_fov = good & ~within_fov
    return np.where(beyond_fov, np.inf, source_x), np.where(beyond_fov, np.inf, source_y)


def apply_sensor(
    rgb: np.ndarray,
    semantic_ids: np.ndarray,
    distance_m: np.ndarray,
    profile: SensorProfile,
    *,
    exposure_ms: float,
    reference_exposure_ms: float,
    seed: int = 0,
    noise_sigma_linear: float = 0.0,
    source_K: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Warp RGB/labels/range together. Input is ideal pinhole sRGB uint8.

    The LUT is a measured monochrome response. Noise sigma is caller supplied,
    never presented as measured by the TUM dataset.
    """
    h, w = rgb.shape[:2]
    if rgb.shape != (h, w, 3) or rgb.dtype != np.uint8:
        raise ValueError("RGB must be uint8 HxWx3")
    if semantic_ids.shape != (h, w) or distance_m.shape != (h, w):
        raise ValueError("labels and distance must match RGB resolution")
    if source_K is None and (h, w) != (profile.height, profile.width):
        raise ValueError("source_K required unless input is native sensor resolution")
    if exposure_ms <= 0 or reference_exposure_ms <= 0 or noise_sigma_linear < 0:
        raise ValueError("exposure must be positive and noise sigma nonnegative")
    sx, sy = inverse_fov_map(profile, source_K)
    valid = (sx >= 0) & (sy >= 0) & (sx <= w - 1) & (sy <= h - 1)
    coords = [np.where(valid, sy, -1), np.where(valid, sx, -1)]
    # Convert display sRGB to linear radiance; actual RTX tone mapping is not
    # inverted and must be identified by the caller as a modeling limitation.
    srgb = rgb.astype(np.float64) / 255
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    luminance = np.tensordot(linear, [0.2126, 0.7152, 0.0722], axes=([2], [0]))
    warped = map_coordinates(luminance, coords, order=1, mode="constant", cval=0)
    latent = warped * (exposure_ms / reference_exposure_ms) * profile.vignette
    if noise_sigma_linear:
        latent += np.random.default_rng(seed).normal(0, noise_sigma_linear, latent.shape)
    measured_code = np.interp(np.clip(latent, 0, 1) * 255,
                              profile.inverse_response, np.arange(256))
    mono = np.rint(measured_code).clip(0, 255).astype(np.uint8)
    sensor_rgb = np.repeat(mono[:, :, None], 3, axis=2)
    sensor_rgb[~valid] = 0
    nearest_x = np.rint(np.where(valid, sx, 0)).clip(0, w - 1).astype(np.int32)
    nearest_y = np.rint(np.where(valid, sy, 0)).clip(0, h - 1).astype(np.int32)
    ids = np.where(valid, semantic_ids[nearest_y, nearest_x], 0).astype(semantic_ids.dtype)
    distance = np.where(valid, distance_m[nearest_y, nearest_x], np.nan).astype(distance_m.dtype)
    return sensor_rgb, ids, distance, {
        "profile": profile.source,
        "exposure_ms": exposure_ms,
        "reference_exposure_ms": reference_exposure_ms,
        "seed": seed,
        "noise_sigma_linear": noise_sigma_linear,
        "noise_provenance": "caller assumption" if noise_sigma_linear else "disabled; no measured noise profile",
        "valid_pixel_fraction": float(valid.mean()),
        "source_K": np.asarray(source_K).tolist() if source_K is not None else None,
    }
