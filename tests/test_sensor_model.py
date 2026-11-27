import numpy as np
import pytest

from harness.sensor_model import SensorProfile, apply_sensor, inverse_fov_map, load_tum_profile


def test_official_profile_and_geometry():
    profile = load_tum_profile("examples/sensors/tum_mono_vo_sequence_01")
    assert (profile.width, profile.height) == (1280, 1024)
    assert profile.inverse_response[0] == 0
    assert profile.inverse_response[-1] == 255
    x, y = inverse_fov_map(profile)
    assert x.shape == y.shape == profile.vignette.shape
    # An off-axis raw pixel maps farther from the principal point in ideal coordinates.
    assert x[512, 1100] > 1100


def test_registered_outputs_and_determinism():
    h, w = 32, 40
    profile = SensorProfile(0.35, 0.44, 0.5, 0.5, 0.93, w, h,
                            np.arange(256, dtype=float), np.ones((h, w)), "test")
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    ids = np.zeros((h, w), dtype=np.uint32)
    ids[:, 20:] = 7
    depth = np.where(ids == 7, 2.0, 1.0).astype(np.float32)
    a = apply_sensor(rgb, ids, depth, profile, exposure_ms=1, reference_exposure_ms=1,
                     noise_sigma_linear=0.01, seed=33)
    b = apply_sensor(rgb, ids, depth, profile, exposure_ms=1, reference_exposure_ms=1,
                     noise_sigma_linear=0.01, seed=33)
    assert np.array_equal(a[0], b[0])
    assert a[0].shape == rgb.shape and a[0].dtype == np.uint8
    assert a[1].dtype == ids.dtype and a[2].dtype == depth.dtype
    assert np.all(a[2][a[1] == 7] == 2)
    assert np.all(a[2][a[1] == 0] == 1) or np.isnan(a[2][a[1] == 0]).any()
    with pytest.raises(ValueError):
        apply_sensor(rgb, ids, depth, profile, exposure_ms=0, reference_exposure_ms=1)


def test_different_source_intrinsics_preserve_mask_and_range_registration():
    profile = SensorProfile(0.35, 0.44, 0.5, 0.5, 0.93, 40, 32,
                            np.arange(256, dtype=float), np.ones((32, 40)), "test")
    rgb = np.full((24, 64, 3), 200, dtype=np.uint8)
    ids = np.zeros((24, 64), dtype=np.uint16)
    ids[:, 32:] = 4
    depth = np.where(ids == 4, 2.5, 1.25).astype(np.float32)
    K = np.array([[40, 0, 32], [0, 40, 12], [0, 0, 1]], dtype=float)
    out_rgb, out_ids, out_depth, meta = apply_sensor(
        rgb, ids, depth, profile, source_K=K, exposure_ms=1, reference_exposure_ms=1)
    assert out_rgb.shape == (32, 40, 3)
    assert np.all(out_depth[out_ids == 4] == 2.5)
    assert np.all(out_depth[(out_ids == 0) & np.isfinite(out_depth)] == 1.25)
    assert 0 < meta["valid_pixel_fraction"] < 1
