import json

import numpy as np

from harness.color_sensor import ColorProfile, distort_normalized, fit_noise, source_map, transfer


def test_identity_color_and_registered_planes():
    h, w = 32, 40
    K = np.array([[30., 0, 20], [0, 30., 16], [0, 0, 1]])
    profile = ColorProfile("identity", "design test", w, h, K, np.zeros(5))
    yy, xx = np.indices((h, w))
    rgb = np.stack([xx.astype(np.uint8), yy.astype(np.uint8), (xx + yy).astype(np.uint8)], axis=2)
    classes = (xx >= 20).astype(np.uint16)
    instances = np.where(classes, 7, 0).astype(np.uint16)
    radial = np.where(classes, 2.0, 1.0).astype(np.float32)
    out = transfer(rgb, classes, instances, radial, K, profile)
    for actual, expected in zip(out[:4], (rgb, classes, instances, radial)):
        assert np.array_equal(actual, expected)
    assert out[4].all()
    assert len(np.unique(out[0].reshape(-1, 3), axis=0)) > 1  # color retained


def test_opencv_brown_conrady_analytic_landmarks():
    # OpenCV's published k1,k2,p1,p2,k3 equations: project known ideal rays,
    # then recover their locations through the target inverse map.
    D = np.array([-0.12, 0.025, 0.002, -0.001, 0.003])
    K = np.array([[100., 0, 80], [0, 110., 60], [0, 0, 1]])
    profile = ColorProfile("analytic", "design test", 160, 120, K, D)
    sx, sy, valid = source_map(profile, K, (120, 160))
    for x, y in [(0., 0.), (0.25, -0.18), (-0.32, 0.2), (0.4, 0.3)]:
        xd, yd = distort_normalized(np.array(x), np.array(y), D)
        u, v = 100 * xd + 80, 110 * yd + 60
        ui, vi = int(round(u)), int(round(v))
        # The map is sampled on integer pixels; account for subpixel rounding
        # by independently solving the OpenCV forward equation at recovered ray.
        recovered_x = (sx[vi, ui] - 80) / 100
        recovered_y = (sy[vi, ui] - 60) / 110
        check_x, check_y = distort_normalized(np.array(recovered_x), np.array(recovered_y), D)
        assert valid[vi, ui]
        assert abs(100 * check_x + 80 - ui) < 1e-3
        assert abs(110 * check_y + 60 - vi) < 1e-3
        assert np.hypot(recovered_x - x, recovered_y - y) < 0.008


def test_rgb_bilinear_but_ids_and_radial_range_nearest():
    source_K = np.array([[10., 0, 2], [0, 10., 2], [0, 0, 1]])
    target_K = source_K.copy()
    target_K[0, 2] -= 0.5  # target pixel 2 samples source x=2.5
    profile = ColorProfile("half-pixel", "design test", 4, 4, target_K, np.zeros(5))
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, 2, :] = [20, 40, 60]
    rgb[:, 3, :] = [100, 120, 140]
    classes = np.tile(np.array([0, 0, 1, 2], dtype=np.uint8), (4, 1))
    instances = classes.astype(np.uint16) + 10
    radial = (classes.astype(np.float32) + 1) * 0.5
    out = transfer(rgb, classes, instances, radial, source_K, profile)
    assert np.array_equal(out[0][2, 2], [60, 80, 100])
    assert out[1][2, 2] in (1, 2)  # nearest tie convention
    assert out[2][2, 2] == out[1][2, 2] + 10
    assert out[3][2, 2] == (out[1][2, 2] + 1) * 0.5


def test_synthetic_photon_transfer_recovers_known_curve(tmp_path):
    rng = np.random.default_rng(38)
    read_var = np.array([2e-5, 3e-5, 4e-5])
    shot = np.array([5e-4, 7e-4, 9e-4])
    pairs = []
    for i, level in enumerate([0.1, 0.25, 0.4, 0.55, 0.7, 0.85]):
        mean = np.array([level, level * 0.9, level * 0.8])
        sigma = np.sqrt(read_var + shot * mean)
        for suffix in ("a", "b"):
            frame = mean + rng.normal(size=(160, 160, 3)) * sigma
            np.save(tmp_path / f"{i}_{suffix}.npy", frame.astype(np.float32))
        pairs.append({"a": f"{i}_a.npy", "b": f"{i}_b.npy",
                      "exposure_ms_a": 5.0, "exposure_ms_b": 5.0})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"provenance": "SYNTHETIC known curve", "pairs": pairs}))
    fitted = fit_noise(manifest)
    assert np.allclose(fitted["read_variance_linear"], read_var, atol=4e-6)
    assert np.allclose(fitted["shot_coefficient_linear"], shot, rtol=0.1)
    assert len(fitted["sources"]) == 6


def test_ros_camera_info_json_and_seeded_noise(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps({
        "name": "design fixture", "provenance": "SYNTHETIC",
        "image_width": 20, "image_height": 16, "distortion_model": "plumb_bob",
        "camera_matrix": {"data": [18, 0, 10, 0, 18, 8, 0, 0, 1]},
        "distortion_coefficients": {"data": [0, 0, 0, 0, 0]},
    }))
    profile = ColorProfile.from_json(path)
    rgb = np.full((16, 20, 3), [90, 120, 150], dtype=np.uint8)
    ids = np.ones((16, 20), dtype=np.uint8)
    ranges = np.full((16, 20), 1.0, dtype=np.float32)
    noise = {"read_variance_linear": [1e-4] * 3,
             "shot_coefficient_linear": [1e-3] * 3}
    a = transfer(rgb, ids, ids, ranges, profile.K, profile, seed=47, noise=noise)
    b = transfer(rgb, ids, ids, ranges, profile.K, profile, seed=47, noise=noise)
    c = transfer(rgb, ids, ids, ranges, profile.K, profile, seed=48, noise=noise)
    assert np.array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0])
    assert np.array_equal(a[1], ids) and np.array_equal(a[3], ranges)
