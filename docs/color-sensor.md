# Reusable color camera transfer

`harness/color_sensor.py` transfers a pinhole-rendered campaign frame into the pixel grid of an **arbitrary color camera**. The target camera is supplied as JSON with `name`, `provenance`, `width`, `height`, `K`, and `D`. It accepts ROS CameraInfo-style `image_width`, `image_height`, `camera_matrix: {data: [...]}`, and `distortion_coefficients: {data: [...]}` as equivalent fields. `distortion_model` must be `plumb_bob`; `D` is the OpenCV order `[k1, k2, p1, p2, k3]` (four terms are accepted with `k3=0`). Rational, fisheye, chromatic, rolling-shutter, and vignetting models are not silently substituted.

The mapping starts at each distorted target pixel, iteratively inverts [OpenCV's radial and tangential model](https://docs.opencv.org/3.4.10/dc/dbb/tutorial_py_calibration.html), then projects that ideal ray through the source `K` recorded in the campaign frame. RGB uses bilinear interpolation; class IDs, instance IDs, and radial range in metres use the same source coordinates with nearest sampling. Out-of-source or nonconvergent rays become black RGB, zero IDs, NaN range, and zero in `valid.png`. Pixel centers are integer coordinates as in OpenCV. The source camera pose is unchanged; this is an optical transfer, not a reprojection between viewpoints.

The profile in `examples/sensors/color-example/campaign-design-brown.json` is **a design demonstration, not a measured camera**. Its `K` copies the saved renderer readback; its nonzero `D` values are illustrative. Run it on the saved native frame:

```sh
PYTHONPATH=. python -m harness.color_sensor apply \
  --profile examples/sensors/color-example/campaign-design-brown.json \
  --frame evidence/campaign-l4/registered/frames/00000000 \
  --output /tmp/issac-color-sensor-demo --seed 17
```

The command writes `rgb.png`, `classes.png`, `instances.png`, `range_m.npz` (`range_m` key), `valid.png`, and `annotation.json`. The annotation records input hashes, profile hash and provenance, source and target intrinsics, seed, noise source, and valid pixel fraction. It refuses a nonempty output directory. The demonstration produced 640×480 color output with 98.53% valid pixels; source and transformed RGB differ by mean absolute 3.49 code values because of the *designed* distortion. The native frame already has strong bright-region clipping, which this geometric transform cannot recover.

For a **real camera**, replace the JSON with that camera's measured `K` and `D` at the output resolution and document the source, calibration date, and reprojection residual in `provenance`. The example numbers must not be reused as measurements. If a camera image is already rectified, use a zero-distortion profile for the rectified pixel grid.

## Optional photon-transfer noise fit

Provide at least three matched-exposure pairs of **lossless, linear**, normalized `[0,1]` `float32` or `float64` RGB `.npy` flats at different brightness levels. Each pair should photograph the same stationary uniform field with the same exposure and gain. A manifest looks like:

```json
{
  "provenance": "physical rig, acquisition settings and calibration date",
  "pairs": [
    {"a": "level1_a.npy", "b": "level1_b.npy", "exposure_ms_a": 5, "exposure_ms_b": 5, "roi_yxyx": [100, 300, 100, 300]},
    {"a": "level2_a.npy", "b": "level2_b.npy", "exposure_ms_a": 5, "exposure_ms_b": 5},
    {"a": "level3_a.npy", "b": "level3_b.npy", "exposure_ms_a": 5, "exposure_ms_b": 5}
  ]
}
```

```sh
PYTHONPATH=. python -m harness.color_sensor fit-noise --manifest /path/to/flats.json --output /path/to/noise-fit.json
PYTHONPATH=. python -m harness.color_sensor apply --profile /path/to/camera.json --frame /path/to/frame --output /path/to/output --noise-fit /path/to/noise-fit.json --seed 17
```

The fit uses half the pair-difference variance at each level, then fits `variance = read_variance + shot_coefficient × mean_signal` for each channel. It records raw frame SHA-256 hashes, mean/variance samples, residuals, source provenance, and normalized-linear units. Different gains, black levels, exposure settings, nonlinear JPEG or sRGB flats, and changing illumination invalidate the fit. Applying measured coefficients to RTX also assumes its sRGB conversion represents the same normalized linear signal scale; that assumption must be checked with a real paired image. The test suite's known noise curve is **synthetic** and only validates recovery mathematics. TUM's separate monochrome white-paper residual belongs to its own acquisition settings and is not a color profile.

Verification: `PYTHONPATH=. pytest -q tests/test_color_sensor.py` checks exact identity, OpenCV-equation landmarks, bilinear RGB with nearest labels and range, and recovery of a known synthetic noise curve. The example was also run on one saved native frame and its output image inspected. This does not measure a physical color camera.

## 公開されたカラー実機で使う

Freiburg 1 Kinect の RGB 校正を配布元の `ost.txt` から読み取った。640 × 480 の元画像に対応するカメラ行列と 5 係数を保持する。既定の OpenNI 投影行列や、同じ機体の IR カメラとは値が異なる。[プロファイルとアーカイブの SHA](../examples/sensors/tum-rgbd-fr1/profile.json) · [配布元の RGB 校正表](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)

```sh
uv run --locked python -m harness.color_sensor apply \
  --profile examples/sensors/tum-rgbd-fr1/profile.json \
  --frame out/multi/frames/00000000 --output out/fr1/00000000 --seed 17
```

`harness.checkerboard` は同じ解像度で撮影したチェスボードで、既知のレンズ係数と観測点の幾何を検算する。`--pattern 8 6` は内側の交点の列・行で、交点を交互に姿勢推定用と誤差測定用に分ける。配布元の校正写真にもとづく検算には、元のレンズ推定にこれらの写真が用いられた可能性がある。色応答、照度、ノイズの独立測定はこのピクセル誤差に含まれない。[検算結果](../evidence/campaign-l4/sensor-color/calibration-check.json)
