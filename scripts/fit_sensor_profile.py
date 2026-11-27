"""Measure a bounded temporal residual from TUM wide white-paper frames.

This is not the sequence_01 wide-gamma noise calibration: acquisition settings
and JPEG artifacts differ. The result is retained as an independent check.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("examples/sensors/tum_mono_vo_sequence_01/white_paper_validation")
OUT = Path("evidence/sensor-calibration/noise_validation.json")


def main():
    times = [line.split() for line in (ROOT / "times.txt").read_text().splitlines()[:3]]
    exposures = [float(line[2]) for line in times]
    if max(exposures) != min(exposures):
        raise ValueError("white-paper frames have different exposures")
    frames = np.stack([
        np.asarray(Image.open(ROOT / f"{i:05d}.jpg").convert("L"), dtype=np.float64)
        for i in range(3)
    ])
    # Coordinates were selected by inspecting the real image. Both ROIs lie
    # wholly within the visible white sheets, away from their boundaries.
    rois = {"left_paper": [350, 600, 470, 650], "right_paper": [340, 580, 840, 950]}
    results = {}
    for name, (y0, y1, x0, x1) in rois.items():
        patch = frames[:, y0:y1, x0:x1]
        differences = np.diff(patch, axis=0)
        # Remove frame-global brightness drift. Remaining variance includes
        # sensor noise, JPEG, and local paper/camera motion.
        residual = differences - np.median(differences, axis=(1, 2), keepdims=True)
        per_pair = [float(np.std(row) / np.sqrt(2)) for row in residual]
        results[name] = {"roi_yxyx": [y0, y1, x0, x1],
                         "mean_code_by_frame": patch.mean(axis=(1, 2)).tolist(),
                         "temporal_residual_sigma_code_by_pair": per_pair,
                         "median_sigma_code": float(np.median(per_pair))}
    report = {"rig": "TUM calib_wide_whitePaper, IDS uEye monochrome wide lens",
              "exposure_reported_units": exposures,
              "source": "https://cvg.cit.tum.de/data/datasets/mono-dataset",
              "interpretation": "upper-bound temporal residual including JPEG and motion; gamma-disabled calibration, not applied to sequence_01 wide-gamma",
              "results": results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
