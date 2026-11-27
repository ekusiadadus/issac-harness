"""Count the proposed camera, layout, semantic and appearance coverage before launching Kit."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .campaign import digest, load, prescribed, sample, write


def survey(path, frames, seed):
    root, cfg = load(path)
    cases = prescribed(root, cfg)
    if frames < 1 or seed < 0 or cases is not None and frames > len(cases):
        raise ValueError("invalid survey range")
    strata = defaultdict(Counter)
    choices = defaultdict(Counter)
    shown = Counter()
    continuous = {}

    def observe(key, value):
        if key not in continuous:
            continuous[key] = [value, value]
        else:
            continuous[key] = [
                min(continuous[key][0], value),
                max(continuous[key][1], value),
            ]

    for i in range(frames):
        row = sample(cfg, cases, seed, i)
        cam = row["camera"]
        strata[cam["id"]][row["layout"]] += 1
        for k in ("azimuth_deg", "elevation_deg"):
            if k in cam:
                observe(k, cam[k])
        if "eye_m" in cam:
            observe(
                "camera_distance_m",
                float(np.linalg.norm(np.asarray(cam["eye_m"]) - cam["target_m"])),
            )
        observe("focal_x_px", cam["K"][0])
        for k, v in row["edits"].items():
            if type(v) in (int, float):
                observe(k, v)
            else:
                choices[k][str(v)] += 1
        for k, obj in row["objects"].items():
            if obj["visible"]:
                shown[k] += 1
                if "yaw_deg" in obj:
                    observe(k + "_yaw_deg", obj["yaw_deg"])
    return {
        "status": "sampled_configuration",
        "rendered_frames": 0,
        "draws": frames,
        "seed": seed,
        "config_sha256": digest(path),
        "camera_by_layout": strata,
        "categorical_edits": choices,
        "ranges": continuous,
        "instances_requested": shown,
    }


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    data = survey(a.config, a.frames, a.seed)
    a.out.parent.mkdir(exist_ok=True, parents=True)
    write(a.out, data)
