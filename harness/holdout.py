"""Apply an observed exposure bracket to held-out poses without reading held-out RGB."""

import argparse
import math
from pathlib import Path

from .campaign import digest, read, write


def apply(poses, freeze, edits, out):
    rows = read(poses)
    frozen = read(freeze)
    gain = frozen["chosen_gain"]
    if not edits or not math.isfinite(gain) or gain <= 0:
        raise ValueError(
            "expected a positive measured gain and named light interventions"
        )
    calibrated = set(frozen["calibration_references"])
    references = [r["reference"] for r in rows]
    if (
        not references
        or len(references) != len(set(references))
        or calibrated.intersection(references)
    ):
        raise ValueError(
            "holdout must have distinct physical references and no calibration references"
        )
    for r in rows:
        r["edits"] = {**r.get("edits", {}), **dict.fromkeys(edits, gain)}
        r["exposure_fit_sha256"] = digest(freeze)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, rows)


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("poses", type=Path)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--edits", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    apply(a.poses, a.freeze, a.edits, a.out)
