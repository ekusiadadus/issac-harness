"""Prepare an arbitrary OpenUSD stage, render deterministic shards, and audit saved data."""

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from . import campaign


def capture(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="read-only directory containing all external USD and texture dependencies",
    )
    if "--help" in argv:
        parser.print_help()
        runtime = Path(__file__).resolve().parents[1] / "runtime/capture_campaign.py"
        return subprocess.call([sys.executable, str(runtime), "--help"])
    args, rest = parser.parse_known_args(argv)
    source = args.config.resolve(strict=True)
    output = args.out.resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("output must be a directory")
    output.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as f:
        cfg = tomllib.load(f)
    root = Path(__file__).resolve().parents[1]
    inputs = [source.parent, (source.parent / cfg["stage"]).resolve().parent]
    inputs.extend(
        (source.parent / i["asset"]).resolve().parent
        for i in cfg["instances"]
        if "asset" in i
    )
    project = (
        args.asset_root.resolve(strict=True)
        if args.asset_root
        else Path(os.path.commonpath(inputs))
    )
    if project == Path(project.anchor):
        raise ValueError(
            "set --asset-root to the project tree containing your USD dependencies"
        )
    if any(
        not p.is_relative_to(project) and not p.is_relative_to(root) for p in inputs
    ):
        raise ValueError(
            "the selected --asset-root does not include the scene and asset inputs"
        )
    env = {
        **os.environ,
        "HARNESS_OUTPUT_ROOT": str(output),
        "HARNESS_PROJECT_ROOT": str(project),
    }
    return subprocess.call(
        [
            root / "scripts/isaac_python.sh",
            root / "runtime/capture_campaign.py",
            source,
            "--out",
            output,
            *rest,
        ],
        env=env,
    )


def main(argv):
    parser = argparse.ArgumentParser(prog="harness campaign", description=__doc__)
    parser.add_argument(
        "operation",
        choices=[
            "init",
            "plan",
            "survey",
            "provenance",
            "fit-exposure",
            "apply-exposure",
            "capture",
            "run-plan",
            "check",
            "gallery",
            "transfer",
            "sensor",
            "texel-check",
            "replay",
            "paired",
            "label-invariance",
            "release",
        ],
    )
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    operation = parser.parse_args(argv[:1]).operation
    rest = argv[1:]
    if operation == "init":
        from . import author
        return author.main(rest)
    if operation == "sensor":
        from . import sensor_dataset
        return sensor_dataset.main(rest)
    if operation == "texel-check":
        from . import texel_check
        return texel_check.main(rest)
    if operation == "provenance":
        from . import provenance

        return provenance.main(rest)
    if operation == "survey":
        from . import survey

        return survey.main(rest)
    if operation == "fit-exposure":
        from . import exposure_fit

        sys.argv = ["harness campaign fit-exposure", *rest]
        return exposure_fit.main()
    if operation == "apply-exposure":
        from . import holdout

        return holdout.main(rest)
    if operation == "transfer":
        from . import transfer

        return transfer.main(rest)
    if operation == "gallery":
        from . import campaign_gallery

        return campaign_gallery.main(rest)
    if operation == "run-plan":
        from . import shards

        return shards.main(rest)
    if operation == "capture":
        return capture(rest)
    if operation in ("check", "plan"):
        sys.argv = ["harness campaign", operation, *rest]
        return campaign.main()
    if operation == "paired":
        from . import paired

        sys.argv = ["harness campaign paired", *rest]
        return paired.main()
    if operation == "label-invariance":
        from . import label_invariance

        return label_invariance.main(rest)
    from . import qualification

    return qualification.main(operation, rest)
