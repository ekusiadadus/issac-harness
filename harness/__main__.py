import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def main():
    if sys.argv[1:2] == ["campaign"]:
        from .campaign_cli import main as campaign_cli
        return campaign_cli(sys.argv[2:])
    from .audit import audit
    from .dataset import dataset_check, export_training, gallery as dataset_gallery
    from .nvidia import simready, validate
    from .project import load_project, write_json
    from .render import ROOT, render
    from .review import accept, compare

    parser = argparse.ArgumentParser(description="Inspect and render portable USD environments.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("campaign", help="Create, capture and qualify a dataset for your own USD scene")
    starter = commands.add_parser("init", help="Copy the complete scene package to a new directory")
    starter.add_argument("destination", type=Path)
    gate = commands.add_parser("accept", help="Verify completed visual decisions and photograph/frame hashes")
    gate.add_argument("directory", type=Path)
    diff = commands.add_parser(
        "compare", help="Measure a new capture against an independently saved golden run"
    )
    diff.add_argument("golden", type=Path)
    diff.add_argument("candidate", type=Path)
    diff.add_argument(
        "--max-rmse", type=float, default=0.025, help="Maximum per-frame RGB RMSE in [0,1] sRGB"
    )
    diff.add_argument("--report", required=True, type=Path)
    profile = commands.add_parser(
        "simready", help="Run the NVIDIA Robotics-Prop profile on individual assets"
    )
    profile.add_argument("assets", type=Path, nargs="+")
    profile.add_argument("--out", type=Path, required=True)
    pilot = commands.add_parser(
        "dataset-check", help="Verify saved RGB, masks, boxes and depth; optionally check a seeded replay"
    )
    pilot.add_argument("directory", type=Path)
    pilot.add_argument("--replay", type=Path)
    pilot.add_argument("--report", type=Path, required=True)
    pilot.add_argument("--gallery", type=Path, help="New directory for the aligned RGB, mask and range preview")
    pilot.add_argument("--training", type=Path, help="New directory for stable-class training masks and labels")
    check = commands.add_parser(
        "check", help="Check composition, materials, physical contacts and camera framing"
    )
    check.add_argument("--report", type=Path, help="Write the machine-readable audit")
    check.add_argument(
        "--nvidia",
        type=Path,
        metavar="NEW_DIR",
        help="Also run NVIDIA's default OpenUSD validator; retain JSON and logs here",
    )
    draw = commands.add_parser("render", help="Render the selected cases and create a local review gallery")
    draw.add_argument("--backend", choices=("cycles", "storm", "rtx"), default="cycles")
    draw.add_argument("--out", type=Path, help="New output directory (default: timestamped under out/)")
    draw.add_argument(
        "--capture-only", action="store_true", help="Finish after capture; review separately with accept"
    )
    draw.add_argument("--width", type=positive, help="Pixel width; height follows the USD camera gate")
    draw.add_argument(
        "--samples", type=positive, default=64, help="Cycles samples or RTX samples per pixel per frame"
    )
    draw.add_argument(
        "--isaac-python", type=Path, help="Isaac Sim python.sh or its installed interpreter for RTX"
    )
    for child in [check, draw]:
        child.add_argument("project", type=Path, help="Scene directory or scene.toml")
        child.add_argument("--case", dest="cases", action="append", help="Only this case (repeatable)")
    args = parser.parse_args()
    try:
        if args.command == "dataset-check":
            result = dataset_check(args.directory, args.replay)
            if args.gallery:
                dataset_gallery(args.directory, args.gallery)
            if args.training and result["status"] == "pass":
                export_training(args.directory, args.training)
            write_json(args.report, result)
            print(args.report)
            return int(result["status"] != "pass")
        if args.command == "accept":
            result = accept(args.directory)
            for issue in result["issues"]:
                print(issue, file=sys.stderr)
            print(args.directory / "acceptance.json")
            return int(result["status"] != "pass")
        if args.command == "compare":
            result = compare(args.golden, args.candidate, args.max_rmse)
            write_json(args.report, result)
            print(args.report)
            return int(result["status"] != "pass")
        if args.command == "simready":
            return int(not simready(args.assets, args.out))
        if args.command == "init":
            template = ROOT / "examples/inspection"
            dest = args.destination.resolve()
            if dest.is_relative_to(template):
                raise ValueError("Create the project outside the example directory")
            shutil.copytree(template, dest, ignore=shutil.ignore_patterns(".DS_Store", "__pycache__"))
            print(dest / "scene.toml")
            return 0
        project = load_project(args.project)
        report = audit(project, args.cases)
        if args.command == "check" and args.nvidia:
            validate(project, report, args.nvidia)
        for name, case in report["cases"].items():
            print(
                f"{case['status'].upper()} {name} · {case['visible_gprims']} visible prims · "
                f"{len(case['materials'])} materials · {len(case['views'])} views"
            )
            for error in case["errors"]:
                print(f"  {error}", file=sys.stderr)
        if args.command == "check":
            if args.report:
                write_json(args.report, report)
                print(args.report.resolve())
            return 0 if report["status"] == "pass" else 1
        default = Path("out") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        gallery = render(
            project,
            report,
            args.out or default,
            args.backend,
            args.samples,
            args.width,
            args.isaac_python,
        )
        print(gallery)
        if not args.capture_only:
            result = accept(gallery.parent)
            if result["status"] != "pass":
                print(
                    f"Visual review pending. Fill in {gallery.parent / 'review.md'} and run accept.",
                    file=sys.stderr,
                )
                return 1
    except (OSError, RuntimeError, TypeError, ValueError, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc or 'interrupted'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
