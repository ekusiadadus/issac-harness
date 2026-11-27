"""Bind a visual review to the actual capture and its reference photograph."""

import json
import re
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from .project import file_hash, inside, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def template(directory, project, run):
    lines = [
        f"# {project.name} の画像レビュー",
        "",
        "画像と実写を原寸で観察し、具体的な差と判定（合格・要修正）を記入する。",
        "",
        "| 条件 | 視点 | PNG SHA-256 | 実写 SHA-256 | 所見・実写との差 | 判定 |",
        "|---|---|---|---|---|---|",
    ]
    for case, data in run["cases"].items():
        for view, frame in data["frames"].items():
            ref = run["references"].get(view, {}).get("sha256", "参照写真なし")
            lines.append(f"| {case} | [{view}]({case}/{view}.png) | {frame['sha256']} | {ref} | | 未確認 |")
    (directory / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def frame_map(directory):
    run = read(directory / "run.json")
    audit = read(directory / "check.json")
    if run.get("status") != "rendered" or audit.get("status") != "pass":
        raise ValueError("capture and scene checks must have completed successfully")
    frames = {}
    for case, result in audit["cases"].items():
        if result["status"] != "pass":
            raise ValueError(f"scene check failed for {case}")
        for view in result["views"]:
            data = run["cases"][case]["frames"][view]
            path = inside(directory, f"{case}/{view}.png")
            if file_hash(path) != data["sha256"]:
                raise ValueError(f"{case}/{view}: captured PNG hash changed")
            if data.get("barcode", {}).get("matches") is False:
                raise ValueError(f"{case}/{view}: barcode check failed")
            frames[case, view] = data
    if not frames:
        raise ValueError("no captured views")
    return run, frames


def accept(directory):
    directory = directory.resolve(strict=True)
    report = {"status": "fail", "issues": [], "frames": []}
    try:
        run, frames = frame_map(directory)
        rows = {}
        review = directory / "review.md"
        report["review_sha256"] = file_hash(review)
        report["run_sha256"] = file_hash(directory / "run.json")
        for line in review.read_text(encoding="utf-8").splitlines():
            if not line.startswith("|") or "](" not in line:
                continue
            columns = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(columns) != 6:
                raise ValueError("malformed review row")
            case, view_link, sha, photo_sha, observation, decision = columns
            m = re.fullmatch(r"\[([\w-]+)\]\(([^)]+)\)", view_link)
            if not m or m[2] != f"{case}/{m[1]}.png" or (case, m[1]) in rows:
                raise ValueError(f"duplicate or malformed review row: {case} {view_link}")
            rows[case, m[1]] = sha, photo_sha, observation, decision
        if rows.keys() != frames.keys():
            raise ValueError("review must cover every captured case and camera exactly once")
        for (case, view), data in frames.items():
            sha, photo_sha, observation, decision = rows[case, view]
            ref = run.get("references", {}).get(view)
            if (
                not ref
                or file_hash(inside(directory, ref["file"])) != photo_sha
                or ref["sha256"] != photo_sha
            ):
                report["issues"].append(f"{case}/{view}: missing or changed reference photograph")
            if sha != data["sha256"] or not observation or decision != "合格":
                report["issues"].append(f"{case}/{view}: {decision} (check observation and frame hash)")
            report["frames"].append(dict(case=case, view=view, observation=observation, decision=decision))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        report["issues"].append(str(exc))
    if not report["issues"]:
        report["status"] = "pass"
    write_json(directory / "acceptance.json", report)
    return report


def capture_conditions(directory, run, case, view):
    renderer = run["cases"][case]["renderer"]
    if run["backend"] != "rtx":
        return renderer
    for manifest in renderer["manifests"]:
        capture = read(inside(directory, f"{case}/{manifest}"))
        for frame in capture["captures"]:
            if frame["name"] == view:
                return dict(
                    versions=capture["versions"],
                    render=capture["render"],
                    camera=frame["camera_prim"],
                    sensor=frame["camera_model"],
                )
    raise ValueError(f"missing RTX capture manifest for {case}/{view}")


def compare(golden, candidate, maximum_rmse):
    """Compare independently captured images under the same sensor and color settings."""
    if not 0 <= maximum_rmse <= 1:
        raise ValueError("maximum RMSE must be between zero and one")
    golden, candidate = golden.resolve(), candidate.resolve()
    if golden == candidate:
        raise ValueError("supply a separate capture directory for the regression")
    a, expected = frame_map(golden)
    b, actual = frame_map(candidate)
    if a["backend"] != b["backend"] or expected.keys() != actual.keys():
        raise ValueError("camera/case sets and rendering backends must match")
    result = {
        "status": "pass",
        "maximum_rmse_srgb": maximum_rmse,
        "golden_run_sha256": file_hash(golden / "run.json"),
        "candidate_run_sha256": file_hash(candidate / "run.json"),
        "frames": [],
    }
    for case, view in expected:
        if capture_conditions(golden, a, case, view) != capture_conditions(candidate, b, case, view):
            raise ValueError(f"{case}/{view}: capture conditions differ")
        with (
            Image.open(golden / case / f"{view}.png") as im1,
            Image.open(candidate / case / f"{view}.png") as im2,
        ):
            if im1.size != im2.size:
                raise ValueError(f"{case}/{view}: image size differs")
            residual = ImageStat.Stat(ImageChops.difference(im1.convert("RGB"), im2.convert("RGB")))
            rmse = (sum(r * r for r in residual.rms) / 3) ** 0.5 / 255
            result["frames"].append(
                dict(
                    case=case,
                    view=view,
                    rmse_srgb=rmse,
                    mae_srgb=sum(residual.mean) / 765,
                    golden_sha256=expected[case, view]["sha256"],
                    candidate_sha256=actual[case, view]["sha256"],
                )
            )
            if rmse > maximum_rmse:
                result["status"] = "fail"
    return result
